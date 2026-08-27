#include "spiral/media_generation.hpp"

#include "spiral/dsp.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <numbers>
#include <stdexcept>

namespace spiral::media_generation {
namespace {

void append_parameters(std::vector<nn::Parameter*>& out, std::vector<nn::Parameter*> values) {
    out.insert(out.end(), values.begin(), values.end());
}

void append_parameters(std::vector<const nn::Parameter*>& out, std::vector<const nn::Parameter*> values) {
    out.insert(out.end(), values.begin(), values.end());
}

void require_matrix(const Tensor& value, std::size_t features, const char* name) {
    if (value.rank() != 2 || value.shape()[0] == 0 || value.shape()[1] != features) {
        throw std::invalid_argument(std::string(name) + " requires [rows,features]");
    }
}

float mse(const Tensor& prediction, const Tensor& target) {
    if (prediction.shape() != target.shape() || prediction.numel() == 0) {
        throw std::invalid_argument("MSE requires equal non-empty tensors");
    }
    double total = 0.0;
    for (std::size_t i = 0; i < prediction.numel(); ++i) {
        const double delta = static_cast<double>(prediction.data()[i] - target.data()[i]);
        total += delta * delta;
    }
    return static_cast<float>(total / static_cast<double>(prediction.numel()));
}

Tensor mse_gradient(const Tensor& prediction, const Tensor& target) {
    if (prediction.shape() != target.shape() || prediction.numel() == 0) {
        throw std::invalid_argument("MSE gradient requires equal non-empty tensors");
    }
    Tensor grad(prediction.shape());
    const float scale = 2.0F / static_cast<float>(prediction.numel());
    for (std::size_t i = 0; i < prediction.numel(); ++i) {
        grad.data()[i] = scale * (prediction.data()[i] - target.data()[i]);
    }
    return grad;
}

Tensor linear_backward(nn::Linear& layer, const Tensor& input, const Tensor& grad_output) {
    Tensor matrix = input;
    Tensor grad_matrix = grad_output;
    const bool collapse = input.rank() == 1;
    if (collapse) {
        matrix = Tensor({1, input.shape()[0]}, input.data());
        grad_matrix = Tensor({1, grad_output.shape()[0]}, grad_output.data());
    }
    if (matrix.rank() != 2 || grad_matrix.rank() != 2 || matrix.shape()[0] != grad_matrix.shape()[0] ||
        matrix.shape()[1] != layer.in_features() || grad_matrix.shape()[1] != layer.out_features()) {
        throw std::invalid_argument("linear backward shape mismatch");
    }

    auto& weight = layer.weight();
    weight.ensure_grad();
    nn::Parameter* bias = nullptr;
    if (layer.uses_bias()) {
        bias = &layer.bias();
        bias->ensure_grad();
    }

    Tensor grad_input({matrix.shape()[0], layer.in_features()});
    for (std::size_t row = 0; row < matrix.shape()[0]; ++row) {
        for (std::size_t out = 0; out < layer.out_features(); ++out) {
            const float g = grad_matrix.data()[row * layer.out_features() + out];
            if (bias != nullptr) bias->grad.data()[out] += g;
            for (std::size_t in = 0; in < layer.in_features(); ++in) {
                weight.grad.data()[in * layer.out_features() + out] += matrix.data()[row * layer.in_features() + in] * g;
                grad_input.data()[row * layer.in_features() + in] += weight.value.data()[in * layer.out_features() + out] * g;
            }
        }
    }
    if (collapse) return Tensor({layer.in_features()}, grad_input.data());
    return grad_input;
}

Tensor silu(const Tensor& input) {
    Tensor out(input.shape());
    for (std::size_t i = 0; i < input.numel(); ++i) {
        const float x = input.data()[i];
        const float s = 1.0F / (1.0F + std::exp(-x));
        out.data()[i] = x * s;
    }
    return out;
}

Tensor silu_backward(const Tensor& input, const Tensor& grad_output) {
    Tensor out(input.shape());
    for (std::size_t i = 0; i < input.numel(); ++i) {
        const float x = input.data()[i];
        const float s = 1.0F / (1.0F + std::exp(-x));
        out.data()[i] = grad_output.data()[i] * s * (1.0F + x * (1.0F - s));
    }
    return out;
}

Tensor sigmoid(const Tensor& input) {
    Tensor out(input.shape());
    for (std::size_t i = 0; i < input.numel(); ++i) out.data()[i] = 1.0F / (1.0F + std::exp(-input.data()[i]));
    return out;
}

Tensor sigmoid_backward(const Tensor& activated, const Tensor& grad_output) {
    Tensor out(activated.shape());
    for (std::size_t i = 0; i < activated.numel(); ++i) {
        const float y = activated.data()[i];
        out.data()[i] = grad_output.data()[i] * y * (1.0F - y);
    }
    return out;
}

Tensor image_target_row(const vision::RgbImage& image) {
    const Tensor source = vision::image_to_tensor(image);
    return Tensor({1, source.numel()}, source.data());
}

Tensor embedding_row(const Tensor& embedding) {
    if (embedding.rank() != 1 || embedding.shape()[0] == 0) throw std::invalid_argument("embedding must be rank-1");
    return Tensor({1, embedding.shape()[0]}, embedding.data());
}

} // namespace

Tensor complex_stft(const audio::AudioBuffer& input, ComplexStftConfig config) {
    if (!dsp::is_power_of_two(config.frame_size) || config.hop_size == 0 || config.hop_size > config.frame_size) {
        throw std::invalid_argument("complex STFT requires power-of-two frame size and valid hop");
    }
    const audio::AudioBuffer mono = input.mono();
    if (mono.frame_count() == 0) throw std::invalid_argument("complex STFT requires non-empty audio");
    const std::size_t frames = mono.frame_count() <= config.frame_size
        ? 1
        : 1 + (mono.frame_count() - config.frame_size + config.hop_size - 1) / config.hop_size;
    const std::size_t bins = config.frame_size / 2 + 1;
    Tensor spectrum({frames, bins, 2});
    const float denom = static_cast<float>(config.frame_size - 1);

    for (std::size_t frame = 0; frame < frames; ++frame) {
        std::vector<std::complex<float>> values(config.frame_size);
        const std::size_t start = frame * config.hop_size;
        for (std::size_t i = 0; i < config.frame_size; ++i) {
            const std::size_t index = start + i;
            const float sample = index < mono.frame_count() ? mono.sample(index, 0) : 0.0F;
            const float hann = 0.5F - 0.5F * std::cos(2.0F * std::numbers::pi_v<float> * static_cast<float>(i) / denom);
            values[i] = {sample * hann, 0.0F};
        }
        dsp::fft_inplace(values, false);
        for (std::size_t bin = 0; bin < bins; ++bin) {
            const std::size_t base = (frame * bins + bin) * 2;
            spectrum.data()[base] = values[bin].real();
            spectrum.data()[base + 1] = values[bin].imag();
        }
    }
    return spectrum;
}

audio::AudioBuffer inverse_complex_stft(const Tensor& spectrum, std::uint32_t sample_rate, ComplexStftConfig config) {
    if (spectrum.rank() != 3 || spectrum.shape()[0] == 0 || spectrum.shape()[2] != 2 || sample_rate == 0 ||
        !dsp::is_power_of_two(config.frame_size) || config.hop_size == 0 || config.hop_size > config.frame_size) {
        throw std::invalid_argument("invalid inverse complex STFT input");
    }
    const std::size_t bins = config.frame_size / 2 + 1;
    if (spectrum.shape()[1] != bins) throw std::invalid_argument("complex STFT bin count mismatch");
    const std::size_t frames = spectrum.shape()[0];
    const std::size_t output_size = (frames - 1) * config.hop_size + config.frame_size;
    std::vector<float> output(output_size, 0.0F);
    std::vector<float> normalization(output_size, 0.0F);
    const float denom = static_cast<float>(config.frame_size - 1);

    for (std::size_t frame = 0; frame < frames; ++frame) {
        std::vector<std::complex<float>> values(config.frame_size);
        for (std::size_t bin = 0; bin < bins; ++bin) {
            const std::size_t base = (frame * bins + bin) * 2;
            values[bin] = {spectrum.data()[base], spectrum.data()[base + 1]};
        }
        for (std::size_t bin = 1; bin + 1 < bins; ++bin) values[config.frame_size - bin] = std::conj(values[bin]);
        dsp::fft_inplace(values, true);
        const std::size_t start = frame * config.hop_size;
        for (std::size_t i = 0; i < config.frame_size; ++i) {
            const float hann = 0.5F - 0.5F * std::cos(2.0F * std::numbers::pi_v<float> * static_cast<float>(i) / denom);
            output[start + i] += values[i].real() * hann;
            normalization[start + i] += hann * hann;
        }
    }
    for (std::size_t i = 0; i < output.size(); ++i) {
        if (normalization[i] > 1.0e-8F) output[i] /= normalization[i];
        output[i] = std::clamp(output[i], -1.0F, 1.0F);
    }
    return audio::AudioBuffer(sample_rate, 1, std::move(output));
}

Tensor complex_spectral_patches(const Tensor& spectrum, std::size_t frames_per_patch) {
    if (spectrum.rank() != 3 || spectrum.shape()[0] == 0 || spectrum.shape()[1] == 0 || spectrum.shape()[2] != 2 || frames_per_patch == 0) {
        throw std::invalid_argument("complex patches require [frames,bins,2] and positive frames_per_patch");
    }
    const std::size_t frames = spectrum.shape()[0];
    const std::size_t bins = spectrum.shape()[1];
    const std::size_t patch_count = (frames + frames_per_patch - 1) / frames_per_patch;
    const std::size_t patch_dim = frames_per_patch * bins * 2;
    Tensor patches({patch_count, patch_dim});
    for (std::size_t patch = 0; patch < patch_count; ++patch) {
        for (std::size_t local = 0; local < frames_per_patch; ++local) {
            const std::size_t frame = patch * frames_per_patch + local;
            if (frame >= frames) continue;
            const std::size_t src = frame * bins * 2;
            const std::size_t dst = patch * patch_dim + local * bins * 2;
            std::copy_n(spectrum.data().begin() + static_cast<std::ptrdiff_t>(src), bins * 2,
                        patches.data().begin() + static_cast<std::ptrdiff_t>(dst));
        }
    }
    return patches;
}

Tensor complex_patches_to_spectrum(const Tensor& patches, std::size_t frame_count, std::size_t bin_count, std::size_t frames_per_patch) {
    if (patches.rank() != 2 || frame_count == 0 || bin_count == 0 || frames_per_patch == 0 ||
        patches.shape()[1] != frames_per_patch * bin_count * 2 ||
        patches.shape()[0] < (frame_count + frames_per_patch - 1) / frames_per_patch) {
        throw std::invalid_argument("complex patch geometry mismatch");
    }
    Tensor spectrum({frame_count, bin_count, 2});
    const std::size_t patch_dim = frames_per_patch * bin_count * 2;
    for (std::size_t frame = 0; frame < frame_count; ++frame) {
        const std::size_t patch = frame / frames_per_patch;
        const std::size_t local = frame % frames_per_patch;
        const std::size_t src = patch * patch_dim + local * bin_count * 2;
        std::copy_n(patches.data().begin() + static_cast<std::ptrdiff_t>(src), bin_count * 2,
                    spectrum.data().begin() + static_cast<std::ptrdiff_t>(frame * bin_count * 2));
    }
    return spectrum;
}

ComplexAudioCodec::ComplexAudioCodec(ComplexAudioCodecConfig config, Random& rng)
    : config_(config), encoder_(config.patch_dim, config.latent_dim, rng, true), decoder_(config.latent_dim, config.patch_dim, rng, true) {
    if (config_.patch_dim == 0 || config_.latent_dim == 0) throw std::invalid_argument("invalid complex audio codec dimensions");
}

Tensor ComplexAudioCodec::encode(const Tensor& complex_patches) const {
    require_matrix(complex_patches, config_.patch_dim, "ComplexAudioCodec::encode");
    return encoder_.forward(complex_patches);
}

Tensor ComplexAudioCodec::decode(const Tensor& latents) const {
    require_matrix(latents, config_.latent_dim, "ComplexAudioCodec::decode");
    return decoder_.forward(latents);
}

Tensor ComplexAudioCodec::reconstruct(const Tensor& complex_patches) const { return decode(encode(complex_patches)); }

std::vector<nn::Parameter*> ComplexAudioCodec::parameters() {
    std::vector<nn::Parameter*> out; append_parameters(out, encoder_.parameters()); append_parameters(out, decoder_.parameters()); return out;
}

std::vector<const nn::Parameter*> ComplexAudioCodec::parameters() const {
    std::vector<const nn::Parameter*> out; append_parameters(out, encoder_.parameters()); append_parameters(out, decoder_.parameters()); return out;
}

ComplexAudioCodecTrainer::ComplexAudioCodecTrainer(ComplexAudioCodec& codec, MediaTrainerConfig config)
    : codec_(codec), config_(config), optimizer_(codec.parameters(), config.optimizer) {}

float ComplexAudioCodecTrainer::evaluate(const Tensor& patches) const { return mse(codec_.reconstruct(patches), patches); }

float ComplexAudioCodecTrainer::train_step(const Tensor& patches) {
    optimizer_.zero_grad();
    const Tensor latent = codec_.encoder().forward(patches);
    const Tensor reconstruction = codec_.decoder().forward(latent);
    const float loss = mse(reconstruction, patches);
    const Tensor grad_reconstruction = mse_gradient(reconstruction, patches);
    const Tensor grad_latent = linear_backward(codec_.decoder(), latent, grad_reconstruction);
    (void)linear_backward(codec_.encoder(), patches, grad_latent);
    auto params = codec_.parameters();
    train::clip_grad_norm(params, config_.max_grad_norm);
    optimizer_.step();
    return loss;
}

FrameEmbeddingDecoder::FrameEmbeddingDecoder(FrameDecoderConfig config, Random& rng)
    : config_(config), hidden_(config.embedding_dim, config.hidden_dim, rng, true), output_(config.hidden_dim, config.width * config.height * 3, rng, true) {
    if (config_.embedding_dim == 0 || config_.width == 0 || config_.height == 0 || config_.hidden_dim == 0) {
        throw std::invalid_argument("invalid frame decoder config");
    }
}

Tensor FrameEmbeddingDecoder::decode_tensor(const Tensor& embedding) const {
    if (embedding.rank() != 1 || embedding.shape()[0] != config_.embedding_dim) throw std::invalid_argument("frame embedding dimension mismatch");
    const Tensor row = embedding_row(embedding);
    const Tensor hidden = silu(hidden_.forward(row));
    const Tensor pixels = sigmoid(output_.forward(hidden));
    return Tensor({config_.height, config_.width, 3}, pixels.data());
}

vision::RgbImage FrameEmbeddingDecoder::decode(const Tensor& embedding) const { return vision::tensor_to_image(decode_tensor(embedding)); }

std::vector<nn::Parameter*> FrameEmbeddingDecoder::parameters() {
    std::vector<nn::Parameter*> out; append_parameters(out, hidden_.parameters()); append_parameters(out, output_.parameters()); return out;
}

std::vector<const nn::Parameter*> FrameEmbeddingDecoder::parameters() const {
    std::vector<const nn::Parameter*> out; append_parameters(out, hidden_.parameters()); append_parameters(out, output_.parameters()); return out;
}

FrameDecoderTrainer::FrameDecoderTrainer(FrameEmbeddingDecoder& decoder, const vision::VisionEncoder& encoder, MediaTrainerConfig config)
    : decoder_(decoder), encoder_(encoder), config_(config), optimizer_(decoder.parameters(), config.optimizer) {
    if (encoder.config().embedding_dim != decoder.config().embedding_dim) throw std::invalid_argument("vision/frame embedding dimension mismatch");
}

float FrameDecoderTrainer::evaluate(const vision::RgbImage& frame) const {
    if (frame.width() != decoder_.config().width || frame.height() != decoder_.config().height) throw std::invalid_argument("frame dimensions mismatch");
    const Tensor prediction = decoder_.decode_tensor(encoder_.encode_pooled(frame));
    return mse(prediction, vision::image_to_tensor(frame));
}

float FrameDecoderTrainer::train_step(const vision::RgbImage& frame) {
    if (frame.width() != decoder_.config().width || frame.height() != decoder_.config().height) throw std::invalid_argument("frame dimensions mismatch");
    optimizer_.zero_grad();
    const Tensor embedding = embedding_row(encoder_.encode_pooled(frame));
    const Tensor hidden_pre = decoder_.hidden().forward(embedding);
    const Tensor hidden = silu(hidden_pre);
    const Tensor output_pre = decoder_.output().forward(hidden);
    const Tensor pixels = sigmoid(output_pre);
    const Tensor target = image_target_row(frame);
    const float loss = mse(pixels, target);
    const Tensor grad_pixels = mse_gradient(pixels, target);
    const Tensor grad_output_pre = sigmoid_backward(pixels, grad_pixels);
    const Tensor grad_hidden = linear_backward(decoder_.output(), hidden, grad_output_pre);
    const Tensor grad_hidden_pre = silu_backward(hidden_pre, grad_hidden);
    (void)linear_backward(decoder_.hidden(), embedding, grad_hidden_pre);
    auto params = decoder_.parameters();
    train::clip_grad_norm(params, config_.max_grad_norm);
    optimizer_.step();
    return loss;
}

AudioMediaGenerator::AudioMediaGenerator(
    ComplexStftConfig stft,
    std::size_t frames_per_patch,
    const ComplexAudioCodec& codec,
    const temporal_generation::CausalTemporalPredictor& predictor)
    : stft_(stft), frames_per_patch_(frames_per_patch), codec_(&codec), predictor_(&predictor) {
    if (frames_per_patch_ == 0) throw std::invalid_argument("frames_per_patch must be non-zero");
    if (predictor.config().input_dim != codec.config().latent_dim || predictor.config().output_dim != codec.config().latent_dim) {
        throw std::invalid_argument("audio predictor latent dimension mismatch");
    }
}

Tensor AudioMediaGenerator::encode_latents(const audio::AudioBuffer& source) const {
    const Tensor spectrum = complex_stft(source, stft_);
    return codec_->encode(complex_spectral_patches(spectrum, frames_per_patch_));
}

Tensor AudioMediaGenerator::continue_latents(const audio::AudioBuffer& seed, std::size_t steps) const {
    return predictor_->generate(encode_latents(seed), steps);
}

audio::AudioBuffer AudioMediaGenerator::decode_latents(
    const Tensor& latents,
    std::uint32_t sample_rate,
    std::size_t frame_count) const {
    const Tensor patches = codec_->decode(latents);
    const std::size_t bins = stft_.frame_size / 2 + 1;
    const Tensor spectrum = complex_patches_to_spectrum(patches, frame_count, bins, frames_per_patch_);
    return inverse_complex_stft(spectrum, sample_rate, stft_);
}

VideoMediaGenerator::VideoMediaGenerator(
    const vision::VisionEncoder& encoder,
    const temporal_generation::CausalTemporalPredictor& predictor,
    const FrameEmbeddingDecoder& decoder)
    : encoder_(&encoder), predictor_(&predictor), decoder_(&decoder) {
    if (encoder.config().embedding_dim != predictor.config().input_dim ||
        predictor.config().input_dim != predictor.config().output_dim ||
        decoder.config().embedding_dim != predictor.config().output_dim) {
        throw std::invalid_argument("video media embedding dimensions mismatch");
    }
}

std::vector<vision::RgbImage> VideoMediaGenerator::continue_frames(
    const temporal::VideoFrameSequence& seed,
    std::size_t steps) const {
    if (seed.size() == 0) throw std::invalid_argument("video generation requires seed frames");
    Tensor embeddings({seed.size(), encoder_->config().embedding_dim});
    for (std::size_t row = 0; row < seed.size(); ++row) {
        const Tensor embedding = encoder_->encode_pooled(seed.at(row));
        std::copy(embedding.data().begin(), embedding.data().end(),
                  embeddings.data().begin() + static_cast<std::ptrdiff_t>(row * embeddings.shape()[1]));
    }
    const Tensor future = predictor_->generate(embeddings, steps);
    std::vector<vision::RgbImage> frames;
    frames.reserve(steps);
    for (std::size_t row = 0; row < steps; ++row) {
        Tensor embedding({future.shape()[1]});
        std::copy_n(future.data().begin() + static_cast<std::ptrdiff_t>(row * future.shape()[1]), future.shape()[1], embedding.data().begin());
        frames.push_back(decoder_->decode(embedding));
    }
    return frames;
}

Tensor prompt_media_bias(std::string_view prompt, std::size_t feature_dim, float scale) {
    if (feature_dim == 0 || !std::isfinite(scale)) throw std::invalid_argument("invalid prompt media bias configuration");
    Tensor bias({feature_dim});
    std::uint64_t state = 1469598103934665603ULL;
    for (const unsigned char byte : prompt) {
        state ^= static_cast<std::uint64_t>(byte);
        state *= 1099511628211ULL;
    }
    for (std::size_t i = 0; i < feature_dim; ++i) {
        state ^= state >> 12; state ^= state << 25; state ^= state >> 27;
        const std::uint64_t mixed = state * 2685821657736338717ULL;
        const float unit = static_cast<float>((mixed >> 40) & 0xFFFFFFULL) / static_cast<float>(0xFFFFFFULL);
        bias.data()[i] = (unit * 2.0F - 1.0F) * scale;
    }
    return bias;
}

Tensor apply_prompt_bias(const Tensor& seed, std::string_view prompt, float scale) {
    if (seed.rank() != 2 || seed.shape()[0] == 0 || seed.shape()[1] == 0) throw std::invalid_argument("prompt bias requires sequence matrix");
    Tensor out = seed;
    const Tensor bias = prompt_media_bias(prompt, seed.shape()[1], scale);
    for (std::size_t row = 0; row < out.shape()[0]; ++row) {
        for (std::size_t col = 0; col < out.shape()[1]; ++col) out.data()[row * out.shape()[1] + col] += bias.data()[col];
    }
    return out;
}

} // namespace spiral::media_generation
