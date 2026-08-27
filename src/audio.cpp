#include "spiral/audio.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>

namespace spiral::audio {
namespace {

constexpr double kPi = 3.14159265358979323846;

void require_matrix(const Tensor& tensor, std::size_t features, const char* operation) {
    if (tensor.rank() != 2 || tensor.shape()[1] != features || tensor.shape()[0] == 0) {
        throw std::invalid_argument(std::string(operation) + " requires non-empty [rows,features] tensor");
    }
}

std::uint16_t read_u16(std::istream& in) {
    std::array<unsigned char, 2> bytes{};
    in.read(reinterpret_cast<char*>(bytes.data()), 2);
    if (!in) throw std::runtime_error("truncated WAV file");
    return static_cast<std::uint16_t>(bytes[0]) |
           (static_cast<std::uint16_t>(bytes[1]) << 8U);
}

std::uint32_t read_u32(std::istream& in) {
    std::array<unsigned char, 4> bytes{};
    in.read(reinterpret_cast<char*>(bytes.data()), 4);
    if (!in) throw std::runtime_error("truncated WAV file");
    return static_cast<std::uint32_t>(bytes[0]) |
           (static_cast<std::uint32_t>(bytes[1]) << 8U) |
           (static_cast<std::uint32_t>(bytes[2]) << 16U) |
           (static_cast<std::uint32_t>(bytes[3]) << 24U);
}

void write_u16(std::ostream& out, std::uint16_t value) {
    const std::array<unsigned char, 2> bytes{
        static_cast<unsigned char>(value & 0xFFU),
        static_cast<unsigned char>((value >> 8U) & 0xFFU)};
    out.write(reinterpret_cast<const char*>(bytes.data()), 2);
}

void write_u32(std::ostream& out, std::uint32_t value) {
    const std::array<unsigned char, 4> bytes{
        static_cast<unsigned char>(value & 0xFFU),
        static_cast<unsigned char>((value >> 8U) & 0xFFU),
        static_cast<unsigned char>((value >> 16U) & 0xFFU),
        static_cast<unsigned char>((value >> 24U) & 0xFFU)};
    out.write(reinterpret_cast<const char*>(bytes.data()), 4);
}

Tensor mse_gradient(const Tensor& prediction, const Tensor& target) {
    if (prediction.shape() != target.shape() || prediction.numel() == 0) {
        throw std::invalid_argument("audio MSE gradient requires equal non-empty tensors");
    }
    Tensor grad(prediction.shape());
    const float scale = 2.0F / static_cast<float>(prediction.numel());
    for (std::size_t i = 0; i < prediction.numel(); ++i) {
        grad.data()[i] = scale * (prediction.data()[i] - target.data()[i]);
    }
    return grad;
}

Tensor linear_backward(nn::Linear& layer, const Tensor& input, const Tensor& grad_output) {
    require_matrix(input, layer.in_features(), "audio linear backward input");
    require_matrix(grad_output, layer.out_features(), "audio linear backward output");
    if (input.shape()[0] != grad_output.shape()[0]) throw std::invalid_argument("audio linear backward row mismatch");

    auto& weight = layer.weight();
    weight.ensure_grad();
    nn::Parameter* bias = nullptr;
    if (layer.uses_bias()) {
        bias = &layer.bias();
        bias->ensure_grad();
    }

    const std::size_t rows = input.shape()[0];
    const std::size_t in_features = layer.in_features();
    const std::size_t out_features = layer.out_features();
    Tensor grad_input({rows, in_features});

    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t out = 0; out < out_features; ++out) {
            const float g = grad_output.data()[row * out_features + out];
            if (bias != nullptr) bias->grad.data()[out] += g;
            for (std::size_t in = 0; in < in_features; ++in) {
                weight.grad.data()[in * out_features + out] += input.data()[row * in_features + in] * g;
                grad_input.data()[row * in_features + in] += weight.value.data()[in * out_features + out] * g;
            }
        }
    }
    return grad_input;
}

} // namespace

AudioBuffer::AudioBuffer(std::uint32_t sample_rate, std::uint16_t channels, std::vector<float> interleaved_samples)
    : sample_rate_(sample_rate), channels_(channels), samples_(std::move(interleaved_samples)) {
    if (sample_rate_ == 0 || channels_ == 0) throw std::invalid_argument("audio sample rate/channels must be non-zero");
    if (samples_.size() % channels_ != 0) throw std::invalid_argument("audio sample count must align to channels");
    for (float& sample_value : samples_) {
        if (!std::isfinite(sample_value)) throw std::invalid_argument("audio samples must be finite");
        sample_value = std::clamp(sample_value, -1.0F, 1.0F);
    }
}

std::size_t AudioBuffer::frame_count() const noexcept {
    return channels_ == 0 ? 0 : samples_.size() / channels_;
}

float AudioBuffer::sample(std::size_t frame, std::size_t channel) const {
    if (frame >= frame_count() || channel >= channels_) throw std::out_of_range("audio sample index out of range");
    return samples_[frame * channels_ + channel];
}

AudioBuffer AudioBuffer::mono() const {
    if (channels_ == 0) return {};
    std::vector<float> mono_samples(frame_count(), 0.0F);
    for (std::size_t frame = 0; frame < frame_count(); ++frame) {
        float sum = 0.0F;
        for (std::size_t channel = 0; channel < channels_; ++channel) sum += sample(frame, channel);
        mono_samples[frame] = sum / static_cast<float>(channels_);
    }
    return AudioBuffer(sample_rate_, 1, std::move(mono_samples));
}

void AudioBuffer::save_wav_pcm16(const std::string& path) const {
    if (sample_rate_ == 0 || channels_ == 0) throw std::runtime_error("cannot save empty audio buffer");
    if (samples_.size() > (std::numeric_limits<std::uint32_t>::max() / 2U)) throw std::overflow_error("WAV sample payload too large");
    const std::uint32_t data_bytes = static_cast<std::uint32_t>(samples_.size() * sizeof(std::int16_t));
    const std::uint32_t riff_size = 36U + data_bytes;
    const std::uint32_t byte_rate = sample_rate_ * static_cast<std::uint32_t>(channels_) * 2U;
    const std::uint16_t block_align = static_cast<std::uint16_t>(channels_ * 2U);

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("failed to open WAV for writing");
    out.write("RIFF", 4);
    write_u32(out, riff_size);
    out.write("WAVE", 4);
    out.write("fmt ", 4);
    write_u32(out, 16U);
    write_u16(out, 1U);
    write_u16(out, channels_);
    write_u32(out, sample_rate_);
    write_u32(out, byte_rate);
    write_u16(out, block_align);
    write_u16(out, 16U);
    out.write("data", 4);
    write_u32(out, data_bytes);
    for (const float sample_value : samples_) {
        const float clamped = std::clamp(sample_value, -1.0F, 1.0F);
        const int scaled = static_cast<int>(std::lrint(clamped * 32767.0F));
        const auto pcm = static_cast<std::int16_t>(std::clamp(scaled, -32768, 32767));
        write_u16(out, static_cast<std::uint16_t>(pcm));
    }
    if (!out) throw std::runtime_error("failed while writing WAV");
}

AudioBuffer AudioBuffer::load_wav(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("failed to open WAV for reading");
    std::array<char, 4> id{};
    in.read(id.data(), 4);
    if (!in || std::memcmp(id.data(), "RIFF", 4) != 0) throw std::runtime_error("invalid WAV RIFF header");
    (void)read_u32(in);
    in.read(id.data(), 4);
    if (!in || std::memcmp(id.data(), "WAVE", 4) != 0) throw std::runtime_error("invalid WAV WAVE header");

    std::uint16_t format = 0;
    std::uint16_t channels = 0;
    std::uint32_t sample_rate = 0;
    std::uint16_t bits_per_sample = 0;
    std::vector<unsigned char> data;

    while (in) {
        in.read(id.data(), 4);
        if (!in) break;
        const std::uint32_t chunk_size = read_u32(in);
        if (std::memcmp(id.data(), "fmt ", 4) == 0) {
            if (chunk_size < 16U) throw std::runtime_error("invalid WAV fmt chunk");
            format = read_u16(in);
            channels = read_u16(in);
            sample_rate = read_u32(in);
            (void)read_u32(in);
            (void)read_u16(in);
            bits_per_sample = read_u16(in);
            if (chunk_size > 16U) in.seekg(static_cast<std::streamoff>(chunk_size - 16U), std::ios::cur);
        } else if (std::memcmp(id.data(), "data", 4) == 0) {
            data.resize(chunk_size);
            in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(chunk_size));
            if (!in) throw std::runtime_error("truncated WAV data chunk");
        } else {
            in.seekg(static_cast<std::streamoff>(chunk_size), std::ios::cur);
        }
        if ((chunk_size & 1U) != 0U) in.seekg(1, std::ios::cur);
    }

    if (format != 1U || bits_per_sample != 16U || channels == 0 || sample_rate == 0 || data.empty()) {
        throw std::runtime_error("WAV loader currently requires PCM16 audio");
    }
    if (data.size() % 2U != 0U) throw std::runtime_error("invalid PCM16 byte count");
    std::vector<float> samples(data.size() / 2U);
    for (std::size_t i = 0; i < samples.size(); ++i) {
        const std::uint16_t raw = static_cast<std::uint16_t>(data[i * 2U]) |
                                  (static_cast<std::uint16_t>(data[i * 2U + 1U]) << 8U);
        const std::int16_t pcm = static_cast<std::int16_t>(raw);
        samples[i] = static_cast<float>(pcm) / 32768.0F;
    }
    return AudioBuffer(sample_rate, channels, std::move(samples));
}

Tensor stft_magnitude(const AudioBuffer& audio, StftConfig config) {
    if (audio.sample_rate() == 0 || audio.channels() == 0 || audio.frame_count() == 0) {
        throw std::invalid_argument("STFT requires non-empty audio");
    }
    if (config.frame_size < 2 || config.hop_size == 0) throw std::invalid_argument("invalid STFT configuration");
    const AudioBuffer mono_audio = audio.channels() == 1 ? audio : audio.mono();
    const std::size_t frames = mono_audio.frame_count() <= config.frame_size
        ? 1
        : 1 + (mono_audio.frame_count() - config.frame_size + config.hop_size - 1) / config.hop_size;
    const std::size_t bins = config.frame_size / 2U + 1U;
    Tensor spectrum({frames, bins});

    for (std::size_t frame = 0; frame < frames; ++frame) {
        const std::size_t start = frame * config.hop_size;
        for (std::size_t bin = 0; bin < bins; ++bin) {
            double real = 0.0;
            double imag = 0.0;
            for (std::size_t n = 0; n < config.frame_size; ++n) {
                const std::size_t index = start + n;
                const double sample_value = index < mono_audio.frame_count() ? mono_audio.samples()[index] : 0.0;
                const double window = 0.5 - 0.5 * std::cos((2.0 * kPi * static_cast<double>(n)) /
                                                           static_cast<double>(config.frame_size - 1U));
                const double angle = -2.0 * kPi * static_cast<double>(bin * n) / static_cast<double>(config.frame_size);
                real += sample_value * window * std::cos(angle);
                imag += sample_value * window * std::sin(angle);
            }
            spectrum.data()[frame * bins + bin] = static_cast<float>(std::sqrt(real * real + imag * imag));
        }
    }
    return spectrum;
}

Tensor spectral_patches(const Tensor& spectrum, std::size_t frames_per_patch) {
    if (spectrum.rank() != 2 || spectrum.shape()[0] == 0 || spectrum.shape()[1] == 0 || frames_per_patch == 0) {
        throw std::invalid_argument("spectral_patches requires non-empty rank-2 spectrum and positive patch size");
    }
    const std::size_t frames = spectrum.shape()[0];
    const std::size_t bins = spectrum.shape()[1];
    const std::size_t patch_count = (frames + frames_per_patch - 1U) / frames_per_patch;
    Tensor patches({patch_count, frames_per_patch * bins});
    for (std::size_t patch = 0; patch < patch_count; ++patch) {
        for (std::size_t local_frame = 0; local_frame < frames_per_patch; ++local_frame) {
            const std::size_t source_frame = patch * frames_per_patch + local_frame;
            if (source_frame >= frames) continue;
            for (std::size_t bin = 0; bin < bins; ++bin) {
                patches.data()[patch * frames_per_patch * bins + local_frame * bins + bin] =
                    spectrum.data()[source_frame * bins + bin];
            }
        }
    }
    return patches;
}

AudioLatentCodec::AudioLatentCodec(AudioCodecConfig config, Random& rng)
    : config_(config),
      encoder_(config.patch_dim, config.latent_dim, rng, true),
      decoder_(config.latent_dim, config.patch_dim, rng, true) {
    if (config_.patch_dim == 0 || config_.latent_dim == 0) throw std::invalid_argument("audio codec dimensions must be non-zero");
}

Tensor AudioLatentCodec::encode(const Tensor& patches) const {
    require_matrix(patches, config_.patch_dim, "AudioLatentCodec::encode");
    return encoder_.forward(patches);
}

Tensor AudioLatentCodec::decode(const Tensor& latents) const {
    require_matrix(latents, config_.latent_dim, "AudioLatentCodec::decode");
    return decoder_.forward(latents);
}

Tensor AudioLatentCodec::reconstruct(const Tensor& patches) const {
    return decode(encode(patches));
}

std::vector<nn::Parameter*> AudioLatentCodec::parameters() {
    auto out = encoder_.parameters();
    auto decoder_parameters = decoder_.parameters();
    out.insert(out.end(), decoder_parameters.begin(), decoder_parameters.end());
    return out;
}

std::vector<const nn::Parameter*> AudioLatentCodec::parameters() const {
    auto out = encoder_.parameters();
    auto decoder_parameters = decoder_.parameters();
    out.insert(out.end(), decoder_parameters.begin(), decoder_parameters.end());
    return out;
}

float reconstruction_loss(const Tensor& reconstructed, const Tensor& target) {
    if (reconstructed.shape() != target.shape() || reconstructed.numel() == 0) {
        throw std::invalid_argument("audio reconstruction loss requires equal non-empty tensors");
    }
    double sum = 0.0;
    for (std::size_t i = 0; i < reconstructed.numel(); ++i) {
        const double delta = static_cast<double>(reconstructed.data()[i]) - target.data()[i];
        sum += delta * delta;
    }
    return static_cast<float>(sum / static_cast<double>(reconstructed.numel()));
}

AudioCodecTrainer::AudioCodecTrainer(AudioLatentCodec& codec, AudioCodecTrainerConfig config)
    : codec_(codec), config_(config), optimizer_(codec.parameters(), config.optimizer) {
    if (config_.max_grad_norm <= 0.0F) throw std::invalid_argument("audio codec max_grad_norm must be positive");
}

float AudioCodecTrainer::evaluate(const Tensor& patches) const {
    return reconstruction_loss(codec_.reconstruct(patches), patches);
}

float AudioCodecTrainer::train_step(const Tensor& patches) {
    require_matrix(patches, codec_.config().patch_dim, "AudioCodecTrainer::train_step");
    optimizer_.zero_grad();
    const Tensor latents = codec_.encoder().forward(patches);
    const Tensor reconstructed = codec_.decoder().forward(latents);
    const float loss = reconstruction_loss(reconstructed, patches);
    const Tensor grad_reconstructed = mse_gradient(reconstructed, patches);
    const Tensor grad_latents = linear_backward(codec_.decoder(), latents, grad_reconstructed);
    (void)linear_backward(codec_.encoder(), patches, grad_latents);
    auto parameters = codec_.parameters();
    train::clip_grad_norm(parameters, config_.max_grad_norm);
    optimizer_.step();
    return loss;
}

} // namespace spiral::audio
