#include "spiral/temporal.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace spiral::temporal {
namespace {

void append_parameters(std::vector<nn::Parameter*>& out, std::vector<nn::Parameter*> values) {
    out.insert(out.end(), values.begin(), values.end());
}

void append_parameters(std::vector<const nn::Parameter*>& out, std::vector<const nn::Parameter*> values) {
    out.insert(out.end(), values.begin(), values.end());
}

Tensor add_scaled(const Tensor& base, const Tensor& delta, float scale) {
    if (base.shape() != delta.shape()) throw std::invalid_argument("temporal residual shape mismatch");
    Tensor out = base;
    for (std::size_t i = 0; i < out.numel(); ++i) out.data()[i] += delta.data()[i] * scale;
    return out;
}

void silu_inplace(Tensor& tensor) {
    for (auto& value : tensor.data()) value = value / (1.0F + std::exp(-value));
}

} // namespace

void add_temporal_sincos_position(Tensor& tokens, float position_scale) {
    if (tokens.rank() != 2 || tokens.shape()[1] == 0) throw std::invalid_argument("temporal position expects rank-2 tokens");
    if (!std::isfinite(position_scale) || position_scale <= 0.0F) throw std::invalid_argument("temporal position scale must be positive");
    const std::size_t sequence = tokens.shape()[0];
    const std::size_t dim = tokens.shape()[1];
    for (std::size_t position = 0; position < sequence; ++position) {
        for (std::size_t channel = 0; channel < dim; ++channel) {
            const std::size_t pair = channel / 2;
            const float exponent = 2.0F * static_cast<float>(pair) / static_cast<float>(dim);
            const float angle = position_scale * static_cast<float>(position) / std::pow(10000.0F, exponent);
            tokens.data()[position * dim + channel] += (channel % 2 == 0) ? std::sin(angle) : std::cos(angle);
        }
    }
}

AudioWindowCursor::AudioWindowCursor(const audio::AudioBuffer& audio, std::size_t window_frames, std::size_t hop_frames)
    : audio_(&audio), window_frames_(window_frames), hop_frames_(hop_frames) {
    if (window_frames_ == 0 || hop_frames_ == 0) throw std::invalid_argument("audio window and hop must be non-zero");
}

bool AudioWindowCursor::has_next() const noexcept {
    return audio_ != nullptr && cursor_ + window_frames_ <= audio_->frame_count();
}

audio::AudioBuffer AudioWindowCursor::next() {
    if (!has_next()) throw std::out_of_range("audio window cursor exhausted");
    std::vector<float> samples(window_frames_ * audio_->channels());
    for (std::size_t frame = 0; frame < window_frames_; ++frame) {
        for (std::size_t channel = 0; channel < audio_->channels(); ++channel) {
            samples[frame * audio_->channels() + channel] = audio_->sample(cursor_ + frame, channel);
        }
    }
    cursor_ += hop_frames_;
    return audio::AudioBuffer(audio_->sample_rate(), audio_->channels(), std::move(samples));
}

TemporalBlock::TemporalBlock(
    std::size_t model_dim,
    std::size_t num_heads,
    std::size_t ffn_dim,
    float norm_epsilon,
    float residual_scale,
    Random& rng)
    : attention_norm_(model_dim, norm_epsilon),
      attention_(model_dim, num_heads, rng),
      ffn_norm_(model_dim, norm_epsilon),
      ffn_in_(model_dim, ffn_dim, rng, true),
      ffn_out_(ffn_dim, model_dim, rng, true),
      residual_scale_(residual_scale) {
    if (!std::isfinite(residual_scale_) || residual_scale_ <= 0.0F) throw std::invalid_argument("temporal residual scale must be positive");
}

Tensor TemporalBlock::forward(const Tensor& tokens) const {
    const Tensor normalized = attention_norm_.forward(tokens);
    const Tensor attended = attention_.forward(normalized, normalized);
    Tensor residual = add_scaled(tokens, attended, residual_scale_);
    Tensor hidden = ffn_in_.forward(ffn_norm_.forward(residual));
    silu_inplace(hidden);
    return add_scaled(residual, ffn_out_.forward(hidden), residual_scale_);
}

std::vector<nn::Parameter*> TemporalBlock::parameters() {
    std::vector<nn::Parameter*> out;
    append_parameters(out, attention_norm_.parameters());
    append_parameters(out, attention_.parameters());
    append_parameters(out, ffn_norm_.parameters());
    append_parameters(out, ffn_in_.parameters());
    append_parameters(out, ffn_out_.parameters());
    return out;
}

std::vector<const nn::Parameter*> TemporalBlock::parameters() const {
    std::vector<const nn::Parameter*> out;
    append_parameters(out, attention_norm_.parameters());
    append_parameters(out, attention_.parameters());
    append_parameters(out, ffn_norm_.parameters());
    append_parameters(out, ffn_in_.parameters());
    append_parameters(out, ffn_out_.parameters());
    return out;
}

TemporalTransformerEncoder::TemporalTransformerEncoder(TemporalTransformerConfig config, Random& rng)
    : config_(config),
      input_projection_(config.input_dim, config.model_dim, rng, true),
      final_norm_(config.model_dim, config.norm_epsilon),
      output_projection_(config.model_dim, config.output_dim, rng, true) {
    if (config_.input_dim == 0 || config_.model_dim == 0 || config_.num_heads == 0 || config_.num_layers == 0 ||
        config_.ffn_dim == 0 || config_.output_dim == 0 || config_.norm_epsilon <= 0.0F ||
        config_.model_dim % config_.num_heads != 0) {
        throw std::invalid_argument("invalid temporal transformer configuration");
    }
    const float residual_scale = 1.0F / std::sqrt(static_cast<float>(config_.num_layers));
    blocks_.reserve(config_.num_layers);
    for (std::size_t layer = 0; layer < config_.num_layers; ++layer) {
        blocks_.emplace_back(config_.model_dim, config_.num_heads, config_.ffn_dim, config_.norm_epsilon, residual_scale, rng);
    }
}

Tensor TemporalTransformerEncoder::encode_tokens(const Tensor& ordered_features) const {
    if (ordered_features.rank() != 2 || ordered_features.shape()[1] != config_.input_dim || ordered_features.shape()[0] == 0) {
        throw std::invalid_argument("temporal encoder input shape mismatch");
    }
    Tensor tokens = input_projection_.forward(ordered_features);
    add_temporal_sincos_position(tokens);
    for (const auto& block : blocks_) tokens = block.forward(tokens);
    return output_projection_.forward(final_norm_.forward(tokens));
}

Tensor TemporalTransformerEncoder::encode_pooled(const Tensor& ordered_features) const {
    const Tensor tokens = encode_tokens(ordered_features);
    Tensor pooled({tokens.shape()[1]});
    for (std::size_t row = 0; row < tokens.shape()[0]; ++row) {
        for (std::size_t col = 0; col < tokens.shape()[1]; ++col) pooled.data()[col] += tokens.data()[row * tokens.shape()[1] + col];
    }
    const float inv = 1.0F / static_cast<float>(tokens.shape()[0]);
    for (auto& value : pooled.data()) value *= inv;
    return pooled;
}

std::vector<nn::Parameter*> TemporalTransformerEncoder::parameters() {
    std::vector<nn::Parameter*> out;
    append_parameters(out, input_projection_.parameters());
    for (auto& block : blocks_) append_parameters(out, block.parameters());
    append_parameters(out, final_norm_.parameters());
    append_parameters(out, output_projection_.parameters());
    return out;
}

std::vector<const nn::Parameter*> TemporalTransformerEncoder::parameters() const {
    std::vector<const nn::Parameter*> out;
    append_parameters(out, input_projection_.parameters());
    for (const auto& block : blocks_) append_parameters(out, block.parameters());
    append_parameters(out, final_norm_.parameters());
    append_parameters(out, output_projection_.parameters());
    return out;
}

AudioTemporalEncoder::AudioTemporalEncoder(AudioTemporalConfig config, const audio::AudioLatentCodec& codec, Random& rng)
    : config_(config), codec_(&codec), temporal_(config.temporal, rng) {
    if (config_.frames_per_patch == 0) throw std::invalid_argument("audio temporal frames_per_patch must be non-zero");
    if (config_.temporal.input_dim != codec.config().latent_dim) throw std::invalid_argument("audio temporal input_dim must match codec latent_dim");
}

Tensor AudioTemporalEncoder::spectral_latents(const audio::AudioBuffer& input) const {
    const Tensor spectrum = dsp::stft_magnitude_fft(input, config_.stft);
    const Tensor patches = audio::spectral_patches(spectrum, config_.frames_per_patch);
    if (patches.shape()[1] != codec_->config().patch_dim) throw std::invalid_argument("audio codec patch_dim does not match temporal spectral patches");
    return codec_->encode(patches);
}

Tensor AudioTemporalEncoder::encode_tokens(const audio::AudioBuffer& input) const {
    return temporal_.encode_tokens(spectral_latents(input));
}

Tensor AudioTemporalEncoder::encode_pooled(const audio::AudioBuffer& input) const {
    return temporal_.encode_pooled(spectral_latents(input));
}

std::vector<nn::Parameter*> AudioTemporalEncoder::parameters() { return temporal_.parameters(); }
std::vector<const nn::Parameter*> AudioTemporalEncoder::parameters() const { return temporal_.parameters(); }

VideoFrameSequence::VideoFrameSequence(float frames_per_second) : frames_per_second_(frames_per_second) {
    if (!std::isfinite(frames_per_second_) || frames_per_second_ <= 0.0F) throw std::invalid_argument("video fps must be positive");
}

void VideoFrameSequence::add_frame(vision::RgbImage frame) {
    if (frame.width() == 0 || frame.height() == 0) throw std::invalid_argument("video frame must be non-empty");
    frames_.push_back(std::move(frame));
}

const vision::RgbImage& VideoFrameSequence::at(std::size_t index) const { return frames_.at(index); }

VideoTemporalEncoder::VideoTemporalEncoder(VideoTemporalConfig config, const vision::VisionEncoder& vision_encoder, Random& rng)
    : config_(config), vision_encoder_(&vision_encoder), temporal_(config.temporal, rng) {
    if (config_.temporal.input_dim != vision_encoder.config().embedding_dim) {
        throw std::invalid_argument("video temporal input_dim must match vision embedding_dim");
    }
}

Tensor VideoTemporalEncoder::frame_features(const VideoFrameSequence& sequence) const {
    if (sequence.size() == 0) throw std::invalid_argument("video sequence must contain frames");
    const std::size_t dim = vision_encoder_->config().embedding_dim;
    Tensor features({sequence.size(), dim});
    for (std::size_t frame = 0; frame < sequence.size(); ++frame) {
        const Tensor pooled = vision_encoder_->encode_pooled(sequence.at(frame));
        for (std::size_t col = 0; col < dim; ++col) features.data()[frame * dim + col] = pooled.data()[col];
    }
    return features;
}

Tensor VideoTemporalEncoder::encode_tokens(const VideoFrameSequence& sequence) const {
    return temporal_.encode_tokens(frame_features(sequence));
}

Tensor VideoTemporalEncoder::encode_pooled(const VideoFrameSequence& sequence) const {
    return temporal_.encode_pooled(frame_features(sequence));
}

std::vector<nn::Parameter*> VideoTemporalEncoder::parameters() { return temporal_.parameters(); }
std::vector<const nn::Parameter*> VideoTemporalEncoder::parameters() const { return temporal_.parameters(); }

} // namespace spiral::temporal
