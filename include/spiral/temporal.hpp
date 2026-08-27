#pragma once

#include "spiral/audio.hpp"
#include "spiral/dsp.hpp"
#include "spiral/latent_transformer.hpp"
#include "spiral/nn.hpp"
#include "spiral/random.hpp"
#include "spiral/tensor.hpp"
#include "spiral/vision.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace spiral::temporal {

void add_temporal_sincos_position(Tensor& tokens, float position_scale = 1.0F);

class AudioWindowCursor final {
public:
    AudioWindowCursor(const audio::AudioBuffer& audio, std::size_t window_frames, std::size_t hop_frames);

    [[nodiscard]] bool has_next() const noexcept;
    [[nodiscard]] audio::AudioBuffer next();
    void reset() noexcept { cursor_ = 0; }

private:
    const audio::AudioBuffer* audio_ = nullptr;
    std::size_t window_frames_ = 0;
    std::size_t hop_frames_ = 0;
    std::size_t cursor_ = 0;
};

struct TemporalTransformerConfig {
    std::size_t input_dim = 16;
    std::size_t model_dim = 32;
    std::size_t num_heads = 4;
    std::size_t num_layers = 2;
    std::size_t ffn_dim = 64;
    std::size_t output_dim = 32;
    float norm_epsilon = 1.0e-5F;
};

class TemporalBlock final {
public:
    TemporalBlock(
        std::size_t model_dim,
        std::size_t num_heads,
        std::size_t ffn_dim,
        float norm_epsilon,
        float residual_scale,
        Random& rng);

    [[nodiscard]] Tensor forward(const Tensor& tokens) const;
    [[nodiscard]] std::vector<nn::Parameter*> parameters();
    [[nodiscard]] std::vector<const nn::Parameter*> parameters() const;

private:
    nn::RMSNorm attention_norm_;
    latent::MultiHeadAttention attention_;
    nn::RMSNorm ffn_norm_;
    nn::Linear ffn_in_;
    nn::Linear ffn_out_;
    float residual_scale_ = 1.0F;
};

class TemporalTransformerEncoder final {
public:
    TemporalTransformerEncoder(TemporalTransformerConfig config, Random& rng);

    [[nodiscard]] Tensor encode_tokens(const Tensor& ordered_features) const;
    [[nodiscard]] Tensor encode_pooled(const Tensor& ordered_features) const;
    [[nodiscard]] std::vector<nn::Parameter*> parameters();
    [[nodiscard]] std::vector<const nn::Parameter*> parameters() const;
    [[nodiscard]] const TemporalTransformerConfig& config() const noexcept { return config_; }

private:
    TemporalTransformerConfig config_;
    nn::Linear input_projection_;
    std::vector<TemporalBlock> blocks_;
    nn::RMSNorm final_norm_;
    nn::Linear output_projection_;
};

struct AudioTemporalConfig {
    audio::StftConfig stft{256, 128};
    std::size_t frames_per_patch = 2;
    TemporalTransformerConfig temporal;
};

class AudioTemporalEncoder final {
public:
    AudioTemporalEncoder(AudioTemporalConfig config, const audio::AudioLatentCodec& codec, Random& rng);

    [[nodiscard]] Tensor spectral_latents(const audio::AudioBuffer& audio) const;
    [[nodiscard]] Tensor encode_tokens(const audio::AudioBuffer& audio) const;
    [[nodiscard]] Tensor encode_pooled(const audio::AudioBuffer& audio) const;
    [[nodiscard]] std::vector<nn::Parameter*> parameters();
    [[nodiscard]] std::vector<const nn::Parameter*> parameters() const;

private:
    AudioTemporalConfig config_;
    const audio::AudioLatentCodec* codec_ = nullptr;
    TemporalTransformerEncoder temporal_;
};

class VideoFrameSequence final {
public:
    explicit VideoFrameSequence(float frames_per_second = 30.0F);
    void add_frame(vision::RgbImage frame);

    [[nodiscard]] float frames_per_second() const noexcept { return frames_per_second_; }
    [[nodiscard]] std::size_t size() const noexcept { return frames_.size(); }
    [[nodiscard]] const vision::RgbImage& at(std::size_t index) const;

private:
    float frames_per_second_ = 30.0F;
    std::vector<vision::RgbImage> frames_;
};

struct VideoTemporalConfig {
    TemporalTransformerConfig temporal;
};

class VideoTemporalEncoder final {
public:
    VideoTemporalEncoder(VideoTemporalConfig config, const vision::VisionEncoder& vision_encoder, Random& rng);

    [[nodiscard]] Tensor frame_features(const VideoFrameSequence& sequence) const;
    [[nodiscard]] Tensor encode_tokens(const VideoFrameSequence& sequence) const;
    [[nodiscard]] Tensor encode_pooled(const VideoFrameSequence& sequence) const;
    [[nodiscard]] std::vector<nn::Parameter*> parameters();
    [[nodiscard]] std::vector<const nn::Parameter*> parameters() const;

private:
    VideoTemporalConfig config_;
    const vision::VisionEncoder* vision_encoder_ = nullptr;
    TemporalTransformerEncoder temporal_;
};

} // namespace spiral::temporal
