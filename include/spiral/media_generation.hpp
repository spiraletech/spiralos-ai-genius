#pragma once

#include "spiral/audio.hpp"
#include "spiral/nn.hpp"
#include "spiral/random.hpp"
#include "spiral/temporal_generation.hpp"
#include "spiral/tensor.hpp"
#include "spiral/train.hpp"
#include "spiral/vision.hpp"

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

namespace spiral::media_generation {

struct ComplexStftConfig {
    std::size_t frame_size = 256;
    std::size_t hop_size = 128;
};

[[nodiscard]] Tensor complex_stft(
    const audio::AudioBuffer& audio,
    ComplexStftConfig config = {});

[[nodiscard]] audio::AudioBuffer inverse_complex_stft(
    const Tensor& spectrum,
    std::uint32_t sample_rate,
    ComplexStftConfig config = {});

[[nodiscard]] Tensor complex_spectral_patches(
    const Tensor& spectrum,
    std::size_t frames_per_patch);

[[nodiscard]] Tensor complex_patches_to_spectrum(
    const Tensor& patches,
    std::size_t frame_count,
    std::size_t bin_count,
    std::size_t frames_per_patch);

struct ComplexAudioCodecConfig {
    std::size_t patch_dim = 0;
    std::size_t latent_dim = 16;
};

class ComplexAudioCodec final {
public:
    ComplexAudioCodec(ComplexAudioCodecConfig config, Random& rng);

    [[nodiscard]] Tensor encode(const Tensor& complex_patches) const;
    [[nodiscard]] Tensor decode(const Tensor& latents) const;
    [[nodiscard]] Tensor reconstruct(const Tensor& complex_patches) const;
    [[nodiscard]] std::vector<nn::Parameter*> parameters();
    [[nodiscard]] std::vector<const nn::Parameter*> parameters() const;
    [[nodiscard]] const ComplexAudioCodecConfig& config() const noexcept { return config_; }
    [[nodiscard]] nn::Linear& encoder() noexcept { return encoder_; }
    [[nodiscard]] nn::Linear& decoder() noexcept { return decoder_; }

private:
    ComplexAudioCodecConfig config_;
    nn::Linear encoder_;
    nn::Linear decoder_;
};

struct MediaTrainerConfig {
    train::AdamWConfig optimizer{0.01F, 0.9F, 0.999F, 1.0e-8F, 0.0F};
    float max_grad_norm = 5.0F;
};

class ComplexAudioCodecTrainer final {
public:
    ComplexAudioCodecTrainer(ComplexAudioCodec& codec, MediaTrainerConfig config = {});
    [[nodiscard]] float evaluate(const Tensor& patches) const;
    float train_step(const Tensor& patches);

private:
    ComplexAudioCodec& codec_;
    MediaTrainerConfig config_;
    train::AdamW optimizer_;
};

struct FrameDecoderConfig {
    std::size_t embedding_dim = 32;
    std::size_t width = 4;
    std::size_t height = 4;
    std::size_t hidden_dim = 64;
};

class FrameEmbeddingDecoder final {
public:
    FrameEmbeddingDecoder(FrameDecoderConfig config, Random& rng);

    [[nodiscard]] vision::RgbImage decode(const Tensor& embedding) const;
    [[nodiscard]] Tensor decode_tensor(const Tensor& embedding) const;
    [[nodiscard]] std::vector<nn::Parameter*> parameters();
    [[nodiscard]] std::vector<const nn::Parameter*> parameters() const;
    [[nodiscard]] const FrameDecoderConfig& config() const noexcept { return config_; }
    [[nodiscard]] nn::Linear& hidden() noexcept { return hidden_; }
    [[nodiscard]] nn::Linear& output() noexcept { return output_; }

private:
    FrameDecoderConfig config_;
    nn::Linear hidden_;
    nn::Linear output_;
};

class FrameDecoderTrainer final {
public:
    FrameDecoderTrainer(
        FrameEmbeddingDecoder& decoder,
        const vision::VisionEncoder& encoder,
        MediaTrainerConfig config = {});

    [[nodiscard]] float evaluate(const vision::RgbImage& frame) const;
    float train_step(const vision::RgbImage& frame);

private:
    FrameEmbeddingDecoder& decoder_;
    const vision::VisionEncoder& encoder_;
    MediaTrainerConfig config_;
    train::AdamW optimizer_;
};

class AudioMediaGenerator final {
public:
    AudioMediaGenerator(
        ComplexStftConfig stft,
        std::size_t frames_per_patch,
        const ComplexAudioCodec& codec,
        const temporal_generation::CausalTemporalPredictor& predictor);

    [[nodiscard]] Tensor encode_latents(const audio::AudioBuffer& audio) const;
    [[nodiscard]] Tensor continue_latents(const audio::AudioBuffer& seed, std::size_t steps) const;
    [[nodiscard]] audio::AudioBuffer decode_latents(
        const Tensor& latents,
        std::uint32_t sample_rate,
        std::size_t frame_count) const;

private:
    ComplexStftConfig stft_;
    std::size_t frames_per_patch_ = 1;
    const ComplexAudioCodec* codec_ = nullptr;
    const temporal_generation::CausalTemporalPredictor* predictor_ = nullptr;
};

class VideoMediaGenerator final {
public:
    VideoMediaGenerator(
        const vision::VisionEncoder& encoder,
        const temporal_generation::CausalTemporalPredictor& predictor,
        const FrameEmbeddingDecoder& decoder);

    [[nodiscard]] std::vector<vision::RgbImage> continue_frames(
        const temporal::VideoFrameSequence& seed,
        std::size_t steps) const;

private:
    const vision::VisionEncoder* encoder_ = nullptr;
    const temporal_generation::CausalTemporalPredictor* predictor_ = nullptr;
    const FrameEmbeddingDecoder* decoder_ = nullptr;
};

[[nodiscard]] Tensor prompt_media_bias(
    std::string_view prompt,
    std::size_t feature_dim,
    float scale = 0.05F);

[[nodiscard]] Tensor apply_prompt_bias(const Tensor& seed, std::string_view prompt, float scale = 0.05F);

} // namespace spiral::media_generation
