#pragma once

#include "spiral/nn.hpp"
#include "spiral/random.hpp"
#include "spiral/tensor.hpp"
#include "spiral/train.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace spiral::audio {

class AudioBuffer {
public:
    AudioBuffer() = default;
    AudioBuffer(std::uint32_t sample_rate, std::uint16_t channels, std::vector<float> interleaved_samples);

    [[nodiscard]] std::uint32_t sample_rate() const noexcept { return sample_rate_; }
    [[nodiscard]] std::uint16_t channels() const noexcept { return channels_; }
    [[nodiscard]] std::size_t frame_count() const noexcept;
    [[nodiscard]] const std::vector<float>& samples() const noexcept { return samples_; }
    [[nodiscard]] std::vector<float>& samples() noexcept { return samples_; }
    [[nodiscard]] float sample(std::size_t frame, std::size_t channel) const;

    [[nodiscard]] AudioBuffer mono() const;
    void save_wav_pcm16(const std::string& path) const;
    [[nodiscard]] static AudioBuffer load_wav(const std::string& path);

private:
    std::uint32_t sample_rate_ = 0;
    std::uint16_t channels_ = 0;
    std::vector<float> samples_;
};

struct StftConfig {
    std::size_t frame_size = 256;
    std::size_t hop_size = 128;
};

[[nodiscard]] Tensor stft_magnitude(const AudioBuffer& audio, StftConfig config = {});
[[nodiscard]] Tensor spectral_patches(const Tensor& spectrum, std::size_t frames_per_patch);

struct AudioCodecConfig {
    std::size_t patch_dim = 0;
    std::size_t latent_dim = 16;
};

class AudioLatentCodec final {
public:
    AudioLatentCodec(AudioCodecConfig config, Random& rng);

    [[nodiscard]] Tensor encode(const Tensor& patches) const;
    [[nodiscard]] Tensor decode(const Tensor& latents) const;
    [[nodiscard]] Tensor reconstruct(const Tensor& patches) const;
    [[nodiscard]] std::vector<nn::Parameter*> parameters();
    [[nodiscard]] std::vector<const nn::Parameter*> parameters() const;
    [[nodiscard]] const AudioCodecConfig& config() const noexcept { return config_; }
    [[nodiscard]] nn::Linear& encoder() noexcept { return encoder_; }
    [[nodiscard]] nn::Linear& decoder() noexcept { return decoder_; }

private:
    AudioCodecConfig config_;
    nn::Linear encoder_;
    nn::Linear decoder_;
};

[[nodiscard]] float reconstruction_loss(const Tensor& reconstructed, const Tensor& target);

struct AudioCodecTrainerConfig {
    train::AdamWConfig optimizer{0.01F, 0.9F, 0.999F, 1.0e-8F, 0.0F};
    float max_grad_norm = 5.0F;
};

class AudioCodecTrainer final {
public:
    AudioCodecTrainer(AudioLatentCodec& codec, AudioCodecTrainerConfig config = {});

    [[nodiscard]] float evaluate(const Tensor& patches) const;
    float train_step(const Tensor& patches);

private:
    AudioLatentCodec& codec_;
    AudioCodecTrainerConfig config_;
    train::AdamW optimizer_;
};

} // namespace spiral::audio
