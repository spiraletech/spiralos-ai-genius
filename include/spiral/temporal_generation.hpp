#pragma once

#include "spiral/audio.hpp"
#include "spiral/dsp.hpp"
#include "spiral/latent_transformer.hpp"
#include "spiral/nn.hpp"
#include "spiral/random.hpp"
#include "spiral/temporal.hpp"
#include "spiral/tensor.hpp"
#include "spiral/train.hpp"
#include "spiral/vision.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace spiral::temporal_generation {

struct CausalTemporalConfig {
    std::size_t input_dim = 8;
    std::size_t model_dim = 32;
    std::size_t num_heads = 4;
    std::size_t num_layers = 2;
    std::size_t ffn_dim = 64;
    std::size_t output_dim = 8;
    std::size_t max_context = 64;
};

class CausalTemporalAttention final {
public:
    CausalTemporalAttention(std::size_t model_dim, std::size_t num_heads, Random& rng);

    [[nodiscard]] Tensor forward(const Tensor& tokens) const;
    [[nodiscard]] std::vector<nn::Parameter*> parameters();
    [[nodiscard]] std::vector<const nn::Parameter*> parameters() const;

    [[nodiscard]] std::size_t model_dim() const noexcept { return model_dim_; }
    [[nodiscard]] std::size_t num_heads() const noexcept { return num_heads_; }
    [[nodiscard]] std::size_t head_dim() const noexcept { return head_dim_; }
    [[nodiscard]] nn::Linear& q_proj() noexcept { return q_proj_; }
    [[nodiscard]] nn::Linear& k_proj() noexcept { return k_proj_; }
    [[nodiscard]] nn::Linear& v_proj() noexcept { return v_proj_; }
    [[nodiscard]] nn::Linear& out_proj() noexcept { return out_proj_; }

private:
    std::size_t model_dim_ = 0;
    std::size_t num_heads_ = 0;
    std::size_t head_dim_ = 0;
    nn::Linear q_proj_;
    nn::Linear k_proj_;
    nn::Linear v_proj_;
    nn::Linear out_proj_;
};

class CausalTemporalBlock final {
public:
    CausalTemporalBlock(std::size_t model_dim, std::size_t num_heads, std::size_t ffn_dim, Random& rng);

    [[nodiscard]] Tensor forward(const Tensor& tokens) const;
    [[nodiscard]] std::vector<nn::Parameter*> parameters();
    [[nodiscard]] std::vector<const nn::Parameter*> parameters() const;

    [[nodiscard]] CausalTemporalAttention& attention() noexcept { return attention_; }
    [[nodiscard]] nn::Linear& ffn_in() noexcept { return ffn_in_; }
    [[nodiscard]] nn::Linear& ffn_out() noexcept { return ffn_out_; }

private:
    CausalTemporalAttention attention_;
    nn::Linear ffn_in_;
    nn::Linear ffn_out_;
};

class CausalTemporalPredictor final {
public:
    CausalTemporalPredictor(CausalTemporalConfig config, Random& rng);

    [[nodiscard]] Tensor predict_all(const Tensor& ordered_features) const;
    [[nodiscard]] Tensor predict_next(const Tensor& context) const;
    [[nodiscard]] Tensor generate(const Tensor& seed_context, std::size_t steps) const;

    [[nodiscard]] std::vector<nn::Parameter*> parameters();
    [[nodiscard]] std::vector<const nn::Parameter*> parameters() const;
    [[nodiscard]] const CausalTemporalConfig& config() const noexcept { return config_; }

    [[nodiscard]] nn::Linear& input_projection() noexcept { return input_projection_; }
    [[nodiscard]] CausalTemporalBlock& block(std::size_t index);
    [[nodiscard]] nn::Linear& output_projection() noexcept { return output_projection_; }

private:
    CausalTemporalConfig config_;
    nn::Linear input_projection_;
    std::vector<CausalTemporalBlock> blocks_;
    nn::Linear output_projection_;
};

struct TemporalTrainerConfig {
    train::AdamWConfig optimizer{0.01F, 0.9F, 0.999F, 1.0e-8F, 0.0F};
    float max_grad_norm = 5.0F;
};

class TemporalNextLatentTrainer final {
public:
    TemporalNextLatentTrainer(CausalTemporalPredictor& model, TemporalTrainerConfig config = {});

    [[nodiscard]] float evaluate(const Tensor& sequence) const;
    float train_step(const Tensor& sequence);

private:
    CausalTemporalPredictor& model_;
    TemporalTrainerConfig config_;
    train::AdamW optimizer_;
};

[[nodiscard]] audio::AudioBuffer magnitude_to_audio_zero_phase(
    const Tensor& magnitude,
    std::uint32_t sample_rate,
    audio::StftConfig config = {});

struct AudioGenerationConfig {
    audio::StftConfig stft{256, 128};
    std::size_t frames_per_patch = 2;
};

class AudioLatentGenerator final {
public:
    AudioLatentGenerator(
        AudioGenerationConfig config,
        const audio::AudioLatentCodec& codec,
        const CausalTemporalPredictor& predictor);

    [[nodiscard]] Tensor encode_latents(const audio::AudioBuffer& audio) const;
    [[nodiscard]] Tensor generate_latents(const audio::AudioBuffer& seed_audio, std::size_t steps) const;
    [[nodiscard]] audio::AudioBuffer synthesize(const Tensor& latents, std::uint32_t sample_rate) const;

private:
    AudioGenerationConfig config_;
    const audio::AudioLatentCodec* codec_ = nullptr;
    const CausalTemporalPredictor* predictor_ = nullptr;
};

class VideoEmbeddingGenerator final {
public:
    VideoEmbeddingGenerator(const vision::VisionEncoder& vision_encoder, const CausalTemporalPredictor& predictor);

    [[nodiscard]] Tensor frame_embeddings(const temporal::VideoFrameSequence& sequence) const;
    [[nodiscard]] Tensor predict_next(const temporal::VideoFrameSequence& sequence) const;
    [[nodiscard]] Tensor generate(const temporal::VideoFrameSequence& sequence, std::size_t steps) const;

private:
    const vision::VisionEncoder* vision_encoder_ = nullptr;
    const CausalTemporalPredictor* predictor_ = nullptr;
};

void save_temporal_predictor(const CausalTemporalPredictor& model, const std::string& path);
void load_temporal_predictor(CausalTemporalPredictor& model, const std::string& path);

} // namespace spiral::temporal_generation
