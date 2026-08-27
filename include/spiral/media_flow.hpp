#pragma once

#include "spiral/flow.hpp"
#include "spiral/media_generation.hpp"
#include "spiral/multimodal.hpp"
#include "spiral/nn.hpp"
#include "spiral/random.hpp"
#include "spiral/tensor.hpp"
#include "spiral/train.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace spiral::media_flow {

struct PromptTemporalFlowConfig {
    std::size_t latent_dim = 8;
    std::size_t text_feature_dim = 24;
    std::size_t time_feature_dim = 8;
    std::size_t position_feature_dim = 8;
    std::size_t hidden_dim = 64;
};

class PromptTemporalFlowModel final {
public:
    PromptTemporalFlowModel(PromptTemporalFlowConfig config, Random& rng);

    [[nodiscard]] Tensor predict_clean(
        const Tensor& noisy_sequence,
        std::string_view prompt,
        float diffusion_time) const;

    [[nodiscard]] Tensor generate(
        std::string_view prompt,
        std::size_t sequence_length,
        std::size_t steps,
        std::uint64_t seed,
        float guidance_scale = 1.0F) const;

    [[nodiscard]] std::vector<nn::Parameter*> parameters();
    [[nodiscard]] std::vector<const nn::Parameter*> parameters() const;
    [[nodiscard]] const PromptTemporalFlowConfig& config() const noexcept { return config_; }
    [[nodiscard]] nn::Linear& input_projection() noexcept { return input_projection_; }
    [[nodiscard]] nn::Linear& output_projection() noexcept { return output_projection_; }

private:
    [[nodiscard]] Tensor conditioning_matrix(
        const Tensor& noisy_sequence,
        std::string_view prompt,
        float diffusion_time) const;

    PromptTemporalFlowConfig config_;
    nn::Linear input_projection_;
    nn::Linear output_projection_;
};

struct PromptMediaExample {
    std::string prompt;
    Tensor latent_sequence;
};

struct PromptTemporalTrainerConfig {
    train::AdamWConfig optimizer{0.01F, 0.9F, 0.999F, 1.0e-8F, 0.0F};
    float max_grad_norm = 5.0F;
    float temporal_consistency_weight = 0.1F;
};

class PromptTemporalFlowTrainer final {
public:
    PromptTemporalFlowTrainer(
        PromptTemporalFlowModel& model,
        flow::NoiseScheduler scheduler = flow::NoiseScheduler{},
        PromptTemporalTrainerConfig config = {});

    [[nodiscard]] float evaluate(
        const PromptMediaExample& example,
        float diffusion_time,
        const Tensor& noise) const;

    float train_step(
        const PromptMediaExample& example,
        float diffusion_time,
        const Tensor& noise);

private:
    PromptTemporalFlowModel& model_;
    flow::NoiseScheduler scheduler_;
    PromptTemporalTrainerConfig config_;
    train::AdamW optimizer_;
};

class PromptAudioGenerator final {
public:
    PromptAudioGenerator(
        media_generation::ComplexStftConfig stft,
        std::size_t frames_per_patch,
        const media_generation::ComplexAudioCodec& codec,
        const PromptTemporalFlowModel& flow_model);

    [[nodiscard]] Tensor generate_latents(
        std::string_view prompt,
        std::size_t patch_count,
        std::size_t steps,
        std::uint64_t seed) const;

    [[nodiscard]] audio::AudioBuffer generate(
        std::string_view prompt,
        std::size_t patch_count,
        std::uint32_t sample_rate,
        std::size_t steps,
        std::uint64_t seed) const;

private:
    media_generation::ComplexStftConfig stft_;
    std::size_t frames_per_patch_ = 1;
    const media_generation::ComplexAudioCodec* codec_ = nullptr;
    const PromptTemporalFlowModel* flow_model_ = nullptr;
};

class PromptVideoGenerator final {
public:
    PromptVideoGenerator(
        const media_generation::FrameEmbeddingDecoder& decoder,
        const PromptTemporalFlowModel& flow_model);

    [[nodiscard]] std::vector<vision::RgbImage> generate(
        std::string_view prompt,
        std::size_t frame_count,
        std::size_t steps,
        std::uint64_t seed) const;

private:
    const media_generation::FrameEmbeddingDecoder* decoder_ = nullptr;
    const PromptTemporalFlowModel* flow_model_ = nullptr;
};

void save_prompt_media_flow(const PromptTemporalFlowModel& model, const std::string& path);
void load_prompt_media_flow(PromptTemporalFlowModel& model, const std::string& path);

} // namespace spiral::media_flow
