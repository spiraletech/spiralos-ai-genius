#pragma once

#include "spiral/flow.hpp"
#include "spiral/latent_transformer.hpp"
#include "spiral/multimodal.hpp"
#include "spiral/nn.hpp"
#include "spiral/random.hpp"
#include "spiral/tensor.hpp"
#include "spiral/train.hpp"
#include "spiral/vision.hpp"

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <string_view>
#include <vector>

namespace spiral::scale {

struct StableLatentTransformerConfig {
    std::size_t latent_dim = 8;
    std::size_t model_dim = 32;
    std::size_t num_heads = 4;
    std::size_t num_layers = 6;
    std::size_t ffn_dim = 96;
    std::size_t text_feature_dim = 24;
    std::size_t prompt_tokens = 6;
    std::size_t time_feature_dim = 8;
    float norm_epsilon = 1.0e-5F;
    float residual_scale = 0.0F; // 0 => 1/sqrt(num_layers)
};

class StableLatentTransformerBlock final {
public:
    StableLatentTransformerBlock(
        std::size_t model_dim,
        std::size_t num_heads,
        std::size_t ffn_dim,
        float norm_epsilon,
        float residual_scale,
        Random& rng);

    [[nodiscard]] Tensor forward(const Tensor& latent_tokens, const Tensor& prompt_tokens) const;
    [[nodiscard]] std::vector<nn::Parameter*> parameters();
    [[nodiscard]] std::vector<const nn::Parameter*> parameters() const;

    [[nodiscard]] nn::RMSNorm& self_norm() noexcept { return self_norm_; }
    [[nodiscard]] const nn::RMSNorm& self_norm() const noexcept { return self_norm_; }
    [[nodiscard]] latent::MultiHeadAttention& self_attention() noexcept { return self_attention_; }
    [[nodiscard]] const latent::MultiHeadAttention& self_attention() const noexcept { return self_attention_; }
    [[nodiscard]] nn::RMSNorm& cross_norm() noexcept { return cross_norm_; }
    [[nodiscard]] const nn::RMSNorm& cross_norm() const noexcept { return cross_norm_; }
    [[nodiscard]] latent::MultiHeadAttention& cross_attention() noexcept { return cross_attention_; }
    [[nodiscard]] const latent::MultiHeadAttention& cross_attention() const noexcept { return cross_attention_; }
    [[nodiscard]] nn::RMSNorm& ffn_norm() noexcept { return ffn_norm_; }
    [[nodiscard]] const nn::RMSNorm& ffn_norm() const noexcept { return ffn_norm_; }
    [[nodiscard]] nn::Linear& ffn_in() noexcept { return ffn_in_; }
    [[nodiscard]] const nn::Linear& ffn_in() const noexcept { return ffn_in_; }
    [[nodiscard]] nn::Linear& ffn_out() noexcept { return ffn_out_; }
    [[nodiscard]] const nn::Linear& ffn_out() const noexcept { return ffn_out_; }
    [[nodiscard]] float residual_scale() const noexcept { return residual_scale_; }

private:
    nn::RMSNorm self_norm_;
    latent::MultiHeadAttention self_attention_;
    nn::RMSNorm cross_norm_;
    latent::MultiHeadAttention cross_attention_;
    nn::RMSNorm ffn_norm_;
    nn::Linear ffn_in_;
    nn::Linear ffn_out_;
    float residual_scale_;
};

class StableLatentTransformerDenoiser final : public flow::LatentPredictor {
public:
    StableLatentTransformerDenoiser(StableLatentTransformerConfig config, Random& rng);

    [[nodiscard]] std::size_t latent_dim() const noexcept override { return config_.latent_dim; }
    [[nodiscard]] Tensor predict(
        const Tensor& noisy_latent,
        std::string_view prompt,
        float time,
        std::size_t grid_height,
        std::size_t grid_width) const override;

    [[nodiscard]] std::vector<nn::Parameter*> parameters();
    [[nodiscard]] std::vector<const nn::Parameter*> parameters() const;
    [[nodiscard]] const StableLatentTransformerConfig& config() const noexcept { return config_; }
    [[nodiscard]] float effective_residual_scale() const noexcept { return residual_scale_; }

    [[nodiscard]] nn::Linear& input_projection() noexcept { return input_projection_; }
    [[nodiscard]] const nn::Linear& input_projection() const noexcept { return input_projection_; }
    [[nodiscard]] nn::Linear& prompt_projection() noexcept { return prompt_projection_; }
    [[nodiscard]] const nn::Linear& prompt_projection() const noexcept { return prompt_projection_; }
    [[nodiscard]] StableLatentTransformerBlock& block(std::size_t index);
    [[nodiscard]] const StableLatentTransformerBlock& block(std::size_t index) const;
    [[nodiscard]] nn::RMSNorm& final_norm() noexcept { return final_norm_; }
    [[nodiscard]] const nn::RMSNorm& final_norm() const noexcept { return final_norm_; }
    [[nodiscard]] nn::Linear& output_projection() noexcept { return output_projection_; }
    [[nodiscard]] const nn::Linear& output_projection() const noexcept { return output_projection_; }

private:
    [[nodiscard]] Tensor latent_conditioning(
        const Tensor& noisy_latent,
        float time,
        std::size_t grid_height,
        std::size_t grid_width) const;

    StableLatentTransformerConfig config_;
    float residual_scale_ = 1.0F;
    nn::Linear input_projection_;
    nn::Linear prompt_projection_;
    std::vector<StableLatentTransformerBlock> blocks_;
    nn::RMSNorm final_norm_;
    nn::Linear output_projection_;
};

[[nodiscard]] latent::ImagePromptDataset load_manifest_tsv(const std::string& manifest_path);
[[nodiscard]] std::vector<std::size_t> shuffled_indices(std::size_t size, std::uint64_t seed);

struct DatasetSplit {
    latent::ImagePromptDataset train;
    latent::ImagePromptDataset validation;
};

[[nodiscard]] DatasetSplit split_dataset(
    const latent::ImagePromptDataset& dataset,
    float validation_fraction,
    std::uint64_t seed);

class StatefulAdamW final {
public:
    StatefulAdamW(std::vector<nn::Parameter*> parameters, train::AdamWConfig config = {});

    void zero_grad();
    void step();
    [[nodiscard]] std::uint64_t step_count() const noexcept { return step_count_; }
    [[nodiscard]] const train::AdamWConfig& config() const noexcept { return config_; }

    void save(std::ostream& out) const;
    void load(std::istream& in);

private:
    struct State {
        nn::Parameter* parameter = nullptr;
        Tensor first_moment;
        Tensor second_moment;
    };

    train::AdamWConfig config_;
    std::vector<State> states_;
    std::uint64_t step_count_ = 0;
};

struct ScaleTrainerConfig {
    train::AdamWConfig optimizer{0.003F, 0.9F, 0.999F, 1.0e-8F, 0.0F};
    float max_grad_norm = 3.0F;
    float training_time = 0.65F;
    std::size_t micro_batch_size = 2;
    std::size_t gradient_accumulation_steps = 2;
    std::uint64_t shuffle_seed = 0x4C313253485546ULL;
    std::uint64_t noise_seed = 0x4C31324E4F4953ULL;
};

struct ScaleMetrics {
    std::uint64_t epoch = 0;
    std::uint64_t optimizer_steps = 0;
    float train_loss = 0.0F;
    float validation_loss = 0.0F;
    float last_grad_norm = 0.0F;
};

class ScaleTrainer final {
public:
    ScaleTrainer(
        StableLatentTransformerDenoiser& model,
        const multimodal::ImageAutoencoder& autoencoder,
        flow::NoiseScheduler scheduler = flow::NoiseScheduler{},
        ScaleTrainerConfig config = {});

    [[nodiscard]] float evaluate_dataset(
        const latent::ImagePromptDataset& dataset,
        std::uint64_t noise_seed) const;

    ScaleMetrics train_epoch(
        const latent::ImagePromptDataset& train_dataset,
        const latent::ImagePromptDataset& validation_dataset);

    [[nodiscard]] std::uint64_t epoch() const noexcept { return epoch_; }
    [[nodiscard]] const ScaleTrainerConfig& config() const noexcept { return config_; }
    [[nodiscard]] StatefulAdamW& optimizer() noexcept { return optimizer_; }
    [[nodiscard]] const StatefulAdamW& optimizer() const noexcept { return optimizer_; }
    void set_epoch(std::uint64_t epoch) noexcept { epoch_ = epoch; }

private:
    StableLatentTransformerDenoiser& model_;
    const multimodal::ImageAutoencoder& autoencoder_;
    flow::NoiseScheduler scheduler_;
    ScaleTrainerConfig config_;
    StatefulAdamW optimizer_;
    std::uint64_t epoch_ = 0;
};

void save_training_checkpoint(
    const StableLatentTransformerDenoiser& model,
    const ScaleTrainer& trainer,
    const std::string& path);

void load_training_checkpoint(
    StableLatentTransformerDenoiser& model,
    ScaleTrainer& trainer,
    const std::string& path);

void append_metrics_csv(const std::string& path, const ScaleMetrics& metrics);

} // namespace spiral::scale
