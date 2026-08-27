#pragma once

#include "spiral/flow.hpp"
#include "spiral/multimodal.hpp"
#include "spiral/nn.hpp"
#include "spiral/random.hpp"
#include "spiral/tensor.hpp"
#include "spiral/train.hpp"
#include "spiral/vision.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace spiral::latent {

[[nodiscard]] Tensor prompt_token_features(
    std::string_view prompt,
    std::size_t token_count,
    std::size_t feature_dim);

class MultiHeadAttention final {
public:
    MultiHeadAttention(std::size_t model_dim, std::size_t num_heads, Random& rng);

    [[nodiscard]] Tensor forward(const Tensor& query, const Tensor& context) const;
    [[nodiscard]] std::vector<nn::Parameter*> parameters();
    [[nodiscard]] std::vector<const nn::Parameter*> parameters() const;

    [[nodiscard]] std::size_t model_dim() const noexcept { return model_dim_; }
    [[nodiscard]] std::size_t num_heads() const noexcept { return num_heads_; }
    [[nodiscard]] std::size_t head_dim() const noexcept { return head_dim_; }
    [[nodiscard]] nn::Linear& q_proj() noexcept { return q_proj_; }
    [[nodiscard]] const nn::Linear& q_proj() const noexcept { return q_proj_; }
    [[nodiscard]] nn::Linear& k_proj() noexcept { return k_proj_; }
    [[nodiscard]] const nn::Linear& k_proj() const noexcept { return k_proj_; }
    [[nodiscard]] nn::Linear& v_proj() noexcept { return v_proj_; }
    [[nodiscard]] const nn::Linear& v_proj() const noexcept { return v_proj_; }
    [[nodiscard]] nn::Linear& out_proj() noexcept { return out_proj_; }
    [[nodiscard]] const nn::Linear& out_proj() const noexcept { return out_proj_; }

private:
    std::size_t model_dim_;
    std::size_t num_heads_;
    std::size_t head_dim_;
    nn::Linear q_proj_;
    nn::Linear k_proj_;
    nn::Linear v_proj_;
    nn::Linear out_proj_;
};

class LatentTransformerBlock final {
public:
    LatentTransformerBlock(
        std::size_t model_dim,
        std::size_t num_heads,
        std::size_t ffn_dim,
        Random& rng);

    [[nodiscard]] Tensor forward(const Tensor& latent_tokens, const Tensor& prompt_tokens) const;
    [[nodiscard]] std::vector<nn::Parameter*> parameters();
    [[nodiscard]] std::vector<const nn::Parameter*> parameters() const;

    [[nodiscard]] MultiHeadAttention& self_attention() noexcept { return self_attention_; }
    [[nodiscard]] const MultiHeadAttention& self_attention() const noexcept { return self_attention_; }
    [[nodiscard]] MultiHeadAttention& cross_attention() noexcept { return cross_attention_; }
    [[nodiscard]] const MultiHeadAttention& cross_attention() const noexcept { return cross_attention_; }
    [[nodiscard]] nn::Linear& ffn_in() noexcept { return ffn_in_; }
    [[nodiscard]] const nn::Linear& ffn_in() const noexcept { return ffn_in_; }
    [[nodiscard]] nn::Linear& ffn_out() noexcept { return ffn_out_; }
    [[nodiscard]] const nn::Linear& ffn_out() const noexcept { return ffn_out_; }

private:
    MultiHeadAttention self_attention_;
    MultiHeadAttention cross_attention_;
    nn::Linear ffn_in_;
    nn::Linear ffn_out_;
};

struct LatentTransformerConfig {
    std::size_t latent_dim = 8;
    std::size_t model_dim = 32;
    std::size_t num_heads = 4;
    std::size_t num_layers = 2;
    std::size_t ffn_dim = 64;
    std::size_t text_feature_dim = 24;
    std::size_t prompt_tokens = 6;
    std::size_t time_feature_dim = 8;
};

class LatentTransformerDenoiser final : public flow::LatentPredictor {
public:
    LatentTransformerDenoiser(LatentTransformerConfig config, Random& rng);

    [[nodiscard]] std::size_t latent_dim() const noexcept override { return config_.latent_dim; }
    [[nodiscard]] Tensor predict(
        const Tensor& noisy_latent,
        std::string_view prompt,
        float time,
        std::size_t grid_height,
        std::size_t grid_width) const override;

    [[nodiscard]] std::vector<nn::Parameter*> parameters();
    [[nodiscard]] std::vector<const nn::Parameter*> parameters() const;
    [[nodiscard]] const LatentTransformerConfig& config() const noexcept { return config_; }

    [[nodiscard]] nn::Linear& input_projection() noexcept { return input_projection_; }
    [[nodiscard]] const nn::Linear& input_projection() const noexcept { return input_projection_; }
    [[nodiscard]] nn::Linear& prompt_projection() noexcept { return prompt_projection_; }
    [[nodiscard]] const nn::Linear& prompt_projection() const noexcept { return prompt_projection_; }
    [[nodiscard]] LatentTransformerBlock& block(std::size_t index);
    [[nodiscard]] const LatentTransformerBlock& block(std::size_t index) const;
    [[nodiscard]] nn::Linear& output_projection() noexcept { return output_projection_; }
    [[nodiscard]] const nn::Linear& output_projection() const noexcept { return output_projection_; }

private:
    [[nodiscard]] Tensor latent_conditioning(
        const Tensor& noisy_latent,
        float time,
        std::size_t grid_height,
        std::size_t grid_width) const;

    LatentTransformerConfig config_;
    nn::Linear input_projection_;
    nn::Linear prompt_projection_;
    std::vector<LatentTransformerBlock> blocks_;
    nn::Linear output_projection_;
};

struct ImagePromptExample {
    std::string prompt;
    vision::RgbImage image;
};

class ImagePromptDataset final {
public:
    void add(std::string prompt, vision::RgbImage image);
    [[nodiscard]] std::size_t size() const noexcept { return examples_.size(); }
    [[nodiscard]] const ImagePromptExample& at(std::size_t index) const;
    [[nodiscard]] const std::vector<ImagePromptExample>& examples() const noexcept { return examples_; }

private:
    std::vector<ImagePromptExample> examples_;
};

struct LatentTransformerTrainerConfig {
    train::AdamWConfig optimizer{0.005F, 0.9F, 0.999F, 1.0e-8F, 0.0F};
    float max_grad_norm = 5.0F;
    float training_time = 0.65F;
    std::uint64_t noise_seed = 0x4C3131545241494EULL;
};

class LatentTransformerTrainer final {
public:
    LatentTransformerTrainer(
        LatentTransformerDenoiser& model,
        const multimodal::ImageAutoencoder& autoencoder,
        flow::NoiseScheduler scheduler = flow::NoiseScheduler{},
        LatentTransformerTrainerConfig config = {});

    [[nodiscard]] float evaluate_example(
        const ImagePromptExample& example,
        std::uint64_t noise_seed) const;

    [[nodiscard]] float evaluate_dataset(
        const ImagePromptDataset& dataset,
        std::uint64_t noise_seed) const;

    float train_example(const ImagePromptExample& example, std::uint64_t noise_seed);
    float train_batch(const ImagePromptDataset& dataset, std::uint64_t noise_seed);
    float train_epoch(const ImagePromptDataset& dataset);

private:
    LatentTransformerDenoiser& model_;
    const multimodal::ImageAutoencoder& autoencoder_;
    flow::NoiseScheduler scheduler_;
    LatentTransformerTrainerConfig config_;
    train::AdamW optimizer_;
    std::uint64_t epoch_ = 0;
};

void save_latent_transformer(const LatentTransformerDenoiser& model, const std::string& path);
void load_latent_transformer(LatentTransformerDenoiser& model, const std::string& path);

} // namespace spiral::latent
