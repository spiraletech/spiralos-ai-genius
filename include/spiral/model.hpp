#pragma once

#include "spiral/attention.hpp"
#include "spiral/nn.hpp"
#include "spiral/random.hpp"
#include "spiral/tensor.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace spiral::nn {

struct ModelConfig {
    std::size_t vocabulary_size = 0;
    std::size_t model_dim = 0;
    std::size_t num_heads = 0;
    std::size_t num_layers = 0;
    std::size_t ffn_hidden_dim = 0;
    float norm_epsilon = 1.0e-5F;
};

class GatedFeedForward final : public Module {
public:
    GatedFeedForward(
        std::size_t model_dim,
        std::size_t hidden_dim,
        Random& rng,
        bool use_bias = false);

    [[nodiscard]] Tensor forward(const Tensor& input) const override;
    [[nodiscard]] std::vector<Parameter*> parameters() override;
    [[nodiscard]] std::vector<const Parameter*> parameters() const override;

private:
    Linear gate_proj_;
    Linear up_proj_;
    Linear down_proj_;
};

class TransformerBlock final : public Module {
public:
    TransformerBlock(
        std::size_t model_dim,
        std::size_t num_heads,
        std::size_t ffn_hidden_dim,
        Random& rng,
        float norm_epsilon = 1.0e-5F);

    [[nodiscard]] Tensor forward(const Tensor& input) const override;
    [[nodiscard]] std::vector<Parameter*> parameters() override;
    [[nodiscard]] std::vector<const Parameter*> parameters() const override;

private:
    RMSNorm attention_norm_;
    CausalSelfAttention attention_;
    RMSNorm feed_forward_norm_;
    GatedFeedForward feed_forward_;
};

class SpiralLanguageModel final {
public:
    SpiralLanguageModel(ModelConfig config, Random& rng);

    [[nodiscard]] Tensor forward(std::span<const std::uint32_t> token_ids) const;
    [[nodiscard]] Tensor last_token_logits(std::span<const std::uint32_t> token_ids) const;

    [[nodiscard]] std::vector<Parameter*> parameters();
    [[nodiscard]] std::vector<const Parameter*> parameters() const;

    [[nodiscard]] const ModelConfig& config() const noexcept { return config_; }
    [[nodiscard]] std::size_t parameter_count() const;

private:
    ModelConfig config_;
    Embedding token_embedding_;
    std::vector<std::unique_ptr<TransformerBlock>> blocks_;
    RMSNorm final_norm_;
    Linear lm_head_;
};

} // namespace spiral::nn
