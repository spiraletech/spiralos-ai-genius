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
    [[nodiscard]] Linear& gate_proj() noexcept { return gate_proj_; }
    [[nodiscard]] const Linear& gate_proj() const noexcept { return gate_proj_; }
    [[nodiscard]] Linear& up_proj() noexcept { return up_proj_; }
    [[nodiscard]] const Linear& up_proj() const noexcept { return up_proj_; }
    [[nodiscard]] Linear& down_proj() noexcept { return down_proj_; }
    [[nodiscard]] const Linear& down_proj() const noexcept { return down_proj_; }

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
    [[nodiscard]] RMSNorm& attention_norm() noexcept { return attention_norm_; }
    [[nodiscard]] const RMSNorm& attention_norm() const noexcept { return attention_norm_; }
    [[nodiscard]] CausalSelfAttention& attention() noexcept { return attention_; }
    [[nodiscard]] const CausalSelfAttention& attention() const noexcept { return attention_; }
    [[nodiscard]] RMSNorm& feed_forward_norm() noexcept { return feed_forward_norm_; }
    [[nodiscard]] const RMSNorm& feed_forward_norm() const noexcept { return feed_forward_norm_; }
    [[nodiscard]] GatedFeedForward& feed_forward() noexcept { return feed_forward_; }
    [[nodiscard]] const GatedFeedForward& feed_forward() const noexcept { return feed_forward_; }

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
    [[nodiscard]] Embedding& token_embedding() noexcept { return token_embedding_; }
    [[nodiscard]] const Embedding& token_embedding() const noexcept { return token_embedding_; }
    [[nodiscard]] std::vector<std::unique_ptr<TransformerBlock>>& blocks() noexcept { return blocks_; }
    [[nodiscard]] const std::vector<std::unique_ptr<TransformerBlock>>& blocks() const noexcept { return blocks_; }
    [[nodiscard]] RMSNorm& final_norm() noexcept { return final_norm_; }
    [[nodiscard]] const RMSNorm& final_norm() const noexcept { return final_norm_; }
    [[nodiscard]] Linear& lm_head() noexcept { return lm_head_; }
    [[nodiscard]] const Linear& lm_head() const noexcept { return lm_head_; }

private:
    ModelConfig config_;
    Embedding token_embedding_;
    std::vector<std::unique_ptr<TransformerBlock>> blocks_;
    RMSNorm final_norm_;
    Linear lm_head_;
};

} // namespace spiral::nn
