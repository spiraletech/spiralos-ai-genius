#include "spiral/model.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace spiral::nn {
namespace {

float silu(float value) {
    return value / (1.0F + std::exp(-value));
}

void append_parameters(std::vector<Parameter*>& out, Module& module) {
    auto child = module.parameters();
    out.insert(out.end(), child.begin(), child.end());
}

void append_parameters(std::vector<const Parameter*>& out, const Module& module) {
    auto child = module.parameters();
    out.insert(out.end(), child.begin(), child.end());
}

void validate_model_config(const ModelConfig& config) {
    if (config.vocabulary_size == 0 || config.model_dim == 0 || config.num_heads == 0
        || config.num_layers == 0 || config.ffn_hidden_dim == 0) {
        throw std::invalid_argument("SpiralLanguageModel dimensions must be non-zero");
    }
    if (config.model_dim % config.num_heads != 0) {
        throw std::invalid_argument("SpiralLanguageModel model_dim must be divisible by num_heads");
    }
    if ((config.model_dim / config.num_heads) % 2 != 0) {
        throw std::invalid_argument("SpiralLanguageModel attention head dimension must be even");
    }
    if (config.norm_epsilon <= 0.0F) {
        throw std::invalid_argument("SpiralLanguageModel norm epsilon must be positive");
    }
}

} // namespace

GatedFeedForward::GatedFeedForward(
    std::size_t model_dim,
    std::size_t hidden_dim,
    Random& rng,
    bool use_bias)
    : gate_proj_(model_dim, hidden_dim, rng, use_bias),
      up_proj_(model_dim, hidden_dim, rng, use_bias),
      down_proj_(hidden_dim, model_dim, rng, use_bias) {}

Tensor GatedFeedForward::forward(const Tensor& input) const {
    const Tensor gate = gate_proj_.forward(input);
    const Tensor up = up_proj_.forward(input);
    if (gate.shape() != up.shape()) {
        throw std::runtime_error("GatedFeedForward projection shape mismatch");
    }

    Tensor mixed(gate.shape());
    for (std::size_t i = 0; i < mixed.numel(); ++i) {
        mixed.data()[i] = silu(gate.data()[i]) * up.data()[i];
    }
    return down_proj_.forward(mixed);
}

std::vector<Parameter*> GatedFeedForward::parameters() {
    std::vector<Parameter*> out;
    append_parameters(out, gate_proj_);
    append_parameters(out, up_proj_);
    append_parameters(out, down_proj_);
    return out;
}

std::vector<const Parameter*> GatedFeedForward::parameters() const {
    std::vector<const Parameter*> out;
    append_parameters(out, gate_proj_);
    append_parameters(out, up_proj_);
    append_parameters(out, down_proj_);
    return out;
}

TransformerBlock::TransformerBlock(
    std::size_t model_dim,
    std::size_t num_heads,
    std::size_t ffn_hidden_dim,
    Random& rng,
    float norm_epsilon)
    : attention_norm_(model_dim, norm_epsilon),
      attention_(model_dim, num_heads, rng, false),
      feed_forward_norm_(model_dim, norm_epsilon),
      feed_forward_(model_dim, ffn_hidden_dim, rng, false) {}

Tensor TransformerBlock::forward(const Tensor& input) const {
    if (input.rank() != 2) {
        throw std::invalid_argument("TransformerBlock expects [sequence, model_dim]");
    }

    const Tensor attention_input = attention_norm_.forward(input);
    const Tensor attention_output = attention_.forward(attention_input);
    const Tensor residual = input.add(attention_output);

    const Tensor feed_forward_input = feed_forward_norm_.forward(residual);
    const Tensor feed_forward_output = feed_forward_.forward(feed_forward_input);
    return residual.add(feed_forward_output);
}

std::vector<Parameter*> TransformerBlock::parameters() {
    std::vector<Parameter*> out;
    append_parameters(out, attention_norm_);
    append_parameters(out, attention_);
    append_parameters(out, feed_forward_norm_);
    append_parameters(out, feed_forward_);
    return out;
}

std::vector<const Parameter*> TransformerBlock::parameters() const {
    std::vector<const Parameter*> out;
    append_parameters(out, attention_norm_);
    append_parameters(out, attention_);
    append_parameters(out, feed_forward_norm_);
    append_parameters(out, feed_forward_);
    return out;
}

SpiralLanguageModel::SpiralLanguageModel(ModelConfig config, Random& rng)
    : config_(config),
      token_embedding_(config.vocabulary_size, config.model_dim, rng),
      final_norm_(config.model_dim, config.norm_epsilon),
      lm_head_(config.model_dim, config.vocabulary_size, rng, false) {
    validate_model_config(config_);
    blocks_.reserve(config_.num_layers);
    for (std::size_t layer = 0; layer < config_.num_layers; ++layer) {
        blocks_.push_back(std::make_unique<TransformerBlock>(
            config_.model_dim,
            config_.num_heads,
            config_.ffn_hidden_dim,
            rng,
            config_.norm_epsilon));
    }
}

Tensor SpiralLanguageModel::forward(std::span<const std::uint32_t> token_ids) const {
    if (token_ids.empty()) {
        throw std::invalid_argument("SpiralLanguageModel requires at least one token");
    }

    Tensor hidden = token_embedding_.forward(token_ids);
    for (const auto& block : blocks_) {
        hidden = block->forward(hidden);
    }
    hidden = final_norm_.forward(hidden);
    return lm_head_.forward(hidden);
}

Tensor SpiralLanguageModel::last_token_logits(std::span<const std::uint32_t> token_ids) const {
    const Tensor logits = forward(token_ids);
    const std::size_t vocabulary_size = config_.vocabulary_size;
    const std::size_t row = logits.shape()[0] - 1;
    std::vector<float> values(vocabulary_size);
    for (std::size_t token = 0; token < vocabulary_size; ++token) {
        values[token] = logits.data()[row * vocabulary_size + token];
    }
    return Tensor({vocabulary_size}, std::move(values));
}

std::vector<Parameter*> SpiralLanguageModel::parameters() {
    std::vector<Parameter*> out;
    out.push_back(&token_embedding_.table());
    for (auto& block : blocks_) {
        append_parameters(out, *block);
    }
    append_parameters(out, final_norm_);
    append_parameters(out, lm_head_);
    return out;
}

std::vector<const Parameter*> SpiralLanguageModel::parameters() const {
    std::vector<const Parameter*> out;
    out.push_back(&token_embedding_.table());
    for (const auto& block : blocks_) {
        append_parameters(out, *block);
    }
    append_parameters(out, final_norm_);
    append_parameters(out, lm_head_);
    return out;
}

std::size_t SpiralLanguageModel::parameter_count() const {
    std::size_t count = 0;
    for (const auto* parameter : parameters()) {
        count += parameter->value.numel();
    }
    return count;
}

} // namespace spiral::nn
