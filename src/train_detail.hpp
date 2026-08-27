#pragma once

#include "spiral/train.hpp"

namespace spiral::train::detail {

struct AttentionCache {
    Tensor input;
    Tensor q_linear;
    Tensor k_linear;
    Tensor v_linear;
    Tensor q;
    Tensor k;
    Tensor v;
    Tensor weights;
    Tensor context_heads;
    Tensor merged;
};

struct FeedForwardCache {
    Tensor input;
    Tensor gate;
    Tensor up;
    Tensor mixed;
};

struct BlockCache {
    Tensor input;
    Tensor attention_input;
    AttentionCache attention;
    Tensor residual;
    Tensor ffn_input;
    FeedForwardCache ffn;
    Tensor output;
};

void require_same_shape(const Tensor& a, const Tensor& b, const char* what);
void add_inplace(Tensor& destination, const Tensor& source);
void accumulate_grad(nn::Parameter& parameter, const Tensor& gradient);
Tensor linear_backward(nn::Linear& layer, const Tensor& input, const Tensor& grad_output);
Tensor rmsnorm_backward(nn::RMSNorm& norm, const Tensor& input, const Tensor& grad_output);
float silu(float x);
float silu_derivative(float x);
void apply_rotary_inverse_to_gradient(Tensor& gradient, float base = 10000.0F);
Tensor split_heads(const Tensor& projected, std::size_t heads, std::size_t head_dim);
Tensor merge_heads(const Tensor& input);

Tensor attention_forward(nn::CausalSelfAttention& attention, const Tensor& input, AttentionCache& cache);
Tensor attention_backward(nn::CausalSelfAttention& attention, const AttentionCache& cache, const Tensor& grad_output);

Tensor feed_forward_forward(nn::GatedFeedForward& ffn, const Tensor& input, FeedForwardCache& cache);
Tensor feed_forward_backward(nn::GatedFeedForward& ffn, const FeedForwardCache& cache, const Tensor& grad_output);
Tensor block_forward(nn::TransformerBlock& block, const Tensor& input, BlockCache& cache);
Tensor block_backward(nn::TransformerBlock& block, const BlockCache& cache, const Tensor& grad_output);
Tensor logits_cross_entropy_gradient(
    const Tensor& logits,
    std::span<const std::uint32_t> targets,
    float& loss_out);

} // namespace spiral::train::detail
