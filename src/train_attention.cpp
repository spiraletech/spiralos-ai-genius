#include "train_detail.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace spiral::train::detail {

Tensor attention_forward(nn::CausalSelfAttention& attention, const Tensor& input, AttentionCache& cache) {
    cache.input = input;
    cache.q_linear = attention.q_proj().forward(input);
    cache.k_linear = attention.k_proj().forward(input);
    cache.v_linear = attention.v_proj().forward(input);

    const auto heads = attention.num_heads();
    const auto head_dim = attention.head_dim();
    const auto sequence = input.shape()[0];
    cache.q = split_heads(cache.q_linear, heads, head_dim);
    cache.k = split_heads(cache.k_linear, heads, head_dim);
    cache.v = split_heads(cache.v_linear, heads, head_dim);
    nn::apply_rotary_inplace(cache.q);
    nn::apply_rotary_inplace(cache.k);

    Tensor scores({heads, sequence, sequence});
    const float scale = 1.0F / std::sqrt(static_cast<float>(head_dim));
    for (std::size_t head = 0; head < heads; ++head) {
        for (std::size_t row = 0; row < sequence; ++row) {
            for (std::size_t col = 0; col < sequence; ++col) {
                float score = 0.0F;
                for (std::size_t dim = 0; dim < head_dim; ++dim) {
                    score += cache.q.data()[(head * sequence + row) * head_dim + dim]
                        * cache.k.data()[(head * sequence + col) * head_dim + dim];
                }
                scores.data()[(head * sequence + row) * sequence + col] =
                    col > row ? -std::numeric_limits<float>::infinity() : score * scale;
            }
        }
    }

    cache.weights = Tensor(scores.shape());
    for (std::size_t head = 0; head < heads; ++head) {
        for (std::size_t row = 0; row < sequence; ++row) {
            const std::size_t base_index = (head * sequence + row) * sequence;
            float maximum = scores.data()[base_index];
            for (std::size_t col = 1; col < sequence; ++col) {
                maximum = std::max(maximum, scores.data()[base_index + col]);
            }
            float sum = 0.0F;
            for (std::size_t col = 0; col < sequence; ++col) {
                const float value = std::exp(scores.data()[base_index + col] - maximum);
                cache.weights.data()[base_index + col] = value;
                sum += value;
            }
            for (std::size_t col = 0; col < sequence; ++col) {
                cache.weights.data()[base_index + col] /= sum;
            }
        }
    }

    cache.context_heads = Tensor({heads, sequence, head_dim});
    for (std::size_t head = 0; head < heads; ++head) {
        for (std::size_t row = 0; row < sequence; ++row) {
            for (std::size_t dim = 0; dim < head_dim; ++dim) {
                float sum = 0.0F;
                for (std::size_t col = 0; col < sequence; ++col) {
                    sum += cache.weights.data()[(head * sequence + row) * sequence + col]
                        * cache.v.data()[(head * sequence + col) * head_dim + dim];
                }
                cache.context_heads.data()[(head * sequence + row) * head_dim + dim] = sum;
            }
        }
    }

    cache.merged = merge_heads(cache.context_heads);
    return attention.out_proj().forward(cache.merged);
}

Tensor attention_backward(nn::CausalSelfAttention& attention, const AttentionCache& cache, const Tensor& grad_output) {
    const auto heads = attention.num_heads();
    const auto head_dim = attention.head_dim();
    const auto sequence = cache.input.shape()[0];
    const float scale = 1.0F / std::sqrt(static_cast<float>(head_dim));

    const Tensor grad_merged = linear_backward(attention.out_proj(), cache.merged, grad_output);
    const Tensor grad_context = split_heads(grad_merged, heads, head_dim);

    Tensor grad_weights({heads, sequence, sequence});
    Tensor grad_v({heads, sequence, head_dim});
    for (std::size_t head = 0; head < heads; ++head) {
        for (std::size_t row = 0; row < sequence; ++row) {
            for (std::size_t col = 0; col < sequence; ++col) {
                float dot = 0.0F;
                for (std::size_t dim = 0; dim < head_dim; ++dim) {
                    const float gc = grad_context.data()[(head * sequence + row) * head_dim + dim];
                    dot += gc * cache.v.data()[(head * sequence + col) * head_dim + dim];
                    grad_v.data()[(head * sequence + col) * head_dim + dim] +=
                        cache.weights.data()[(head * sequence + row) * sequence + col] * gc;
                }
                grad_weights.data()[(head * sequence + row) * sequence + col] = dot;
            }
        }
    }

    Tensor grad_scores({heads, sequence, sequence});
    for (std::size_t head = 0; head < heads; ++head) {
        for (std::size_t row = 0; row < sequence; ++row) {
            const std::size_t base_index = (head * sequence + row) * sequence;
            float weighted_sum = 0.0F;
            for (std::size_t col = 0; col < sequence; ++col) {
                weighted_sum += grad_weights.data()[base_index + col] * cache.weights.data()[base_index + col];
            }
            for (std::size_t col = 0; col < sequence; ++col) {
                const float w = cache.weights.data()[base_index + col];
                grad_scores.data()[base_index + col] = w * (grad_weights.data()[base_index + col] - weighted_sum);
            }
        }
    }

    Tensor grad_q({heads, sequence, head_dim});
    Tensor grad_k({heads, sequence, head_dim});
    for (std::size_t head = 0; head < heads; ++head) {
        for (std::size_t row = 0; row < sequence; ++row) {
            for (std::size_t col = 0; col < sequence; ++col) {
                const float gs = grad_scores.data()[(head * sequence + row) * sequence + col] * scale;
                for (std::size_t dim = 0; dim < head_dim; ++dim) {
                    grad_q.data()[(head * sequence + row) * head_dim + dim] +=
                        gs * cache.k.data()[(head * sequence + col) * head_dim + dim];
                    grad_k.data()[(head * sequence + col) * head_dim + dim] +=
                        gs * cache.q.data()[(head * sequence + row) * head_dim + dim];
                }
            }
        }
    }

    apply_rotary_inverse_to_gradient(grad_q);
    apply_rotary_inverse_to_gradient(grad_k);

    const Tensor grad_q_linear = merge_heads(grad_q);
    const Tensor grad_k_linear = merge_heads(grad_k);
    const Tensor grad_v_linear = merge_heads(grad_v);

    Tensor grad_input = linear_backward(attention.q_proj(), cache.input, grad_q_linear);
    add_inplace(grad_input, linear_backward(attention.k_proj(), cache.input, grad_k_linear));
    add_inplace(grad_input, linear_backward(attention.v_proj(), cache.input, grad_v_linear));
    return grad_input;
}

} // namespace spiral::train::detail
