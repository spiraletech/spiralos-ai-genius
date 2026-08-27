#include "spiral/attention.hpp"

#include "spiral/tensor_ops.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace spiral::nn {

void apply_rotary_inplace(Tensor& tensor, float base) {
    if (tensor.rank() != 3) {
        throw std::invalid_argument("rotary embedding expects [heads, sequence, head_dim]");
    }
    if (base <= 1.0F) {
        throw std::invalid_argument("rotary embedding base must be > 1");
    }

    const auto heads = tensor.shape()[0];
    const auto sequence = tensor.shape()[1];
    const auto dim = tensor.shape()[2];
    if (dim == 0 || dim % 2 != 0) {
        throw std::invalid_argument("rotary embedding requires non-zero even head dimension");
    }

    for (std::size_t head = 0; head < heads; ++head) {
        for (std::size_t position = 0; position < sequence; ++position) {
            for (std::size_t pair = 0; pair < dim; pair += 2) {
                const float exponent = static_cast<float>(pair) / static_cast<float>(dim);
                const float theta = static_cast<float>(position) / std::pow(base, exponent);
                const float cosine = std::cos(theta);
                const float sine = std::sin(theta);
                const auto offset = (head * sequence + position) * dim + pair;
                const float x = tensor.data()[offset];
                const float y = tensor.data()[offset + 1];
                tensor.data()[offset] = x * cosine - y * sine;
                tensor.data()[offset + 1] = x * sine + y * cosine;
            }
        }
    }
}

CausalSelfAttention::CausalSelfAttention(
    std::size_t model_dim,
    std::size_t num_heads,
    Random& rng,
    bool use_bias)
    : model_dim_(model_dim),
      num_heads_(num_heads),
      head_dim_(num_heads == 0 ? 0 : model_dim / num_heads),
      q_proj_(model_dim, model_dim, rng, use_bias),
      k_proj_(model_dim, model_dim, rng, use_bias),
      v_proj_(model_dim, model_dim, rng, use_bias),
      out_proj_(model_dim, model_dim, rng, use_bias) {
    if (model_dim == 0 || num_heads == 0 || model_dim % num_heads != 0) {
        throw std::invalid_argument("attention requires model_dim divisible by non-zero num_heads");
    }
    if (head_dim_ % 2 != 0) {
        throw std::invalid_argument("attention head_dim must be even for rotary embedding");
    }
}

Tensor CausalSelfAttention::split_heads(const Tensor& projected) const {
    if (projected.rank() != 2 || projected.shape()[1] != model_dim_) {
        throw std::invalid_argument("attention projection shape mismatch");
    }

    const auto sequence = projected.shape()[0];
    Tensor out({num_heads_, sequence, head_dim_});
    for (std::size_t token = 0; token < sequence; ++token) {
        for (std::size_t head = 0; head < num_heads_; ++head) {
            for (std::size_t dim = 0; dim < head_dim_; ++dim) {
                out.data()[(head * sequence + token) * head_dim_ + dim] =
                    projected.data()[token * model_dim_ + head * head_dim_ + dim];
            }
        }
    }
    return out;
}

Tensor CausalSelfAttention::attend(const Tensor& q, const Tensor& k, const Tensor& v) const {
    const auto sequence = q.shape()[1];
    Tensor scores({num_heads_, sequence, sequence});
    const float scale = 1.0F / std::sqrt(static_cast<float>(head_dim_));

    for (std::size_t head = 0; head < num_heads_; ++head) {
        for (std::size_t row = 0; row < sequence; ++row) {
            for (std::size_t col = 0; col < sequence; ++col) {
                float score = 0.0F;
                for (std::size_t dim = 0; dim < head_dim_; ++dim) {
                    score += q.data()[(head * sequence + row) * head_dim_ + dim]
                        * k.data()[(head * sequence + col) * head_dim_ + dim];
                }
                scores.data()[(head * sequence + row) * sequence + col] =
                    col > row ? -std::numeric_limits<float>::infinity() : score * scale;
            }
        }
    }

    const Tensor weights = ops::softmax_last_dim(scores);
    Tensor context({num_heads_, sequence, head_dim_});
    for (std::size_t head = 0; head < num_heads_; ++head) {
        for (std::size_t row = 0; row < sequence; ++row) {
            for (std::size_t dim = 0; dim < head_dim_; ++dim) {
                float sum = 0.0F;
                for (std::size_t col = 0; col < sequence; ++col) {
                    sum += weights.data()[(head * sequence + row) * sequence + col]
                        * v.data()[(head * sequence + col) * head_dim_ + dim];
                }
                context.data()[(head * sequence + row) * head_dim_ + dim] = sum;
            }
        }
    }
    return context;
}

Tensor CausalSelfAttention::merge_heads(const Tensor& heads) const {
    const auto sequence = heads.shape()[1];
    Tensor out({sequence, model_dim_});
    for (std::size_t token = 0; token < sequence; ++token) {
        for (std::size_t head = 0; head < num_heads_; ++head) {
            for (std::size_t dim = 0; dim < head_dim_; ++dim) {
                out.data()[token * model_dim_ + head * head_dim_ + dim] =
                    heads.data()[(head * sequence + token) * head_dim_ + dim];
            }
        }
    }
    return out;
}

Tensor CausalSelfAttention::forward(const Tensor& input) const {
    if (input.rank() != 2 || input.shape()[1] != model_dim_) {
        throw std::invalid_argument("CausalSelfAttention expects [sequence, model_dim]");
    }

    Tensor q = split_heads(q_proj_.forward(input));
    Tensor k = split_heads(k_proj_.forward(input));
    const Tensor v = split_heads(v_proj_.forward(input));
    apply_rotary_inplace(q);
    apply_rotary_inplace(k);

    return out_proj_.forward(merge_heads(attend(q, k, v)));
}

std::vector<Parameter*> CausalSelfAttention::parameters() {
    std::vector<Parameter*> out;
    for (auto* layer : {&q_proj_, &k_proj_, &v_proj_, &out_proj_}) {
        auto child = layer->parameters();
        out.insert(out.end(), child.begin(), child.end());
    }
    return out;
}

std::vector<const Parameter*> CausalSelfAttention::parameters() const {
    std::vector<const Parameter*> out;
    for (const auto* layer : {&q_proj_, &k_proj_, &v_proj_, &out_proj_}) {
        auto child = layer->parameters();
        out.insert(out.end(), child.begin(), child.end());
    }
    return out;
}

} // namespace spiral::nn
