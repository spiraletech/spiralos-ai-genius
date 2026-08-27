#pragma once

#include "spiral/nn.hpp"
#include "spiral/random.hpp"
#include "spiral/tensor.hpp"

#include <cstddef>
#include <vector>

namespace spiral::nn {

void apply_rotary_inplace(Tensor& tensor, float base = 10000.0F);

class CausalSelfAttention final : public Module {
public:
    CausalSelfAttention(
        std::size_t model_dim,
        std::size_t num_heads,
        Random& rng,
        bool use_bias = false);

    [[nodiscard]] Tensor forward(const Tensor& input) const override;
    [[nodiscard]] std::vector<Parameter*> parameters() override;
    [[nodiscard]] std::vector<const Parameter*> parameters() const override;

    [[nodiscard]] std::size_t model_dim() const noexcept { return model_dim_; }
    [[nodiscard]] std::size_t num_heads() const noexcept { return num_heads_; }
    [[nodiscard]] std::size_t head_dim() const noexcept { return head_dim_; }

private:
    [[nodiscard]] Tensor split_heads(const Tensor& projected) const;
    [[nodiscard]] Tensor attend(const Tensor& q, const Tensor& k, const Tensor& v) const;
    [[nodiscard]] Tensor merge_heads(const Tensor& heads) const;

    std::size_t model_dim_;
    std::size_t num_heads_;
    std::size_t head_dim_;
    Linear q_proj_;
    Linear k_proj_;
    Linear v_proj_;
    Linear out_proj_;
};

} // namespace spiral::nn
