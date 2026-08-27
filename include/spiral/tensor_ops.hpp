#pragma once

#include "spiral/tensor.hpp"

#include <cstddef>
#include <vector>

namespace spiral::ops {

[[nodiscard]] Tensor reshape_copy(const Tensor& input, std::vector<std::size_t> shape);
[[nodiscard]] Tensor batched_matmul(const Tensor& lhs, const Tensor& rhs);
[[nodiscard]] Tensor softmax_last_dim(const Tensor& input);
[[nodiscard]] Tensor causal_mask(std::size_t sequence_length);

} // namespace spiral::ops
