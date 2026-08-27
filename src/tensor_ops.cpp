#include "spiral/tensor_ops.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace spiral::ops {

Tensor reshape_copy(const Tensor& input, std::vector<std::size_t> shape) {
    std::size_t count = shape.empty() ? 0 : 1;
    for (const auto dim : shape) {
        if (dim == 0) {
            count = 0;
            break;
        }
        if (count > std::numeric_limits<std::size_t>::max() / dim) {
            throw std::overflow_error("reshape shape overflows size_t");
        }
        count *= dim;
    }

    if (count != input.numel()) {
        throw std::invalid_argument("reshape_copy element count mismatch");
    }
    return Tensor(std::move(shape), input.data());
}

Tensor batched_matmul(const Tensor& lhs, const Tensor& rhs) {
    if (lhs.rank() != 3 || rhs.rank() != 3) {
        throw std::invalid_argument("batched_matmul requires rank-3 tensors");
    }

    const auto batch = lhs.shape()[0];
    const auto m = lhs.shape()[1];
    const auto k = lhs.shape()[2];
    if (rhs.shape()[0] != batch || rhs.shape()[1] != k) {
        throw std::invalid_argument("batched_matmul dimension mismatch");
    }
    const auto n = rhs.shape()[2];

    Tensor out({batch, m, n});
    for (std::size_t b = 0; b < batch; ++b) {
        for (std::size_t row = 0; row < m; ++row) {
            for (std::size_t col = 0; col < n; ++col) {
                float sum = 0.0F;
                for (std::size_t inner = 0; inner < k; ++inner) {
                    sum += lhs.data()[(b * m + row) * k + inner]
                        * rhs.data()[(b * k + inner) * n + col];
                }
                out.data()[(b * m + row) * n + col] = sum;
            }
        }
    }
    return out;
}

Tensor softmax_last_dim(const Tensor& input) {
    if (input.rank() == 0) {
        throw std::invalid_argument("softmax_last_dim requires rank >= 1");
    }

    const auto width = input.shape().back();
    if (width == 0) {
        return Tensor(input.shape());
    }

    const auto rows = input.numel() / width;
    Tensor out(input.shape());
    for (std::size_t row = 0; row < rows; ++row) {
        const auto base = row * width;
        float max_value = input.data()[base];
        for (std::size_t col = 1; col < width; ++col) {
            max_value = std::max(max_value, input.data()[base + col]);
        }

        float sum = 0.0F;
        for (std::size_t col = 0; col < width; ++col) {
            const float value = std::exp(input.data()[base + col] - max_value);
            out.data()[base + col] = value;
            sum += value;
        }
        if (sum == 0.0F || !std::isfinite(sum)) {
            throw std::runtime_error("softmax_last_dim normalization failed");
        }
        for (std::size_t col = 0; col < width; ++col) {
            out.data()[base + col] /= sum;
        }
    }
    return out;
}

Tensor causal_mask(std::size_t sequence_length) {
    Tensor out({sequence_length, sequence_length});
    const float blocked = -std::numeric_limits<float>::infinity();
    for (std::size_t row = 0; row < sequence_length; ++row) {
        for (std::size_t col = 0; col < sequence_length; ++col) {
            out.data()[row * sequence_length + col] = col <= row ? 0.0F : blocked;
        }
    }
    return out;
}

} // namespace spiral::ops
