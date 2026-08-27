#pragma once

#include "spiral/tensor.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace spiral::precision {

enum class NumericFormat : std::uint8_t {
    Float32 = 0,
    Float16 = 1,
    BFloat16 = 2,
    Int8Symmetric = 3,
};

[[nodiscard]] std::uint16_t float_to_float16(float value) noexcept;
[[nodiscard]] float float16_to_float(std::uint16_t value) noexcept;
[[nodiscard]] std::uint16_t float_to_bfloat16(float value) noexcept;
[[nodiscard]] float bfloat16_to_float(std::uint16_t value) noexcept;

struct QuantizedTensor {
    std::vector<std::size_t> shape;
    float scale = 1.0F;
    std::vector<std::int8_t> values;

    [[nodiscard]] Tensor dequantize() const;
    [[nodiscard]] std::size_t numel() const noexcept { return values.size(); }
};

[[nodiscard]] QuantizedTensor quantize_symmetric_int8(const Tensor& tensor);

} // namespace spiral::precision
