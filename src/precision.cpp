#include "spiral/precision.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace spiral::precision {

std::uint16_t float_to_float16(float value) noexcept {
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    const std::uint32_t sign = (bits >> 16U) & 0x8000U;
    const std::uint32_t exponent = (bits >> 23U) & 0xFFU;
    const std::uint32_t mantissa = bits & 0x7FFFFFU;

    if (exponent == 0xFFU) {
        if (mantissa == 0U) return static_cast<std::uint16_t>(sign | 0x7C00U);
        return static_cast<std::uint16_t>(sign | 0x7E00U);
    }

    const int adjusted = static_cast<int>(exponent) - 127 + 15;
    if (adjusted >= 31) return static_cast<std::uint16_t>(sign | 0x7C00U);
    if (adjusted <= 0) {
        if (adjusted < -10) return static_cast<std::uint16_t>(sign);
        std::uint32_t subnormal = mantissa | 0x800000U;
        const int shift = 14 - adjusted;
        std::uint32_t rounded = subnormal >> shift;
        const std::uint32_t remainder = subnormal & ((1U << shift) - 1U);
        const std::uint32_t halfway = 1U << (shift - 1);
        if (remainder > halfway || (remainder == halfway && (rounded & 1U) != 0U)) ++rounded;
        return static_cast<std::uint16_t>(sign | rounded);
    }

    std::uint32_t half_mantissa = mantissa >> 13U;
    const std::uint32_t remainder = mantissa & 0x1FFFU;
    if (remainder > 0x1000U || (remainder == 0x1000U && (half_mantissa & 1U) != 0U)) {
        ++half_mantissa;
        if (half_mantissa == 0x400U) {
            half_mantissa = 0;
            if (adjusted + 1 >= 31) return static_cast<std::uint16_t>(sign | 0x7C00U);
            return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(adjusted + 1) << 10U));
        }
    }
    return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(adjusted) << 10U) | half_mantissa);
}

float float16_to_float(std::uint16_t value) noexcept {
    const std::uint32_t sign = (static_cast<std::uint32_t>(value & 0x8000U)) << 16U;
    const std::uint32_t exponent = (value >> 10U) & 0x1FU;
    const std::uint32_t mantissa = value & 0x3FFU;
    std::uint32_t bits = 0;

    if (exponent == 0U) {
        if (mantissa == 0U) {
            bits = sign;
        } else {
            std::uint32_t m = mantissa;
            int e = -14;
            while ((m & 0x400U) == 0U) {
                m <<= 1U;
                --e;
            }
            m &= 0x3FFU;
            bits = sign | (static_cast<std::uint32_t>(e + 127) << 23U) | (m << 13U);
        }
    } else if (exponent == 0x1FU) {
        bits = sign | 0x7F800000U | (mantissa << 13U);
    } else {
        bits = sign | ((exponent - 15U + 127U) << 23U) | (mantissa << 13U);
    }
    return std::bit_cast<float>(bits);
}

std::uint16_t float_to_bfloat16(float value) noexcept {
    std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    const std::uint32_t lsb = (bits >> 16U) & 1U;
    bits += 0x7FFFU + lsb;
    return static_cast<std::uint16_t>(bits >> 16U);
}

float bfloat16_to_float(std::uint16_t value) noexcept {
    return std::bit_cast<float>(static_cast<std::uint32_t>(value) << 16U);
}

Tensor QuantizedTensor::dequantize() const {
    Tensor tensor(shape);
    if (tensor.numel() != values.size()) throw std::runtime_error("quantized tensor shape/value mismatch");
    for (std::size_t i = 0; i < values.size(); ++i) {
        tensor.data()[i] = static_cast<float>(values[i]) * scale;
    }
    return tensor;
}

QuantizedTensor quantize_symmetric_int8(const Tensor& tensor) {
    QuantizedTensor result;
    result.shape = tensor.shape();
    result.values.resize(tensor.numel());
    if (tensor.numel() == 0) return result;

    float maximum = 0.0F;
    for (const float value : tensor.data()) maximum = std::max(maximum, std::fabs(value));
    result.scale = maximum > 0.0F ? maximum / 127.0F : 1.0F;
    for (std::size_t i = 0; i < tensor.numel(); ++i) {
        const float scaled = tensor.data()[i] / result.scale;
        const int rounded = static_cast<int>(std::lrint(scaled));
        result.values[i] = static_cast<std::int8_t>(std::clamp(rounded, -127, 127));
    }
    return result;
}

} // namespace spiral::precision
