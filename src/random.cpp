#include "spiral/random.hpp"

#include "spiral/tensor.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace spiral {

Random::Random(std::uint64_t seed) noexcept
    : state_(seed) {}

std::uint64_t Random::next_u64() noexcept {
    state_ += 0x9E3779B97F4A7C15ULL;
    std::uint64_t z = state_;
    z = (z ^ (z >> 30U)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27U)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31U);
}

float Random::uniform(float low, float high) noexcept {
    const auto bits = next_u64() >> 40U;
    constexpr float scale = 1.0F / static_cast<float>(1U << 24U);
    const float unit = static_cast<float>(bits) * scale;
    return low + (high - low) * unit;
}

float Random::normal(float mean, float stddev) {
    constexpr float two_pi = 6.28318530717958647692F;
    const float u1 = std::max(uniform(), std::numeric_limits<float>::min());
    const float u2 = uniform();
    const float radius = std::sqrt(-2.0F * std::log(u1));
    return mean + stddev * radius * std::cos(two_pi * u2);
}

void Random::fill_uniform(Tensor& tensor, float low, float high) noexcept {
    for (auto& value : tensor.data()) {
        value = uniform(low, high);
    }
}

void Random::fill_normal(Tensor& tensor, float mean, float stddev) {
    for (auto& value : tensor.data()) {
        value = normal(mean, stddev);
    }
}

} // namespace spiral
