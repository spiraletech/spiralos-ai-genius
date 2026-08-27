#pragma once

#include <cstdint>

namespace spiral {

class Tensor;

class Random {
public:
    explicit Random(std::uint64_t seed = 0x53504952414CULL) noexcept;

    [[nodiscard]] std::uint64_t next_u64() noexcept;
    [[nodiscard]] float uniform(float low = 0.0F, float high = 1.0F) noexcept;
    [[nodiscard]] float normal(float mean = 0.0F, float stddev = 1.0F);

    void fill_uniform(Tensor& tensor, float low = 0.0F, float high = 1.0F) noexcept;
    void fill_normal(Tensor& tensor, float mean = 0.0F, float stddev = 1.0F);

private:
    std::uint64_t state_;
};

} // namespace spiral
