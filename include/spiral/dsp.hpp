#pragma once

#include "spiral/audio.hpp"
#include "spiral/tensor.hpp"

#include <complex>
#include <cstddef>
#include <span>
#include <vector>

namespace spiral::dsp {

[[nodiscard]] bool is_power_of_two(std::size_t value) noexcept;

void fft_inplace(std::vector<std::complex<float>>& values, bool inverse = false);

[[nodiscard]] std::vector<float> real_fft_magnitude(std::span<const float> samples);

[[nodiscard]] Tensor stft_magnitude_fft(
    const audio::AudioBuffer& audio,
    audio::StftConfig config = {});

} // namespace spiral::dsp
