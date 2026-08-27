#include "spiral/dsp.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>

namespace spiral::dsp {

bool is_power_of_two(std::size_t value) noexcept {
    return value != 0 && (value & (value - 1)) == 0;
}

void fft_inplace(std::vector<std::complex<float>>& values, bool inverse) {
    const std::size_t n = values.size();
    if (!is_power_of_two(n)) throw std::invalid_argument("FFT size must be a non-zero power of two");

    for (std::size_t i = 1, j = 0; i < n; ++i) {
        std::size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(values[i], values[j]);
    }

    for (std::size_t length = 2; length <= n; length <<= 1) {
        const float angle = (inverse ? 2.0F : -2.0F) * std::numbers::pi_v<float> / static_cast<float>(length);
        const std::complex<float> root(std::cos(angle), std::sin(angle));
        for (std::size_t start = 0; start < n; start += length) {
            std::complex<float> w(1.0F, 0.0F);
            const std::size_t half = length / 2;
            for (std::size_t offset = 0; offset < half; ++offset) {
                const auto even = values[start + offset];
                const auto odd = values[start + offset + half] * w;
                values[start + offset] = even + odd;
                values[start + offset + half] = even - odd;
                w *= root;
            }
        }
    }

    if (inverse) {
        const float inv = 1.0F / static_cast<float>(n);
        for (auto& value : values) value *= inv;
    }
}

std::vector<float> real_fft_magnitude(std::span<const float> samples) {
    if (!is_power_of_two(samples.size())) throw std::invalid_argument("real FFT requires power-of-two sample count");
    std::vector<std::complex<float>> values(samples.size());
    for (std::size_t i = 0; i < samples.size(); ++i) values[i] = {samples[i], 0.0F};
    fft_inplace(values, false);
    std::vector<float> magnitude(samples.size() / 2 + 1);
    for (std::size_t i = 0; i < magnitude.size(); ++i) magnitude[i] = std::abs(values[i]);
    return magnitude;
}

Tensor stft_magnitude_fft(const audio::AudioBuffer& input, audio::StftConfig config) {
    if (!is_power_of_two(config.frame_size) || config.hop_size == 0 || config.hop_size > config.frame_size) {
        throw std::invalid_argument("FFT STFT requires power-of-two frame size and valid hop size");
    }
    const audio::AudioBuffer mono = input.mono();
    if (mono.frame_count() < config.frame_size) return Tensor({0, config.frame_size / 2 + 1});
    const std::size_t frames = 1 + (mono.frame_count() - config.frame_size) / config.hop_size;
    const std::size_t bins = config.frame_size / 2 + 1;
    Tensor spectrum({frames, bins});
    std::vector<float> window(config.frame_size);
    const float denom = static_cast<float>(config.frame_size - 1);

    for (std::size_t frame = 0; frame < frames; ++frame) {
        const std::size_t start = frame * config.hop_size;
        for (std::size_t i = 0; i < config.frame_size; ++i) {
            const float hann = config.frame_size == 1
                ? 1.0F
                : 0.5F - 0.5F * std::cos(2.0F * std::numbers::pi_v<float> * static_cast<float>(i) / denom);
            window[i] = mono.sample(start + i, 0) * hann;
        }
        const auto magnitude = real_fft_magnitude(window);
        for (std::size_t bin = 0; bin < bins; ++bin) spectrum.data()[frame * bins + bin] = magnitude[bin];
    }
    return spectrum;
}

} // namespace spiral::dsp
