#include "spiral/audio.hpp"
#include "spiral/dsp.hpp"
#include "spiral/temporal.hpp"
#include "spiral/vision.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstddef>
#include <iostream>
#include <numbers>
#include <vector>

namespace {

float max_abs_diff(const spiral::Tensor& a, const spiral::Tensor& b) {
    assert(a.shape() == b.shape());
    float result = 0.0F;
    for (std::size_t i = 0; i < a.numel(); ++i) result = std::max(result, std::abs(a.data()[i] - b.data()[i]));
    return result;
}

spiral::vision::RgbImage solid(std::uint8_t r, std::uint8_t g, std::uint8_t b) {
    spiral::vision::RgbImage image(4, 4);
    for (std::size_t y = 0; y < 4; ++y) {
        for (std::size_t x = 0; x < 4; ++x) {
            image.at(x, y, 0) = r;
            image.at(x, y, 1) = g;
            image.at(x, y, 2) = b;
        }
    }
    return image;
}

} // namespace

int main() {
    using namespace spiral;

    {
        std::vector<std::complex<float>> values{{1.0F, 0.0F}, {2.0F, 0.0F}, {3.0F, 0.0F}, {4.0F, 0.0F}};
        const auto original = values;
        dsp::fft_inplace(values, false);
        dsp::fft_inplace(values, true);
        for (std::size_t i = 0; i < values.size(); ++i) {
            assert(std::abs(values[i].real() - original[i].real()) < 1.0e-5F);
            assert(std::abs(values[i].imag()) < 1.0e-5F);
        }
    }

    audio::AudioBuffer tone;
    {
        constexpr std::uint32_t sample_rate = 8000;
        constexpr std::size_t sample_count = 512;
        std::vector<float> samples(sample_count);
        for (std::size_t i = 0; i < sample_count; ++i) {
            samples[i] = 0.7F * std::sin(2.0F * std::numbers::pi_v<float> * 1000.0F * static_cast<float>(i) / static_cast<float>(sample_rate));
        }
        tone = audio::AudioBuffer(sample_rate, 1, std::move(samples));
        const audio::StftConfig config{64, 32};
        const Tensor reference = audio::stft_magnitude(tone, config);
        const Tensor fast = dsp::stft_magnitude_fft(tone, config);
        assert(reference.shape() == fast.shape());
        assert(max_abs_diff(reference, fast) < 5.0e-3F);
        const auto first = fast.data().begin();
        const auto last = first + static_cast<std::ptrdiff_t>(fast.shape()[1]);
        const std::size_t dominant = static_cast<std::size_t>(std::distance(first, std::max_element(first, last)));
        assert(dominant == 8);
    }

    {
        temporal::AudioWindowCursor cursor(tone, 128, 64);
        std::size_t windows = 0;
        while (cursor.has_next()) {
            const auto window = cursor.next();
            assert(window.frame_count() == 128);
            ++windows;
        }
        assert(windows == 7);
        cursor.reset();
        assert(cursor.has_next());
    }

    temporal::TemporalTransformerConfig temporal_config;
    temporal_config.input_dim = 4;
    temporal_config.model_dim = 8;
    temporal_config.num_heads = 2;
    temporal_config.num_layers = 2;
    temporal_config.ffn_dim = 16;
    temporal_config.output_dim = 6;

    {
        Tensor sequence({4, 4}, {
            1.0F, 0.0F, 0.0F, 0.0F,
            0.0F, 1.0F, 0.0F, 0.0F,
            0.0F, 0.0F, 1.0F, 0.0F,
            0.0F, 0.0F, 0.0F, 1.0F});
        Tensor reversed({4, 4});
        for (std::size_t row = 0; row < 4; ++row) {
            for (std::size_t col = 0; col < 4; ++col) reversed.data()[row * 4 + col] = sequence.data()[(3 - row) * 4 + col];
        }
        Random a_rng(77);
        Random b_rng(77);
        temporal::TemporalTransformerEncoder a(temporal_config, a_rng);
        temporal::TemporalTransformerEncoder b(temporal_config, b_rng);
        const Tensor same_a = a.encode_pooled(sequence);
        const Tensor same_b = b.encode_pooled(sequence);
        assert(max_abs_diff(same_a, same_b) < 1.0e-6F);
        const Tensor reversed_out = a.encode_pooled(reversed);
        assert(max_abs_diff(same_a, reversed_out) > 1.0e-5F);
    }

    {
        Random codec_rng(101);
        audio::AudioCodecConfig codec_config{66, 4};
        audio::AudioLatentCodec codec(codec_config, codec_rng);
        temporal::AudioTemporalConfig config;
        config.stft = {64, 32};
        config.frames_per_patch = 2;
        config.temporal = temporal_config;
        config.temporal.input_dim = 4;
        Random time_rng(202);
        temporal::AudioTemporalEncoder encoder(config, codec, time_rng);
        const Tensor latents = encoder.spectral_latents(tone);
        assert(latents.rank() == 2 && latents.shape()[1] == 4 && latents.shape()[0] == 8);
        const Tensor pooled = encoder.encode_pooled(tone);
        assert(pooled.shape() == std::vector<std::size_t>({6}));
    }

    {
        vision::VisionConfig vision_config;
        vision_config.patch_size = 2;
        vision_config.model_dim = 8;
        vision_config.num_heads = 2;
        vision_config.num_layers = 1;
        vision_config.ffn_hidden_dim = 16;
        vision_config.embedding_dim = 6;
        Random vision_rng(303);
        vision::VisionEncoder vision_encoder(vision_config, vision_rng);

        temporal::VideoTemporalConfig video_config;
        video_config.temporal.input_dim = 6;
        video_config.temporal.model_dim = 8;
        video_config.temporal.num_heads = 2;
        video_config.temporal.num_layers = 2;
        video_config.temporal.ffn_dim = 16;
        video_config.temporal.output_dim = 5;
        Random temporal_rng(404);
        temporal::VideoTemporalEncoder video_encoder(video_config, vision_encoder, temporal_rng);

        temporal::VideoFrameSequence forward(24.0F);
        forward.add_frame(solid(255, 0, 0));
        forward.add_frame(solid(0, 255, 0));
        forward.add_frame(solid(0, 0, 255));
        temporal::VideoFrameSequence reverse(24.0F);
        reverse.add_frame(solid(0, 0, 255));
        reverse.add_frame(solid(0, 255, 0));
        reverse.add_frame(solid(255, 0, 0));

        const Tensor forward_tokens = video_encoder.encode_tokens(forward);
        const Tensor reverse_tokens = video_encoder.encode_tokens(reverse);
        assert(forward_tokens.shape() == std::vector<std::size_t>({3, 5}));
        assert(max_abs_diff(forward_tokens, reverse_tokens) > 1.0e-5F);
    }

    std::cout << "spiral_temporal_tests: PASS\n";
    return 0;
}
