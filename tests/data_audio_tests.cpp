#include "spiral/audio.hpp"
#include "spiral/data.hpp"
#include "spiral/precision.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace {

bool near(float a, float b, float tolerance) {
    return std::fabs(a - b) <= tolerance;
}

spiral::vision::RgbImage solid_image(std::uint8_t r, std::uint8_t g, std::uint8_t b) {
    std::vector<std::uint8_t> pixels(2U * 2U * 3U);
    for (std::size_t i = 0; i < 4; ++i) {
        pixels[i * 3U] = r;
        pixels[i * 3U + 1U] = g;
        pixels[i * 3U + 2U] = b;
    }
    return spiral::vision::RgbImage(2, 2, std::move(pixels));
}

} // namespace

int main() {
    namespace fs = std::filesystem;
    using spiral::Tensor;

    // Native FP16/BF16 conversion and symmetric int8 weight quantization.
    for (const float value : {0.0F, 1.5F, -2.25F, 12.75F}) {
        const auto half = spiral::precision::float_to_float16(value);
        assert(near(spiral::precision::float16_to_float(half), value, 0.01F));
        const auto bf16 = spiral::precision::float_to_bfloat16(value);
        assert(near(spiral::precision::bfloat16_to_float(bf16), value, 0.06F));
    }
    const Tensor weights({2, 3}, {-2.0F, -0.5F, 0.0F, 0.25F, 1.0F, 2.0F});
    const auto quantized = spiral::precision::quantize_symmetric_int8(weights);
    const Tensor restored = quantized.dequantize();
    assert(restored.shape() == weights.shape());
    float max_quant_error = 0.0F;
    for (std::size_t i = 0; i < weights.numel(); ++i) {
        max_quant_error = std::max(max_quant_error, std::fabs(weights.data()[i] - restored.data()[i]));
    }
    assert(max_quant_error <= quantized.scale + 1.0e-5F);

    const fs::path temp = fs::temp_directory_path() / "spiral_l13_regression";
    fs::create_directories(temp);

    // Lazy sharded image/prompt data with bounded cache.
    const fs::path shard_a = temp / "a.spshard";
    const fs::path shard_b = temp / "b.spshard";
    spiral::data::write_image_prompt_shard(shard_a.string(), {
        {"red signal", solid_image(255, 0, 0)},
        {"green signal", solid_image(0, 255, 0)}});
    spiral::data::write_image_prompt_shard(shard_b.string(), {
        {"blue signal", solid_image(0, 0, 255)},
        {"white signal", solid_image(255, 255, 255)}});

    spiral::data::ShardedImagePromptDataset dataset(2);
    dataset.add_shard(shard_a.string());
    dataset.add_shard(shard_b.string());
    assert(dataset.shard_count() == 2);
    assert(dataset.size() == 4);
    assert(dataset.load(0).prompt == "red signal");
    const auto blue = dataset.load(2);
    assert(blue.prompt == "blue signal");
    assert(blue.image.at(0, 0, 2) == 255);
    dataset.prefetch(1, 3);
    assert(dataset.resident_count() <= 2);
    assert(dataset.resident_count() > 0);
    dataset.clear_cache();
    assert(dataset.resident_count() == 0);

    // PCM16 WAV round-trip and spectral peak detection.
    constexpr std::uint32_t sample_rate = 8000;
    constexpr float frequency = 1000.0F;
    std::vector<float> sine(512);
    for (std::size_t i = 0; i < sine.size(); ++i) {
        sine[i] = 0.65F * std::sin(2.0F * 3.14159265358979323846F * frequency *
                                  static_cast<float>(i) / static_cast<float>(sample_rate));
    }
    const spiral::audio::AudioBuffer source(sample_rate, 1, sine);
    const fs::path wav = temp / "tone.wav";
    source.save_wav_pcm16(wav.string());
    const auto loaded = spiral::audio::AudioBuffer::load_wav(wav.string());
    assert(loaded.sample_rate() == sample_rate);
    assert(loaded.channels() == 1);
    assert(loaded.frame_count() == source.frame_count());
    assert(near(loaded.sample(17, 0), source.sample(17, 0), 5.0e-5F));

    const Tensor spectrum = spiral::audio::stft_magnitude(loaded, {64, 32});
    assert(spectrum.rank() == 2);
    assert(spectrum.shape()[1] == 33);
    const auto first_begin = spectrum.data().begin();
    const auto first_end = first_begin + static_cast<std::ptrdiff_t>(spectrum.shape()[1]);
    const std::size_t peak_bin = static_cast<std::size_t>(std::distance(first_begin, std::max_element(first_begin, first_end)));
    assert(peak_bin == 8);
    const Tensor patches = spiral::audio::spectral_patches(spectrum, 2);
    assert(patches.rank() == 2);
    assert(patches.shape()[1] == 66);

    // Trainable native audio latent codec must reduce reconstruction loss.
    Tensor training({4, 4}, {
        1.0F, 0.0F, 0.2F, 0.0F,
        0.0F, 1.0F, 0.0F, 0.2F,
        0.8F, 0.1F, 0.2F, 0.0F,
        0.1F, 0.8F, 0.0F, 0.2F});
    spiral::Random codec_rng(0xA0D10ULL);
    spiral::audio::AudioLatentCodec codec({4, 2}, codec_rng);
    spiral::audio::AudioCodecTrainer trainer(codec, {{0.02F, 0.9F, 0.999F, 1.0e-8F, 0.0F}, 5.0F});
    const float initial_loss = trainer.evaluate(training);
    for (std::size_t step = 0; step < 250; ++step) (void)trainer.train_step(training);
    const float final_loss = trainer.evaluate(training);
    assert(std::isfinite(initial_loss) && std::isfinite(final_loss));
    assert(final_loss < initial_loss * 0.45F);
    const Tensor latents = codec.encode(training);
    assert(latents.shape() == std::vector<std::size_t>({4, 2}));

    fs::remove_all(temp);
    std::cout << "spiral_data_audio_tests: PASS initial=" << initial_loss << " final=" << final_loss << '\n';
    return 0;
}
