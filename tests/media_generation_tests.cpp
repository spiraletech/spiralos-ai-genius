#include "spiral/media_generation.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numbers>

using namespace spiral;

namespace {

audio::AudioBuffer tone(float hz, std::uint32_t sample_rate, std::size_t frames) {
    std::vector<float> samples(frames);
    for (std::size_t i = 0; i < frames; ++i) {
        samples[i] = 0.6F * std::sin(2.0F * std::numbers::pi_v<float> * hz * static_cast<float>(i) /
                                   static_cast<float>(sample_rate));
    }
    return audio::AudioBuffer(sample_rate, 1, std::move(samples));
}

vision::RgbImage solid(std::uint8_t r, std::uint8_t g, std::uint8_t b) {
    vision::RgbImage image(4, 4);
    for (std::size_t y = 0; y < 4; ++y) {
        for (std::size_t x = 0; x < 4; ++x) {
            image.at(x, y, 0) = r;
            image.at(x, y, 1) = g;
            image.at(x, y, 2) = b;
        }
    }
    return image;
}

bool tensors_close(const Tensor& lhs, const Tensor& rhs, float tolerance = 1.0e-5F) {
    if (lhs.shape() != rhs.shape()) return false;
    for (std::size_t i = 0; i < lhs.numel(); ++i) {
        if (std::abs(lhs.data()[i] - rhs.data()[i]) > tolerance) return false;
    }
    return true;
}

float peak_abs(const audio::AudioBuffer& audio) {
    float peak = 0.0F;
    for (float value : audio.samples()) peak = std::max(peak, std::abs(value));
    return peak;
}

} // namespace

int main() {
    using namespace spiral::media_generation;

    constexpr std::uint32_t sample_rate = 8000;
    const ComplexStftConfig stft{32, 16};
    const auto source = tone(1000.0F, sample_rate, 256);
    const Tensor complex = complex_stft(source, stft);
    assert(complex.rank() == 3);
    assert(complex.shape()[1] == 17);
    assert(complex.shape()[2] == 2);

    const auto rebuilt = inverse_complex_stft(complex, sample_rate, stft);
    assert(rebuilt.sample_rate() == sample_rate);
    assert(rebuilt.channels() == 1);
    assert(rebuilt.frame_count() >= source.frame_count());
    assert(peak_abs(rebuilt) > 0.2F);

    const Tensor patches = complex_spectral_patches(complex, 2);
    const Tensor patch_roundtrip = complex_patches_to_spectrum(patches, complex.shape()[0], complex.shape()[1], 2);
    assert(tensors_close(complex, patch_roundtrip, 1.0e-6F));

    Random audio_rng(1601);
    ComplexAudioCodec codec({patches.shape()[1], 8}, audio_rng);
    ComplexAudioCodecTrainer codec_trainer(codec, {{0.01F, 0.9F, 0.999F, 1.0e-8F, 0.0F}, 5.0F});
    const float audio_initial = codec_trainer.evaluate(patches);
    for (int i = 0; i < 400; ++i) (void)codec_trainer.train_step(patches);
    const float audio_final = codec_trainer.evaluate(patches);
    assert(std::isfinite(audio_final));
    assert(audio_final < audio_initial * 0.55F);

    vision::VisionConfig vision_config;
    vision_config.patch_size = 2;
    vision_config.model_dim = 8;
    vision_config.num_heads = 2;
    vision_config.num_layers = 1;
    vision_config.ffn_hidden_dim = 16;
    vision_config.embedding_dim = 8;
    Random vision_rng(1602);
    vision::VisionEncoder vision_encoder(vision_config, vision_rng);

    Random frame_rng(1603);
    FrameEmbeddingDecoder frame_decoder({8, 4, 4, 16}, frame_rng);
    FrameDecoderTrainer frame_trainer(frame_decoder, vision_encoder, {{0.02F, 0.9F, 0.999F, 1.0e-8F, 0.0F}, 5.0F});
    const auto red = solid(230, 20, 20);
    const float frame_initial = frame_trainer.evaluate(red);
    for (int i = 0; i < 500; ++i) (void)frame_trainer.train_step(red);
    const float frame_final = frame_trainer.evaluate(red);
    assert(std::isfinite(frame_final));
    assert(frame_final < frame_initial * 0.20F);
    const auto decoded_red = frame_decoder.decode(vision_encoder.encode_pooled(red));
    assert(decoded_red.width() == 4 && decoded_red.height() == 4);
    assert(decoded_red.at(0, 0, 0) > decoded_red.at(0, 0, 2));

    temporal_generation::CausalTemporalConfig causal;
    causal.input_dim = 8;
    causal.model_dim = 16;
    causal.num_heads = 2;
    causal.num_layers = 1;
    causal.ffn_dim = 24;
    causal.output_dim = 8;
    causal.max_context = 16;

    Random audio_predict_rng(1604);
    temporal_generation::CausalTemporalPredictor audio_predictor(causal, audio_predict_rng);
    AudioMediaGenerator audio_generator(stft, 2, codec, audio_predictor);
    const Tensor seed_latents = audio_generator.encode_latents(source);
    assert(seed_latents.shape()[1] == 8);
    const Tensor future_audio_latents = audio_generator.continue_latents(source, 2);
    assert(future_audio_latents.shape() == std::vector<std::size_t>({2, 8}));
    const auto future_audio = audio_generator.decode_latents(future_audio_latents, sample_rate, 4);
    assert(future_audio.frame_count() > 0);
    for (float sample : future_audio.samples()) assert(std::isfinite(sample));

    Random video_predict_rng(1605);
    temporal_generation::CausalTemporalPredictor video_predictor(causal, video_predict_rng);
    VideoMediaGenerator video_generator(vision_encoder, video_predictor, frame_decoder);
    temporal::VideoFrameSequence video(12.0F);
    video.add_frame(red);
    video.add_frame(solid(20, 20, 230));
    const auto future_frames = video_generator.continue_frames(video, 3);
    assert(future_frames.size() == 3);
    for (const auto& frame : future_frames) {
        assert(frame.width() == 4 && frame.height() == 4);
    }

    Tensor seed({2, 8});
    for (std::size_t i = 0; i < seed.numel(); ++i) seed.data()[i] = static_cast<float>(i) * 0.01F;
    const Tensor bias_a = apply_prompt_bias(seed, "red signal");
    const Tensor bias_b = apply_prompt_bias(seed, "blue signal");
    const Tensor bias_a_again = apply_prompt_bias(seed, "red signal");
    assert(tensors_close(bias_a, bias_a_again));
    assert(!tensors_close(bias_a, bias_b));

    std::cout << "L16 complex audio loss: " << audio_initial << " -> " << audio_final << '\n';
    std::cout << "L16 frame decoder loss: " << frame_initial << " -> " << frame_final << '\n';
    std::cout << "L16 generative media decoder tests passed\n";
    return 0;
}
