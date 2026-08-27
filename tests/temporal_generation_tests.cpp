#include "spiral/temporal_generation.hpp"

#include <cassert>
#include <cmath>
#include <filesystem>
#include <iostream>

namespace {

bool close(float a, float b, float tolerance = 1.0e-5F) {
    return std::fabs(a - b) <= tolerance;
}

spiral::vision::RgbImage solid_image(std::uint8_t r, std::uint8_t g, std::uint8_t b) {
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
    using namespace spiral::temporal_generation;

    CausalTemporalConfig config;
    config.input_dim = 2;
    config.model_dim = 8;
    config.num_heads = 2;
    config.num_layers = 1;
    config.ffn_dim = 16;
    config.output_dim = 2;
    config.max_context = 6;

    Random causal_rng(101);
    CausalTemporalPredictor causal_model(config, causal_rng);
    Tensor prefix_a({4, 2}, {0.0F, 0.0F, 0.2F, -0.1F, 0.4F, -0.2F, 0.6F, -0.3F});
    Tensor prefix_b({4, 2}, {0.0F, 0.0F, 0.2F, -0.1F, 9.0F, 9.0F, -9.0F, 7.0F});
    const Tensor pred_a = causal_model.predict_all(prefix_a);
    const Tensor pred_b = causal_model.predict_all(prefix_b);
    for (std::size_t i = 0; i < 4; ++i) assert(close(pred_a.data()[i], pred_b.data()[i], 1.0e-6F));

    Tensor sequence({8, 2});
    for (std::size_t row = 0; row < 8; ++row) {
        sequence.data()[row * 2] = 0.15F * static_cast<float>(row);
        sequence.data()[row * 2 + 1] = -0.08F * static_cast<float>(row);
    }

    Random train_rng(2026);
    CausalTemporalPredictor trained(config, train_rng);
    TemporalTrainerConfig trainer_config;
    trainer_config.optimizer.learning_rate = 0.01F;
    trainer_config.optimizer.weight_decay = 0.0F;
    trainer_config.max_grad_norm = 5.0F;
    TemporalNextLatentTrainer trainer(trained, trainer_config);
    const float initial_loss = trainer.evaluate(sequence);
    for (int step = 0; step < 500; ++step) (void)trainer.train_step(sequence);
    const float final_loss = trainer.evaluate(sequence);
    assert(std::isfinite(initial_loss) && std::isfinite(final_loss));
    assert(final_loss < initial_loss * 0.25F);

    const Tensor next = trained.predict_next(sequence);
    assert(next.shape() == std::vector<std::size_t>({2}));
    assert(std::isfinite(next.data()[0]) && std::isfinite(next.data()[1]));
    const Tensor generated = trained.generate(sequence, 4);
    assert(generated.shape() == std::vector<std::size_t>({4, 2}));
    for (const float value : generated.data()) assert(std::isfinite(value));

    const auto checkpoint = std::filesystem::temp_directory_path() / "spiral_l15_temporal.bin";
    save_temporal_predictor(trained, checkpoint.string());
    Random restore_rng(999);
    CausalTemporalPredictor restored(config, restore_rng);
    load_temporal_predictor(restored, checkpoint.string());
    const Tensor restored_prediction = restored.predict_all(sequence);
    const Tensor trained_prediction = trained.predict_all(sequence);
    assert(restored_prediction.shape() == trained_prediction.shape());
    for (std::size_t i = 0; i < trained_prediction.numel(); ++i) assert(close(restored_prediction.data()[i], trained_prediction.data()[i], 0.0F));
    std::filesystem::remove(checkpoint);

    audio::StftConfig stft{16, 8};
    Tensor magnitude({3, 9});
    magnitude.data()[2] = 4.0F;
    magnitude.data()[9 + 2] = 3.0F;
    magnitude.data()[18 + 2] = 2.0F;
    const audio::AudioBuffer reconstructed = magnitude_to_audio_zero_phase(magnitude, 8000, stft);
    assert(reconstructed.channels() == 1);
    assert(reconstructed.frame_count() == 32);
    float energy = 0.0F;
    for (const float sample : reconstructed.samples()) {
        assert(std::isfinite(sample));
        energy += sample * sample;
    }
    assert(energy > 0.0F);

    Random codec_rng(303);
    audio::AudioCodecConfig codec_config;
    codec_config.patch_dim = 9;
    codec_config.latent_dim = 2;
    audio::AudioLatentCodec codec(codec_config, codec_rng);
    AudioGenerationConfig audio_config;
    audio_config.stft = stft;
    audio_config.frames_per_patch = 1;
    AudioLatentGenerator audio_generator(audio_config, codec, trained);
    std::vector<float> samples(64);
    for (std::size_t i = 0; i < samples.size(); ++i) {
        samples[i] = 0.4F * std::sin(2.0F * 3.14159265358979323846F * 1000.0F * static_cast<float>(i) / 8000.0F);
    }
    audio::AudioBuffer seed_audio(8000, 1, samples);
    const Tensor seed_latents = audio_generator.encode_latents(seed_audio);
    assert(seed_latents.rank() == 2 && seed_latents.shape()[1] == 2);
    const Tensor future_audio_latents = audio_generator.generate_latents(seed_audio, 3);
    assert(future_audio_latents.shape() == std::vector<std::size_t>({3, 2}));
    const audio::AudioBuffer synthesized = audio_generator.synthesize(future_audio_latents, 8000);
    assert(synthesized.frame_count() > 0 && synthesized.channels() == 1);
    for (const float sample : synthesized.samples()) assert(std::isfinite(sample));

    vision::VisionConfig vision_config;
    vision_config.patch_size = 2;
    vision_config.model_dim = 4;
    vision_config.num_heads = 2;
    vision_config.num_layers = 1;
    vision_config.ffn_hidden_dim = 8;
    vision_config.embedding_dim = 2;
    Random vision_rng(404);
    vision::VisionEncoder vision_encoder(vision_config, vision_rng);
    VideoEmbeddingGenerator video_generator(vision_encoder, trained);
    temporal::VideoFrameSequence video(12.0F);
    video.add_frame(solid_image(255, 0, 0));
    video.add_frame(solid_image(0, 255, 0));
    video.add_frame(solid_image(0, 0, 255));
    const Tensor frame_embeddings = video_generator.frame_embeddings(video);
    assert(frame_embeddings.shape() == std::vector<std::size_t>({3, 2}));
    const Tensor next_frame_embedding = video_generator.predict_next(video);
    assert(next_frame_embedding.shape() == std::vector<std::size_t>({2}));
    const Tensor future_frames = video_generator.generate(video, 2);
    assert(future_frames.shape() == std::vector<std::size_t>({2, 2}));
    for (const float value : future_frames.data()) assert(std::isfinite(value));

    std::cout << "spiral_temporal_generation_tests: PASS loss " << initial_loss << " -> " << final_loss << '\n';
    return 0;
}
