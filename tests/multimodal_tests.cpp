#include "spiral/multimodal.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <string>

namespace {

spiral::vision::RgbImage make_quadrant_image() {
    spiral::vision::RgbImage image(4, 4);
    for (std::size_t y = 0; y < 4; ++y) {
        for (std::size_t x = 0; x < 4; ++x) {
            const bool top = y < 2;
            const bool left = x < 2;
            if (top && left) {
                image.at(x, y, 0) = 255;
            } else if (top && !left) {
                image.at(x, y, 1) = 255;
            } else if (!top && left) {
                image.at(x, y, 2) = 255;
            } else {
                image.at(x, y, 0) = 255;
                image.at(x, y, 1) = 255;
            }
        }
    }
    return image;
}

spiral::vision::RgbImage make_solid(std::uint8_t r, std::uint8_t g, std::uint8_t b) {
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

float channel_mean(const spiral::vision::RgbImage& image, std::size_t channel) {
    double sum = 0.0;
    for (std::size_t y = 0; y < image.height(); ++y) {
        for (std::size_t x = 0; x < image.width(); ++x) sum += image.at(x, y, channel);
    }
    return static_cast<float>(sum / static_cast<double>(image.width() * image.height()));
}

} // namespace

int main() {
    using namespace spiral;
    using namespace spiral::multimodal;

    const Tensor text = prompt_features("red signal", 24);
    assert(text.shape() == std::vector<std::size_t>({24}));
    double norm_sq = 0.0;
    for (const float value : text.data()) norm_sq += static_cast<double>(value) * value;
    assert(std::abs(norm_sq - 1.0) < 1.0e-5);
    assert(prompt_features("red signal", 24).data() == text.data());
    assert(prompt_features("blue signal", 24).data() != text.data());
    assert(cosine_similarity(text, text) > 0.9999F);

    Random auto_rng(0xA901ULL);
    AutoencoderConfig auto_config;
    auto_config.patch_size = 2;
    auto_config.latent_dim = 6;
    ImageAutoencoder autoencoder(auto_config, auto_rng);

    ImageTrainerConfig trainer_config;
    trainer_config.optimizer.learning_rate = 0.03F;
    trainer_config.optimizer.weight_decay = 0.0F;
    trainer_config.max_grad_norm = 10.0F;
    AutoencoderTrainer auto_trainer(autoencoder, trainer_config);

    const auto training_image = make_quadrant_image();
    const float initial_reconstruction = auto_trainer.evaluate(training_image);
    for (int step = 0; step < 350; ++step) (void)auto_trainer.train_step(training_image);
    const float final_reconstruction = auto_trainer.evaluate(training_image);
    assert(final_reconstruction < initial_reconstruction * 0.15F);
    assert(final_reconstruction < 0.01F);

    const auto red = make_solid(255, 0, 0);
    const auto blue = make_solid(0, 0, 255);
    assert(auto_trainer.evaluate(red) < 0.02F);
    assert(auto_trainer.evaluate(blue) < 0.02F);

    Random generator_rng(0xB902ULL);
    PromptGeneratorConfig generator_config;
    generator_config.text_feature_dim = 24;
    generator_config.latent_dim = auto_config.latent_dim;
    PromptLatentGenerator prompt_model(generator_config, generator_rng);
    PromptGeneratorTrainer prompt_trainer(prompt_model, autoencoder, trainer_config);

    const float initial_prompt_loss = 0.5F * (
        prompt_trainer.evaluate("red signal", red) + prompt_trainer.evaluate("blue signal", blue));
    for (int step = 0; step < 400; ++step) {
        (void)prompt_trainer.train_step("red signal", red);
        (void)prompt_trainer.train_step("blue signal", blue);
    }
    const float final_prompt_loss = 0.5F * (
        prompt_trainer.evaluate("red signal", red) + prompt_trainer.evaluate("blue signal", blue));
    assert(final_prompt_loss < initial_prompt_loss * 0.15F);
    assert(final_prompt_loss < 0.01F);

    SpiralImageGenerator image_generator(autoencoder, prompt_model);
    const auto generated_red = image_generator.generate("red signal", 2, 2);
    const auto generated_blue = image_generator.generate("blue signal", 2, 2);
    assert(generated_red.width() == 4 && generated_red.height() == 4);
    assert(generated_blue.width() == 4 && generated_blue.height() == 4);
    assert(channel_mean(generated_red, 0) > channel_mean(generated_red, 2) + 80.0F);
    assert(channel_mean(generated_blue, 2) > channel_mean(generated_blue, 0) + 80.0F);

    const auto bundle_path = std::filesystem::temp_directory_path() / "spiral_l9_image_bundle.bin";
    save_image_bundle(autoencoder, prompt_model, bundle_path.string());

    Random clone_auto_rng(1234);
    Random clone_prompt_rng(5678);
    ImageAutoencoder loaded_autoencoder(auto_config, clone_auto_rng);
    PromptLatentGenerator loaded_prompt(generator_config, clone_prompt_rng);
    load_image_bundle(loaded_autoencoder, loaded_prompt, bundle_path.string());
    SpiralImageGenerator loaded_generator(loaded_autoencoder, loaded_prompt);
    const auto reloaded_red = loaded_generator.generate("red signal", 2, 2);
    assert(reloaded_red.pixels() == generated_red.pixels());
    std::filesystem::remove(bundle_path);

    std::cout << "Spiral L9 multimodal tests passed\n";
    std::cout << "reconstruction loss: " << initial_reconstruction << " -> " << final_reconstruction << '\n';
    std::cout << "prompt latent loss: " << initial_prompt_loss << " -> " << final_prompt_loss << '\n';
    return 0;
}
