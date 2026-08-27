#include "spiral/latent_transformer.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <vector>

namespace {

spiral::vision::RgbImage make_quadrant_image() {
    spiral::vision::RgbImage image(4, 4);
    for (std::size_t y = 0; y < 4; ++y) {
        for (std::size_t x = 0; x < 4; ++x) {
            const bool top = y < 2;
            const bool left = x < 2;
            if (top && left) image.at(x, y, 0) = 255;
            else if (top && !left) image.at(x, y, 1) = 255;
            else if (!top && left) image.at(x, y, 2) = 255;
            else {
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

float tensor_difference(const spiral::Tensor& a, const spiral::Tensor& b) {
    assert(a.shape() == b.shape());
    float sum = 0.0F;
    for (std::size_t i = 0; i < a.numel(); ++i) sum += std::abs(a.data()[i] - b.data()[i]);
    return sum;
}

std::uint64_t pixel_difference(const spiral::vision::RgbImage& a, const spiral::vision::RgbImage& b) {
    assert(a.width() == b.width() && a.height() == b.height());
    std::uint64_t sum = 0;
    for (std::size_t i = 0; i < a.pixels().size(); ++i) {
        const int delta = static_cast<int>(a.pixels()[i]) - static_cast<int>(b.pixels()[i]);
        sum += static_cast<std::uint64_t>(delta < 0 ? -delta : delta);
    }
    return sum;
}

} // namespace

int main() {
    using namespace spiral;
    using namespace spiral::flow;
    using namespace spiral::latent;
    using namespace spiral::multimodal;

    const Tensor prompt_tokens = prompt_token_features("red signal", 4, 12);
    assert(prompt_tokens.shape() == std::vector<std::size_t>({4, 12}));
    assert(prompt_token_features("red signal", 4, 12).data() == prompt_tokens.data());
    assert(prompt_token_features("blue signal", 4, 12).data() != prompt_tokens.data());
    for (std::size_t row = 0; row < 4; ++row) {
        double norm_sq = 0.0;
        for (std::size_t d = 0; d < 12; ++d) {
            const float value = prompt_tokens.data()[row * 12 + d];
            norm_sq += static_cast<double>(value) * value;
        }
        assert(norm_sq < 1.0e-6 || std::abs(norm_sq - 1.0) < 1.0e-5);
    }

    AutoencoderConfig auto_config;
    auto_config.patch_size = 2;
    auto_config.latent_dim = 6;
    Random auto_rng(0x1101ULL);
    ImageAutoencoder autoencoder(auto_config, auto_rng);
    ImageTrainerConfig image_train_config;
    image_train_config.optimizer.learning_rate = 0.03F;
    image_train_config.optimizer.weight_decay = 0.0F;
    image_train_config.max_grad_norm = 10.0F;
    AutoencoderTrainer auto_trainer(autoencoder, image_train_config);
    const auto quadrant = make_quadrant_image();
    for (int step = 0; step < 350; ++step) (void)auto_trainer.train_step(quadrant);

    const auto red = make_solid(255, 0, 0);
    const auto blue = make_solid(0, 0, 255);
    assert(auto_trainer.evaluate(red) < 0.03F);
    assert(auto_trainer.evaluate(blue) < 0.03F);

    LatentTransformerConfig config;
    config.latent_dim = auto_config.latent_dim;
    config.model_dim = 12;
    config.num_heads = 3;
    config.num_layers = 1;
    config.ffn_dim = 24;
    config.text_feature_dim = 12;
    config.prompt_tokens = 4;
    config.time_feature_dim = 4;

    Random model_rng(0x1102ULL);
    LatentTransformerDenoiser model(config, model_rng);
    assert(!model.parameters().empty());

    Tensor base_latent({4, config.latent_dim});
    Tensor changed_latent = base_latent;
    for (std::size_t d = 0; d < config.latent_dim; ++d) changed_latent.data()[3 * config.latent_dim + d] = 1.0F + static_cast<float>(d) * 0.1F;
    const Tensor base_output = model.predict(base_latent, "red signal", 0.5F, 2, 2);
    const Tensor changed_output = model.predict(changed_latent, "red signal", 0.5F, 2, 2);
    float first_patch_difference = 0.0F;
    for (std::size_t d = 0; d < config.latent_dim; ++d) {
        first_patch_difference += std::abs(base_output.data()[d] - changed_output.data()[d]);
    }
    assert(first_patch_difference > 1.0e-6F);
    assert(tensor_difference(base_output, model.predict(base_latent, "blue signal", 0.5F, 2, 2)) > 1.0e-6F);

    ImagePromptDataset dataset;
    dataset.add("red signal", red);
    dataset.add("blue signal", blue);
    assert(dataset.size() == 2);
    assert(dataset.at(0).prompt == "red signal");

    LatentTransformerTrainerConfig trainer_config;
    trainer_config.optimizer.learning_rate = 0.004F;
    trainer_config.optimizer.weight_decay = 0.0F;
    trainer_config.max_grad_norm = 5.0F;
    trainer_config.training_time = 0.65F;
    trainer_config.noise_seed = 0x1103ULL;
    LatentTransformerTrainer trainer(model, autoencoder, NoiseScheduler{}, trainer_config);

    const float initial_loss = trainer.evaluate_dataset(dataset, 0x2201ULL);
    for (int epoch = 0; epoch < 260; ++epoch) (void)trainer.train_epoch(dataset);
    const float final_loss = trainer.evaluate_dataset(dataset, 0x2201ULL);
    assert(std::isfinite(initial_loss) && std::isfinite(final_loss));
    assert(final_loss < initial_loss * 0.35F);
    assert(final_loss < 0.08F);

    Random fixed_noise_rng(0x3301ULL);
    const Tensor red_target = autoencoder.encode(red);
    const Tensor fixed_noise = gaussian_noise(red_target.shape(), fixed_noise_rng);
    const Tensor noisy = NoiseScheduler{}.add_noise(red_target, fixed_noise, trainer_config.training_time);
    const Tensor learned_red = model.predict(noisy, "red signal", trainer_config.training_time, 2, 2);
    const Tensor learned_blue = model.predict(noisy, "blue signal", trainer_config.training_time, 2, 2);
    assert(tensor_difference(learned_red, learned_blue) > 0.05F);

    IterativeImageGenerator iterative(autoencoder, model);
    SamplingConfig sampling;
    sampling.steps = 6;
    sampling.guidance_scale = 1.5F;
    sampling.seed = 0x4401ULL;
    const auto generated_red = iterative.generate("red signal", 2, 2, sampling);
    const auto generated_red_again = iterative.generate("red signal", 2, 2, sampling);
    const auto generated_blue = iterative.generate("blue signal", 2, 2, sampling);
    assert(generated_red.pixels() == generated_red_again.pixels());
    assert(pixel_difference(generated_red, generated_blue) > 100);

    const auto checkpoint = std::filesystem::temp_directory_path() / "spiral_l11_latent_transformer.bin";
    save_latent_transformer(model, checkpoint.string());
    Random clone_rng(0xDEADBEEFULL);
    LatentTransformerDenoiser clone(config, clone_rng);
    load_latent_transformer(clone, checkpoint.string());
    const Tensor cloned_prediction = clone.predict(noisy, "red signal", trainer_config.training_time, 2, 2);
    assert(cloned_prediction.data() == learned_red.data());
    IterativeImageGenerator clone_generator(autoencoder, clone);
    assert(clone_generator.generate("red signal", 2, 2, sampling).pixels() == generated_red.pixels());
    std::filesystem::remove(checkpoint);

    std::cout << "Spiral L11 latent transformer tests passed\n";
    std::cout << "batch denoising loss: " << initial_loss << " -> " << final_loss << '\n';
    std::cout << "cross-prompt latent delta: " << tensor_difference(learned_red, learned_blue) << '\n';
    return 0;
}
