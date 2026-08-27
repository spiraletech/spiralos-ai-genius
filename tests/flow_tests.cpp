#include "spiral/flow.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

bool close(float lhs, float rhs, float tolerance = 1.0e-5F) {
    return std::fabs(lhs - rhs) <= tolerance;
}

spiral::vision::RgbImage solid_image(std::uint8_t r, std::uint8_t g, std::uint8_t b) {
    spiral::vision::RgbImage image(4, 4);
    for (std::size_t y = 0; y < image.height(); ++y) {
        for (std::size_t x = 0; x < image.width(); ++x) {
            image.at(x, y, 0) = r;
            image.at(x, y, 1) = g;
            image.at(x, y, 2) = b;
        }
    }
    return image;
}

float average_channel(const spiral::vision::RgbImage& image, std::size_t channel) {
    double sum = 0.0;
    for (std::size_t y = 0; y < image.height(); ++y) {
        for (std::size_t x = 0; x < image.width(); ++x) sum += image.at(x, y, channel);
    }
    return static_cast<float>(sum / static_cast<double>(image.width() * image.height()));
}

bool same_pixels(const spiral::vision::RgbImage& lhs, const spiral::vision::RgbImage& rhs) {
    return lhs.width() == rhs.width() && lhs.height() == rhs.height() && lhs.pixels() == rhs.pixels();
}

} // namespace

int main() {
    using namespace spiral;
    using namespace spiral::flow;
    using namespace spiral::multimodal;

    const Tensor clean({2}, std::vector<float>{1.0F, -1.0F});
    const Tensor noise({2}, std::vector<float>{0.25F, 0.75F});
    NoiseScheduler linear(NoiseScheduleKind::Linear);
    require(linear.add_noise(clean, noise, 0.0F).data() == clean.data(), "t=0 must preserve clean latent");
    require(linear.add_noise(clean, noise, 1.0F).data() == noise.data(), "t=1 must become pure noise");

    NoiseScheduler cosine(NoiseScheduleKind::Cosine);
    require(cosine.noise_level(0.5F) > 0.5F, "cosine schedule midpoint must differ from linear schedule");
    const auto times = cosine.sampling_times(4);
    require(times.size() == 5 && close(times.front(), 1.0F) && close(times.back(), 0.0F), "sampling times must span one to zero");
    for (std::size_t i = 1; i < times.size(); ++i) require(times[i] < times[i - 1], "sampling times must descend");

    const Tensor time_a = timestep_features(0.4F, 8);
    const Tensor time_b = timestep_features(0.4F, 8);
    require(time_a.shape() == std::vector<std::size_t>{8} && time_a.data() == time_b.data(), "timestep features must be deterministic");

    const Tensor unconditional({2}, std::vector<float>{0.0F, 1.0F});
    const Tensor conditional({2}, std::vector<float>{1.0F, 3.0F});
    const Tensor guided = guided_prediction(unconditional, conditional, 2.0F);
    require(close(guided.data()[0], 2.0F) && close(guided.data()[1], 5.0F), "classifier-free guidance equation is incorrect");

    const auto red = solid_image(255, 16, 16);
    const auto blue = solid_image(16, 16, 255);

    Random auto_rng(1001);
    ImageAutoencoder autoencoder({2, 4}, auto_rng);
    ImageTrainerConfig image_train_config;
    image_train_config.optimizer.learning_rate = 0.02F;
    image_train_config.optimizer.weight_decay = 0.0F;
    AutoencoderTrainer auto_trainer(autoencoder, image_train_config);

    const float initial_reconstruction = 0.5F * (auto_trainer.evaluate(red) + auto_trainer.evaluate(blue));
    for (std::size_t step = 0; step < 300; ++step) {
        (void)auto_trainer.train_step((step % 2 == 0) ? red : blue);
    }
    const float trained_reconstruction = 0.5F * (auto_trainer.evaluate(red) + auto_trainer.evaluate(blue));
    require(trained_reconstruction < initial_reconstruction * 0.20F, "autoencoder must learn the tiny visual world before denoising");

    Random denoiser_rng(2002);
    DenoiserConfig denoiser_config;
    denoiser_config.text_feature_dim = 16;
    denoiser_config.latent_dim = 4;
    denoiser_config.time_feature_dim = 8;
    denoiser_config.hidden_dim = 32;
    LatentDenoiser denoiser(denoiser_config, denoiser_rng);

    DenoiserTrainerConfig trainer_config;
    trainer_config.optimizer.learning_rate = 0.01F;
    trainer_config.optimizer.weight_decay = 0.0F;
    DenoiserTrainer trainer(denoiser, autoencoder, cosine, trainer_config);

    const Tensor red_latent = autoencoder.encode(red);
    Random fixed_noise_rng(3030);
    const Tensor fixed_noise = gaussian_noise(red_latent.shape(), fixed_noise_rng);
    const float initial_denoise_loss = trainer.evaluate("red signal", red, 0.8F, fixed_noise);

    Random train_noise_rng(4040);
    for (std::size_t step = 0; step < 1200; ++step) {
        const bool red_step = (step % 2 == 0);
        const float time = 0.20F + 0.80F * static_cast<float>((step % 5) + 1) / 5.0F;
        const Tensor target_latent = autoencoder.encode(red_step ? red : blue);
        const Tensor step_noise = gaussian_noise(target_latent.shape(), train_noise_rng);
        (void)trainer.train_step(red_step ? "red signal" : "blue signal", red_step ? red : blue, time, step_noise);
    }

    const float trained_denoise_loss = trainer.evaluate("red signal", red, 0.8F, fixed_noise);
    require(trained_denoise_loss < initial_denoise_loss * 0.35F, "denoiser training must substantially reduce fixed-noise loss");

    IterativeImageGenerator generator(autoencoder, denoiser, cosine);
    SamplingConfig sampling;
    sampling.steps = 10;
    sampling.guidance_scale = 1.5F;
    sampling.seed = 5151;

    const auto generated_red_a = generator.generate("red signal", 2, 2, sampling);
    const auto generated_red_b = generator.generate("red signal", 2, 2, sampling);
    require(same_pixels(generated_red_a, generated_red_b), "seeded iterative generation must be deterministic");

    const auto generated_blue = generator.generate("blue signal", 2, 2, sampling);
    require(!same_pixels(generated_red_a, generated_blue), "different learned prompts must not collapse to identical images");
    require(average_channel(generated_red_a, 0) > average_channel(generated_red_a, 2) + 40.0F,
            "red prompt should produce a red-dominant learned output");
    require(average_channel(generated_blue, 2) > average_channel(generated_blue, 0) + 40.0F,
            "blue prompt should produce a blue-dominant learned output");

    SamplingConfig no_guidance = sampling;
    no_guidance.guidance_scale = 0.0F;
    const auto unconditioned = generator.generate("red signal", 2, 2, no_guidance);
    require(!same_pixels(unconditioned, generated_red_a), "classifier-free guidance must affect iterative output");

    const auto reconstructed_red = autoencoder.reconstruct(red);
    const auto zero_strength = generator.image_to_image("blue signal", red, 0.0F, sampling);
    require(same_pixels(reconstructed_red, zero_strength), "img2img strength zero must preserve reconstructed source");

    const Tensor preserve_all({4}, std::vector<float>{0.0F, 0.0F, 0.0F, 0.0F});
    const auto preserved = generator.inpaint("blue signal", red, preserve_all, 1.0F, sampling);
    require(same_pixels(reconstructed_red, preserved), "zero inpaint mask must preserve every source latent patch");

    const Tensor edit_all({4}, std::vector<float>{1.0F, 1.0F, 1.0F, 1.0F});
    const auto edited = generator.inpaint("blue signal", red, edit_all, 1.0F, sampling);
    require(!same_pixels(reconstructed_red, edited), "full inpaint mask must permit prompt-driven editing");

    std::cout << "L10 iterative generative engine passed"
              << " | reconstruction " << initial_reconstruction << " -> " << trained_reconstruction
              << " | denoise " << initial_denoise_loss << " -> " << trained_denoise_loss << '\n';
    return 0;
}
