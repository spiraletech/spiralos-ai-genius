#pragma once

#include "spiral/nn.hpp"
#include "spiral/random.hpp"
#include "spiral/tensor.hpp"
#include "spiral/train.hpp"
#include "spiral/vision.hpp"

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace spiral::multimodal {

[[nodiscard]] Tensor prompt_features(std::string_view prompt, std::size_t feature_dim);
[[nodiscard]] float mean_squared_error(const Tensor& prediction, const Tensor& target);
[[nodiscard]] float cosine_similarity(const Tensor& lhs, const Tensor& rhs);

struct AutoencoderConfig {
    std::size_t patch_size = 2;
    std::size_t latent_dim = 8;
};

class ImageAutoencoder final {
public:
    ImageAutoencoder(AutoencoderConfig config, Random& rng);

    [[nodiscard]] Tensor encode(const vision::RgbImage& image) const;
    [[nodiscard]] vision::RgbImage decode(const Tensor& latent, std::size_t grid_height, std::size_t grid_width) const;
    [[nodiscard]] vision::RgbImage reconstruct(const vision::RgbImage& image) const;
    [[nodiscard]] std::vector<nn::Parameter*> parameters();
    [[nodiscard]] std::vector<const nn::Parameter*> parameters() const;
    [[nodiscard]] const AutoencoderConfig& config() const noexcept { return config_; }
    [[nodiscard]] nn::Linear& encoder() noexcept { return encoder_; }
    [[nodiscard]] const nn::Linear& encoder() const noexcept { return encoder_; }
    [[nodiscard]] nn::Linear& decoder() noexcept { return decoder_; }
    [[nodiscard]] const nn::Linear& decoder() const noexcept { return decoder_; }

private:
    AutoencoderConfig config_;
    nn::Linear encoder_;
    nn::Linear decoder_;
};

struct PromptGeneratorConfig {
    std::size_t text_feature_dim = 32;
    std::size_t latent_dim = 8;
};

class PromptLatentGenerator final {
public:
    PromptLatentGenerator(PromptGeneratorConfig config, Random& rng);

    [[nodiscard]] Tensor predict(std::string_view prompt, std::size_t grid_height, std::size_t grid_width) const;
    [[nodiscard]] std::vector<nn::Parameter*> parameters();
    [[nodiscard]] std::vector<const nn::Parameter*> parameters() const;
    [[nodiscard]] const PromptGeneratorConfig& config() const noexcept { return config_; }
    [[nodiscard]] nn::Linear& projection() noexcept { return projection_; }
    [[nodiscard]] const nn::Linear& projection() const noexcept { return projection_; }

private:
    [[nodiscard]] Tensor conditioning_matrix(std::string_view prompt, std::size_t grid_height, std::size_t grid_width) const;

    PromptGeneratorConfig config_;
    nn::Linear projection_;
};

struct ImageTrainerConfig {
    train::AdamWConfig optimizer{0.02F, 0.9F, 0.999F, 1.0e-8F, 0.0F};
    float max_grad_norm = 5.0F;
};

class AutoencoderTrainer final {
public:
    AutoencoderTrainer(ImageAutoencoder& model, ImageTrainerConfig config = {});
    [[nodiscard]] float evaluate(const vision::RgbImage& image) const;
    float train_step(const vision::RgbImage& image);

private:
    ImageAutoencoder& model_;
    ImageTrainerConfig config_;
    train::AdamW optimizer_;
};

class PromptGeneratorTrainer final {
public:
    PromptGeneratorTrainer(
        PromptLatentGenerator& generator,
        const ImageAutoencoder& target_encoder,
        ImageTrainerConfig config = {});

    [[nodiscard]] float evaluate(std::string_view prompt, const vision::RgbImage& image) const;
    float train_step(std::string_view prompt, const vision::RgbImage& image);

private:
    PromptLatentGenerator& generator_;
    const ImageAutoencoder& target_encoder_;
    ImageTrainerConfig config_;
    train::AdamW optimizer_;
};

class SpiralImageGenerator final {
public:
    SpiralImageGenerator(ImageAutoencoder& autoencoder, PromptLatentGenerator& generator)
        : autoencoder_(autoencoder), generator_(generator) {}

    [[nodiscard]] vision::RgbImage generate(
        std::string_view prompt,
        std::size_t grid_height,
        std::size_t grid_width) const;

private:
    ImageAutoencoder& autoencoder_;
    PromptLatentGenerator& generator_;
};

void save_image_bundle(
    const ImageAutoencoder& autoencoder,
    const PromptLatentGenerator& generator,
    const std::string& path);

void load_image_bundle(
    ImageAutoencoder& autoencoder,
    PromptLatentGenerator& generator,
    const std::string& path);

} // namespace spiral::multimodal
