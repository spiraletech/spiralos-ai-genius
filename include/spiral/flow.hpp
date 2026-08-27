#pragma once

#include "spiral/multimodal.hpp"
#include "spiral/nn.hpp"
#include "spiral/random.hpp"
#include "spiral/tensor.hpp"
#include "spiral/train.hpp"
#include "spiral/vision.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace spiral::flow {

enum class NoiseScheduleKind {
    Linear,
    Cosine,
};

class NoiseScheduler final {
public:
    explicit NoiseScheduler(NoiseScheduleKind kind = NoiseScheduleKind::Cosine) noexcept : kind_(kind) {}

    [[nodiscard]] float noise_level(float time) const;
    [[nodiscard]] Tensor add_noise(const Tensor& clean, const Tensor& noise, float time) const;
    [[nodiscard]] std::vector<float> sampling_times(std::size_t steps, float start_time = 1.0F) const;
    [[nodiscard]] NoiseScheduleKind kind() const noexcept { return kind_; }

private:
    NoiseScheduleKind kind_;
};

[[nodiscard]] Tensor timestep_features(float time, std::size_t feature_dim);
[[nodiscard]] Tensor gaussian_noise(const std::vector<std::size_t>& shape, Random& rng);
[[nodiscard]] Tensor guided_prediction(const Tensor& unconditional, const Tensor& conditional, float guidance_scale);

struct LatentPredictorConfig {
    std::size_t latent_dim = 0;
};

class LatentPredictor {
public:
    virtual ~LatentPredictor() = default;
    [[nodiscard]] virtual std::size_t latent_dim() const noexcept = 0;
    [[nodiscard]] LatentPredictorConfig config() const noexcept { return {latent_dim()}; }
    [[nodiscard]] virtual Tensor predict(
        const Tensor& noisy_latent,
        std::string_view prompt,
        float time,
        std::size_t grid_height,
        std::size_t grid_width) const = 0;
};

struct DenoiserConfig {
    std::size_t text_feature_dim = 32;
    std::size_t latent_dim = 8;
    std::size_t time_feature_dim = 8;
    std::size_t hidden_dim = 64;
};

class LatentDenoiser final : public LatentPredictor {
public:
    LatentDenoiser(DenoiserConfig config, Random& rng);

    [[nodiscard]] std::size_t latent_dim() const noexcept override { return config_.latent_dim; }
    [[nodiscard]] Tensor predict(
        const Tensor& noisy_latent,
        std::string_view prompt,
        float time,
        std::size_t grid_height,
        std::size_t grid_width) const override;

    [[nodiscard]] std::vector<nn::Parameter*> parameters();
    [[nodiscard]] std::vector<const nn::Parameter*> parameters() const;
    [[nodiscard]] const DenoiserConfig& config() const noexcept { return config_; }
    [[nodiscard]] nn::Linear& input_projection() noexcept { return input_projection_; }
    [[nodiscard]] const nn::Linear& input_projection() const noexcept { return input_projection_; }
    [[nodiscard]] nn::Linear& output_projection() noexcept { return output_projection_; }
    [[nodiscard]] const nn::Linear& output_projection() const noexcept { return output_projection_; }

private:
    [[nodiscard]] Tensor conditioning_matrix(
        const Tensor& noisy_latent,
        std::string_view prompt,
        float time,
        std::size_t grid_height,
        std::size_t grid_width) const;

    DenoiserConfig config_;
    nn::Linear input_projection_;
    nn::Linear output_projection_;
};

struct DenoiserTrainerConfig {
    train::AdamWConfig optimizer{0.01F, 0.9F, 0.999F, 1.0e-8F, 0.0F};
    float max_grad_norm = 5.0F;
};

class DenoiserTrainer final {
public:
    DenoiserTrainer(
        LatentDenoiser& denoiser,
        const multimodal::ImageAutoencoder& autoencoder,
        NoiseScheduler scheduler = NoiseScheduler{},
        DenoiserTrainerConfig config = {});

    [[nodiscard]] float evaluate(
        std::string_view prompt,
        const vision::RgbImage& image,
        float time,
        const Tensor& noise) const;

    float train_step(
        std::string_view prompt,
        const vision::RgbImage& image,
        float time,
        const Tensor& noise);

private:
    LatentDenoiser& denoiser_;
    const multimodal::ImageAutoencoder& autoencoder_;
    NoiseScheduler scheduler_;
    DenoiserTrainerConfig config_;
    train::AdamW optimizer_;
};

struct SamplingConfig {
    std::size_t steps = 8;
    float guidance_scale = 1.0F;
    std::uint64_t seed = 0x53504952414CULL;
};

class IterativeImageGenerator final {
public:
    IterativeImageGenerator(
        multimodal::ImageAutoencoder& autoencoder,
        LatentPredictor& denoiser,
        NoiseScheduler scheduler = NoiseScheduler{})
        : autoencoder_(autoencoder), denoiser_(denoiser), scheduler_(scheduler) {}

    [[nodiscard]] Tensor generate_latent(
        std::string_view prompt,
        std::size_t grid_height,
        std::size_t grid_width,
        SamplingConfig config = {}) const;

    [[nodiscard]] vision::RgbImage generate(
        std::string_view prompt,
        std::size_t grid_height,
        std::size_t grid_width,
        SamplingConfig config = {}) const;

    [[nodiscard]] vision::RgbImage image_to_image(
        std::string_view prompt,
        const vision::RgbImage& source,
        float strength,
        SamplingConfig config = {}) const;

    [[nodiscard]] vision::RgbImage inpaint(
        std::string_view prompt,
        const vision::RgbImage& source,
        const Tensor& patch_mask,
        float strength,
        SamplingConfig config = {}) const;

private:
    [[nodiscard]] Tensor refine(
        Tensor current,
        std::string_view prompt,
        std::size_t grid_height,
        std::size_t grid_width,
        float start_time,
        SamplingConfig config,
        const Tensor* preserved_latent = nullptr,
        const Tensor* patch_mask = nullptr) const;

    multimodal::ImageAutoencoder& autoencoder_;
    LatentPredictor& denoiser_;
    NoiseScheduler scheduler_;
};

} // namespace spiral::flow
