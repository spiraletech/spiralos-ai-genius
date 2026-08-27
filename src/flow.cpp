#include "spiral/flow.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace spiral::flow {
namespace {

constexpr float kPi = 3.14159265358979323846F;

float clamp_unit(float value) {
    if (!std::isfinite(value)) throw std::invalid_argument("time/strength must be finite");
    return std::clamp(value, 0.0F, 1.0F);
}

void require_same_shape(const Tensor& lhs, const Tensor& rhs, const char* operation) {
    if (lhs.shape() != rhs.shape()) {
        throw std::invalid_argument(std::string(operation) + " requires matching tensor shapes");
    }
}

float silu(float x) {
    if (x >= 0.0F) {
        const float z = std::exp(-x);
        return x / (1.0F + z);
    }
    const float z = std::exp(x);
    return x * z / (1.0F + z);
}

float silu_derivative(float x) {
    float sigmoid = 0.0F;
    if (x >= 0.0F) {
        const float z = std::exp(-x);
        sigmoid = 1.0F / (1.0F + z);
    } else {
        const float z = std::exp(x);
        sigmoid = z / (1.0F + z);
    }
    return sigmoid * (1.0F + x * (1.0F - sigmoid));
}

Tensor apply_silu(const Tensor& input) {
    Tensor output(input.shape());
    for (std::size_t i = 0; i < input.numel(); ++i) output.data()[i] = silu(input.data()[i]);
    return output;
}

Tensor mse_gradient(const Tensor& prediction, const Tensor& target) {
    require_same_shape(prediction, target, "MSE gradient");
    if (prediction.numel() == 0) throw std::invalid_argument("MSE gradient requires non-empty tensors");
    Tensor grad(prediction.shape());
    const float scale = 2.0F / static_cast<float>(prediction.numel());
    for (std::size_t i = 0; i < prediction.numel(); ++i) {
        grad.data()[i] = scale * (prediction.data()[i] - target.data()[i]);
    }
    return grad;
}

Tensor linear_backward(nn::Linear& layer, const Tensor& input, const Tensor& grad_output) {
    if (input.rank() != 2 || grad_output.rank() != 2) {
        throw std::invalid_argument("linear_backward requires rank-2 tensors");
    }
    const std::size_t rows = input.shape()[0];
    const std::size_t in_features = input.shape()[1];
    const std::size_t out_features = grad_output.shape()[1];
    if (rows != grad_output.shape()[0] || in_features != layer.in_features() || out_features != layer.out_features()) {
        throw std::invalid_argument("linear_backward shape mismatch");
    }

    auto& weight = layer.weight();
    weight.ensure_grad();
    nn::Parameter* bias = nullptr;
    if (layer.uses_bias()) {
        bias = &layer.bias();
        bias->ensure_grad();
    }

    Tensor grad_input({rows, in_features});
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t out = 0; out < out_features; ++out) {
            const float grad = grad_output.data()[row * out_features + out];
            if (bias != nullptr) bias->grad.data()[out] += grad;
            for (std::size_t in = 0; in < in_features; ++in) {
                weight.grad.data()[in * out_features + out] += input.data()[row * in_features + in] * grad;
                grad_input.data()[row * in_features + in] += weight.value.data()[in * out_features + out] * grad;
            }
        }
    }
    return grad_input;
}

Tensor build_conditioning(
    const Tensor& noisy_latent,
    std::string_view prompt,
    float time,
    const DenoiserConfig& config,
    std::size_t grid_height,
    std::size_t grid_width) {
    if (grid_height == 0 || grid_width == 0) throw std::invalid_argument("denoiser grid must be non-zero");
    if (noisy_latent.rank() != 2 || noisy_latent.shape()[0] != grid_height * grid_width ||
        noisy_latent.shape()[1] != config.latent_dim) {
        throw std::invalid_argument("denoiser latent shape mismatch");
    }

    const Tensor text = multimodal::prompt_features(prompt, config.text_feature_dim);
    const Tensor time_features = timestep_features(time, config.time_feature_dim);
    const std::size_t row_width = config.latent_dim + config.text_feature_dim + 2 + config.time_feature_dim;
    Tensor matrix({grid_height * grid_width, row_width});

    for (std::size_t y = 0; y < grid_height; ++y) {
        for (std::size_t x = 0; x < grid_width; ++x) {
            const std::size_t row = y * grid_width + x;
            std::size_t column = 0;
            for (std::size_t d = 0; d < config.latent_dim; ++d) {
                matrix.data()[row * row_width + column++] = noisy_latent.data()[row * config.latent_dim + d];
            }
            for (std::size_t d = 0; d < config.text_feature_dim; ++d) {
                matrix.data()[row * row_width + column++] = text.data()[d];
            }
            const float nx = grid_width == 1 ? 0.0F :
                (2.0F * static_cast<float>(x) / static_cast<float>(grid_width - 1) - 1.0F);
            const float ny = grid_height == 1 ? 0.0F :
                (2.0F * static_cast<float>(y) / static_cast<float>(grid_height - 1) - 1.0F);
            matrix.data()[row * row_width + column++] = nx;
            matrix.data()[row * row_width + column++] = ny;
            for (std::size_t d = 0; d < config.time_feature_dim; ++d) {
                matrix.data()[row * row_width + column++] = time_features.data()[d];
            }
        }
    }
    return matrix;
}

std::vector<nn::Parameter*> concat_parameters(
    std::vector<nn::Parameter*> first,
    std::vector<nn::Parameter*> second) {
    first.insert(first.end(), second.begin(), second.end());
    return first;
}

std::vector<const nn::Parameter*> concat_parameters(
    std::vector<const nn::Parameter*> first,
    std::vector<const nn::Parameter*> second) {
    first.insert(first.end(), second.begin(), second.end());
    return first;
}

void require_source_alignment(const multimodal::ImageAutoencoder& autoencoder, const vision::RgbImage& source) {
    const std::size_t patch = autoencoder.config().patch_size;
    if (source.width() == 0 || source.height() == 0 || source.width() % patch != 0 || source.height() % patch != 0) {
        throw std::invalid_argument("source image dimensions must align to autoencoder patch size");
    }
}

} // namespace

float NoiseScheduler::noise_level(float time) const {
    const float t = clamp_unit(time);
    if (kind_ == NoiseScheduleKind::Linear) return t;
    return std::sin(0.5F * kPi * t);
}

Tensor NoiseScheduler::add_noise(const Tensor& clean, const Tensor& noise, float time) const {
    require_same_shape(clean, noise, "NoiseScheduler::add_noise");
    const float amount = noise_level(time);
    Tensor mixed(clean.shape());
    for (std::size_t i = 0; i < clean.numel(); ++i) {
        mixed.data()[i] = clean.data()[i] * (1.0F - amount) + noise.data()[i] * amount;
    }
    return mixed;
}

std::vector<float> NoiseScheduler::sampling_times(std::size_t steps, float start_time) const {
    if (steps == 0) throw std::invalid_argument("sampling requires at least one step");
    const float start = clamp_unit(start_time);
    std::vector<float> times(steps + 1);
    for (std::size_t i = 0; i <= steps; ++i) {
        times[i] = start * (1.0F - static_cast<float>(i) / static_cast<float>(steps));
    }
    return times;
}

Tensor timestep_features(float time, std::size_t feature_dim) {
    if (feature_dim == 0) throw std::invalid_argument("timestep feature dimension must be non-zero");
    const float t = clamp_unit(time);
    Tensor features({feature_dim});
    for (std::size_t d = 0; d < feature_dim; ++d) {
        const std::size_t pair = d / 2;
        const double frequency = std::pow(10000.0, -2.0 * static_cast<double>(pair) / static_cast<double>(feature_dim));
        const double angle = static_cast<double>(t) * frequency * 1000.0;
        features.data()[d] = static_cast<float>((d % 2 == 0) ? std::sin(angle) : std::cos(angle));
    }
    return features;
}

Tensor gaussian_noise(const std::vector<std::size_t>& shape, Random& rng) {
    Tensor noise(shape);
    rng.fill_normal(noise, 0.0F, 1.0F);
    return noise;
}

Tensor guided_prediction(const Tensor& unconditional, const Tensor& conditional, float guidance_scale) {
    require_same_shape(unconditional, conditional, "guided_prediction");
    if (!std::isfinite(guidance_scale)) throw std::invalid_argument("guidance scale must be finite");
    Tensor guided(unconditional.shape());
    for (std::size_t i = 0; i < unconditional.numel(); ++i) {
        guided.data()[i] = unconditional.data()[i] +
            guidance_scale * (conditional.data()[i] - unconditional.data()[i]);
    }
    return guided;
}

LatentDenoiser::LatentDenoiser(DenoiserConfig config, Random& rng)
    : config_(config),
      input_projection_(config.latent_dim + config.text_feature_dim + 2 + config.time_feature_dim, config.hidden_dim, rng, true),
      output_projection_(config.hidden_dim, config.latent_dim, rng, true) {
    if (config_.text_feature_dim == 0 || config_.latent_dim == 0 ||
        config_.time_feature_dim == 0 || config_.hidden_dim == 0) {
        throw std::invalid_argument("denoiser dimensions must be non-zero");
    }
}

Tensor LatentDenoiser::conditioning_matrix(
    const Tensor& noisy_latent,
    std::string_view prompt,
    float time,
    std::size_t grid_height,
    std::size_t grid_width) const {
    return build_conditioning(noisy_latent, prompt, time, config_, grid_height, grid_width);
}

Tensor LatentDenoiser::predict(
    const Tensor& noisy_latent,
    std::string_view prompt,
    float time,
    std::size_t grid_height,
    std::size_t grid_width) const {
    const Tensor conditioning = conditioning_matrix(noisy_latent, prompt, time, grid_height, grid_width);
    return output_projection_.forward(apply_silu(input_projection_.forward(conditioning)));
}

std::vector<nn::Parameter*> LatentDenoiser::parameters() {
    return concat_parameters(input_projection_.parameters(), output_projection_.parameters());
}

std::vector<const nn::Parameter*> LatentDenoiser::parameters() const {
    return concat_parameters(input_projection_.parameters(), output_projection_.parameters());
}

DenoiserTrainer::DenoiserTrainer(
    LatentDenoiser& denoiser,
    const multimodal::ImageAutoencoder& autoencoder,
    NoiseScheduler scheduler,
    DenoiserTrainerConfig config)
    : denoiser_(denoiser),
      autoencoder_(autoencoder),
      scheduler_(scheduler),
      config_(config),
      optimizer_(denoiser.parameters(), config.optimizer) {
    if (denoiser_.config().latent_dim != autoencoder_.config().latent_dim) {
        throw std::invalid_argument("denoiser latent dimension must match autoencoder");
    }
}

float DenoiserTrainer::evaluate(
    std::string_view prompt,
    const vision::RgbImage& image,
    float time,
    const Tensor& noise) const {
    require_source_alignment(autoencoder_, image);
    const Tensor clean = autoencoder_.encode(image);
    require_same_shape(clean, noise, "DenoiserTrainer::evaluate noise");
    const std::size_t grid_height = image.height() / autoencoder_.config().patch_size;
    const std::size_t grid_width = image.width() / autoencoder_.config().patch_size;
    const Tensor noisy = scheduler_.add_noise(clean, noise, time);
    return multimodal::mean_squared_error(denoiser_.predict(noisy, prompt, time, grid_height, grid_width), clean);
}

float DenoiserTrainer::train_step(
    std::string_view prompt,
    const vision::RgbImage& image,
    float time,
    const Tensor& noise) {
    require_source_alignment(autoencoder_, image);
    const Tensor clean = autoencoder_.encode(image);
    require_same_shape(clean, noise, "DenoiserTrainer::train_step noise");
    const std::size_t grid_height = image.height() / autoencoder_.config().patch_size;
    const std::size_t grid_width = image.width() / autoencoder_.config().patch_size;
    const Tensor noisy = scheduler_.add_noise(clean, noise, time);
    const Tensor conditioning = build_conditioning(noisy, prompt, time, denoiser_.config(), grid_height, grid_width);
    const Tensor hidden_pre = denoiser_.input_projection().forward(conditioning);
    const Tensor hidden = apply_silu(hidden_pre);
    const Tensor prediction = denoiser_.output_projection().forward(hidden);
    const float loss = multimodal::mean_squared_error(prediction, clean);

    optimizer_.zero_grad();
    const Tensor grad_prediction = mse_gradient(prediction, clean);
    Tensor grad_hidden = linear_backward(denoiser_.output_projection(), hidden, grad_prediction);
    for (std::size_t i = 0; i < grad_hidden.numel(); ++i) {
        grad_hidden.data()[i] *= silu_derivative(hidden_pre.data()[i]);
    }
    (void)linear_backward(denoiser_.input_projection(), conditioning, grad_hidden);
    auto parameters = denoiser_.parameters();
    train::clip_grad_norm(parameters, config_.max_grad_norm);
    optimizer_.step();
    return loss;
}

Tensor IterativeImageGenerator::refine(
    Tensor current,
    std::string_view prompt,
    std::size_t grid_height,
    std::size_t grid_width,
    float start_time,
    SamplingConfig config,
    const Tensor* preserved_latent,
    const Tensor* patch_mask) const {
    if (config.steps == 0) throw std::invalid_argument("sampling requires at least one step");
    if (current.rank() != 2 || current.shape()[0] != grid_height * grid_width ||
        current.shape()[1] != denoiser_.config().latent_dim) {
        throw std::invalid_argument("sampling latent shape mismatch");
    }
    if (preserved_latent != nullptr) require_same_shape(current, *preserved_latent, "preserved latent");
    if (patch_mask != nullptr) {
        if (patch_mask->rank() != 1 || patch_mask->numel() != grid_height * grid_width) {
            throw std::invalid_argument("patch mask must be rank-1 with one value per latent patch");
        }
    }

    const auto times = scheduler_.sampling_times(config.steps, start_time);
    for (std::size_t step = 0; step < config.steps; ++step) {
        const float time = times[step];
        const float next_time = times[step + 1];
        const Tensor unconditional = denoiser_.predict(current, "", time, grid_height, grid_width);
        const Tensor conditional = denoiser_.predict(current, prompt, time, grid_height, grid_width);
        const Tensor target = guided_prediction(unconditional, conditional, config.guidance_scale);
        const float relaxation = time <= std::numeric_limits<float>::epsilon()
            ? 1.0F
            : std::clamp((time - next_time) / time, 0.0F, 1.0F);

        for (std::size_t row = 0; row < current.shape()[0]; ++row) {
            const float mask = patch_mask == nullptr ? 1.0F : std::clamp(patch_mask->data()[row], 0.0F, 1.0F);
            for (std::size_t d = 0; d < current.shape()[1]; ++d) {
                const std::size_t index = row * current.shape()[1] + d;
                const float candidate = current.data()[index] + relaxation * (target.data()[index] - current.data()[index]);
                if (preserved_latent != nullptr) {
                    current.data()[index] = mask * candidate + (1.0F - mask) * preserved_latent->data()[index];
                } else {
                    current.data()[index] = candidate;
                }
            }
        }
    }
    return current;
}

Tensor IterativeImageGenerator::generate_latent(
    std::string_view prompt,
    std::size_t grid_height,
    std::size_t grid_width,
    SamplingConfig config) const {
    if (grid_height == 0 || grid_width == 0) throw std::invalid_argument("generation grid must be non-zero");
    Random rng(config.seed);
    Tensor current = gaussian_noise({grid_height * grid_width, denoiser_.config().latent_dim}, rng);
    return refine(std::move(current), prompt, grid_height, grid_width, 1.0F, config);
}

vision::RgbImage IterativeImageGenerator::generate(
    std::string_view prompt,
    std::size_t grid_height,
    std::size_t grid_width,
    SamplingConfig config) const {
    return autoencoder_.decode(generate_latent(prompt, grid_height, grid_width, config), grid_height, grid_width);
}

vision::RgbImage IterativeImageGenerator::image_to_image(
    std::string_view prompt,
    const vision::RgbImage& source,
    float strength,
    SamplingConfig config) const {
    require_source_alignment(autoencoder_, source);
    const float start = clamp_unit(strength);
    if (start == 0.0F) return autoencoder_.reconstruct(source);

    const std::size_t grid_height = source.height() / autoencoder_.config().patch_size;
    const std::size_t grid_width = source.width() / autoencoder_.config().patch_size;
    const Tensor clean = autoencoder_.encode(source);
    Random rng(config.seed);
    const Tensor noise = gaussian_noise(clean.shape(), rng);
    Tensor current = scheduler_.add_noise(clean, noise, start);
    current = refine(std::move(current), prompt, grid_height, grid_width, start, config);
    return autoencoder_.decode(current, grid_height, grid_width);
}

vision::RgbImage IterativeImageGenerator::inpaint(
    std::string_view prompt,
    const vision::RgbImage& source,
    const Tensor& patch_mask,
    float strength,
    SamplingConfig config) const {
    require_source_alignment(autoencoder_, source);
    const float start = clamp_unit(strength);
    const std::size_t grid_height = source.height() / autoencoder_.config().patch_size;
    const std::size_t grid_width = source.width() / autoencoder_.config().patch_size;
    if (patch_mask.rank() != 1 || patch_mask.numel() != grid_height * grid_width) {
        throw std::invalid_argument("inpaint mask must contain one value per latent patch");
    }
    const Tensor clean = autoencoder_.encode(source);
    if (start == 0.0F) return autoencoder_.decode(clean, grid_height, grid_width);

    Random rng(config.seed);
    const Tensor noise = gaussian_noise(clean.shape(), rng);
    Tensor current = scheduler_.add_noise(clean, noise, start);
    current = refine(std::move(current), prompt, grid_height, grid_width, start, config, &clean, &patch_mask);
    return autoencoder_.decode(current, grid_height, grid_width);
}

} // namespace spiral::flow
