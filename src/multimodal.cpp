#include "spiral/multimodal.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace spiral::multimodal {
namespace {

void require_same_shape(const Tensor& lhs, const Tensor& rhs, const char* operation) {
    if (lhs.shape() != rhs.shape()) {
        throw std::invalid_argument(std::string(operation) + " requires matching tensor shapes");
    }
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
    std::string_view prompt,
    const PromptGeneratorConfig& config,
    std::size_t grid_height,
    std::size_t grid_width) {
    if (grid_height == 0 || grid_width == 0) throw std::invalid_argument("conditioning grid must be non-zero");
    Tensor features = prompt_features(prompt, config.text_feature_dim);
    Tensor matrix({grid_height * grid_width, config.text_feature_dim + 2});
    for (std::size_t y = 0; y < grid_height; ++y) {
        for (std::size_t x = 0; x < grid_width; ++x) {
            const std::size_t row = y * grid_width + x;
            for (std::size_t d = 0; d < config.text_feature_dim; ++d) {
                matrix.data()[row * (config.text_feature_dim + 2) + d] = features.data()[d];
            }
            const float nx = grid_width == 1 ? 0.0F : (2.0F * static_cast<float>(x) / static_cast<float>(grid_width - 1) - 1.0F);
            const float ny = grid_height == 1 ? 0.0F : (2.0F * static_cast<float>(y) / static_cast<float>(grid_height - 1) - 1.0F);
            matrix.data()[row * (config.text_feature_dim + 2) + config.text_feature_dim] = nx;
            matrix.data()[row * (config.text_feature_dim + 2) + config.text_feature_dim + 1] = ny;
        }
    }
    return matrix;
}

std::vector<nn::Parameter*> concat_parameters(std::vector<nn::Parameter*> first, std::vector<nn::Parameter*> second) {
    first.insert(first.end(), second.begin(), second.end());
    return first;
}

std::vector<const nn::Parameter*> concat_parameters(
    std::vector<const nn::Parameter*> first,
    std::vector<const nn::Parameter*> second) {
    first.insert(first.end(), second.begin(), second.end());
    return first;
}

void write_u64(std::ostream& out, std::uint64_t value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

std::uint64_t read_u64(std::istream& in) {
    std::uint64_t value = 0;
    in.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!in) throw std::runtime_error("truncated Spiral image bundle");
    return value;
}

void write_parameters(std::ostream& out, const std::vector<const nn::Parameter*>& parameters) {
    write_u64(out, static_cast<std::uint64_t>(parameters.size()));
    for (const auto* parameter : parameters) {
        write_u64(out, static_cast<std::uint64_t>(parameter->value.rank()));
        for (const auto dimension : parameter->value.shape()) write_u64(out, static_cast<std::uint64_t>(dimension));
        write_u64(out, static_cast<std::uint64_t>(parameter->value.numel()));
        out.write(
            reinterpret_cast<const char*>(parameter->value.data().data()),
            static_cast<std::streamsize>(parameter->value.numel() * sizeof(float)));
        if (!out) throw std::runtime_error("failed while writing Spiral image bundle");
    }
}

void read_parameters(std::istream& in, const std::vector<nn::Parameter*>& parameters) {
    const auto count = read_u64(in);
    if (count != parameters.size()) throw std::runtime_error("Spiral image bundle parameter count mismatch");
    for (auto* parameter : parameters) {
        const auto rank = read_u64(in);
        if (rank != parameter->value.rank()) throw std::runtime_error("Spiral image bundle rank mismatch");
        for (std::size_t i = 0; i < parameter->value.rank(); ++i) {
            if (read_u64(in) != parameter->value.shape()[i]) throw std::runtime_error("Spiral image bundle shape mismatch");
        }
        const auto numel = read_u64(in);
        if (numel != parameter->value.numel()) throw std::runtime_error("Spiral image bundle tensor size mismatch");
        in.read(
            reinterpret_cast<char*>(parameter->value.data().data()),
            static_cast<std::streamsize>(parameter->value.numel() * sizeof(float)));
        if (!in) throw std::runtime_error("truncated Spiral image bundle weights");
    }
}

} // namespace

Tensor prompt_features(std::string_view prompt, std::size_t feature_dim) {
    if (feature_dim == 0) throw std::invalid_argument("prompt feature dimension must be non-zero");
    Tensor features({feature_dim});
    for (const unsigned char byte : prompt) {
        const std::size_t primary = static_cast<std::size_t>(byte) % feature_dim;
        const std::size_t secondary = (static_cast<std::size_t>(byte) * 131U + 17U) % feature_dim;
        features.data()[primary] += 1.0F;
        features.data()[secondary] += 0.5F;
    }
    double norm_sq = 0.0;
    for (const float value : features.data()) norm_sq += static_cast<double>(value) * value;
    if (norm_sq > 0.0) {
        const float inv_norm = 1.0F / static_cast<float>(std::sqrt(norm_sq));
        for (float& value : features.data()) value *= inv_norm;
    }
    return features;
}

float mean_squared_error(const Tensor& prediction, const Tensor& target) {
    require_same_shape(prediction, target, "mean_squared_error");
    if (prediction.numel() == 0) throw std::invalid_argument("mean_squared_error requires non-empty tensors");
    double sum = 0.0;
    for (std::size_t i = 0; i < prediction.numel(); ++i) {
        const double delta = static_cast<double>(prediction.data()[i]) - target.data()[i];
        sum += delta * delta;
    }
    return static_cast<float>(sum / static_cast<double>(prediction.numel()));
}

float cosine_similarity(const Tensor& lhs, const Tensor& rhs) {
    if (lhs.rank() != 1 || rhs.rank() != 1 || lhs.shape() != rhs.shape() || lhs.numel() == 0) {
        throw std::invalid_argument("cosine_similarity requires equal non-empty rank-1 tensors");
    }
    double dot = 0.0;
    double lhs_norm = 0.0;
    double rhs_norm = 0.0;
    for (std::size_t i = 0; i < lhs.numel(); ++i) {
        dot += static_cast<double>(lhs.data()[i]) * rhs.data()[i];
        lhs_norm += static_cast<double>(lhs.data()[i]) * lhs.data()[i];
        rhs_norm += static_cast<double>(rhs.data()[i]) * rhs.data()[i];
    }
    if (!(lhs_norm > 0.0) || !(rhs_norm > 0.0)) return 0.0F;
    return static_cast<float>(dot / std::sqrt(lhs_norm * rhs_norm));
}

ImageAutoencoder::ImageAutoencoder(AutoencoderConfig config, Random& rng)
    : config_(config),
      encoder_(config.patch_size * config.patch_size * 3, config.latent_dim, rng, true),
      decoder_(config.latent_dim, config.patch_size * config.patch_size * 3, rng, true) {
    if (config_.patch_size == 0 || config_.latent_dim == 0) throw std::invalid_argument("autoencoder dimensions must be non-zero");
}

Tensor ImageAutoencoder::encode(const vision::RgbImage& image) const {
    return encoder_.forward(vision::patchify(vision::image_to_tensor(image), config_.patch_size));
}

vision::RgbImage ImageAutoencoder::decode(const Tensor& latent, std::size_t grid_height, std::size_t grid_width) const {
    if (latent.rank() != 2 || latent.shape()[0] != grid_height * grid_width || latent.shape()[1] != config_.latent_dim) {
        throw std::invalid_argument("latent tensor shape mismatch");
    }
    const Tensor patches = decoder_.forward(latent);
    const std::size_t patch_dim = config_.patch_size * config_.patch_size * 3;
    Tensor image({grid_height * config_.patch_size, grid_width * config_.patch_size, 3});
    const std::size_t width = image.shape()[1];
    for (std::size_t gy = 0; gy < grid_height; ++gy) {
        for (std::size_t gx = 0; gx < grid_width; ++gx) {
            const std::size_t patch_index = gy * grid_width + gx;
            std::size_t feature = 0;
            for (std::size_t py = 0; py < config_.patch_size; ++py) {
                for (std::size_t px = 0; px < config_.patch_size; ++px) {
                    const std::size_t y = gy * config_.patch_size + py;
                    const std::size_t x = gx * config_.patch_size + px;
                    const std::size_t image_base = (y * width + x) * 3;
                    for (std::size_t channel = 0; channel < 3; ++channel) {
                        image.data()[image_base + channel] = patches.data()[patch_index * patch_dim + feature++];
                    }
                }
            }
        }
    }
    return vision::tensor_to_image(image);
}

vision::RgbImage ImageAutoencoder::reconstruct(const vision::RgbImage& image) const {
    if (image.width() % config_.patch_size != 0 || image.height() % config_.patch_size != 0) {
        throw std::invalid_argument("image dimensions must be divisible by autoencoder patch size");
    }
    return decode(encode(image), image.height() / config_.patch_size, image.width() / config_.patch_size);
}

std::vector<nn::Parameter*> ImageAutoencoder::parameters() {
    return concat_parameters(encoder_.parameters(), decoder_.parameters());
}

std::vector<const nn::Parameter*> ImageAutoencoder::parameters() const {
    return concat_parameters(encoder_.parameters(), decoder_.parameters());
}

PromptLatentGenerator::PromptLatentGenerator(PromptGeneratorConfig config, Random& rng)
    : config_(config), projection_(config.text_feature_dim + 2, config.latent_dim, rng, true) {
    if (config_.text_feature_dim == 0 || config_.latent_dim == 0) {
        throw std::invalid_argument("prompt generator dimensions must be non-zero");
    }
}

Tensor PromptLatentGenerator::conditioning_matrix(
    std::string_view prompt,
    std::size_t grid_height,
    std::size_t grid_width) const {
    return build_conditioning(prompt, config_, grid_height, grid_width);
}

Tensor PromptLatentGenerator::predict(std::string_view prompt, std::size_t grid_height, std::size_t grid_width) const {
    return projection_.forward(conditioning_matrix(prompt, grid_height, grid_width));
}

std::vector<nn::Parameter*> PromptLatentGenerator::parameters() { return projection_.parameters(); }
std::vector<const nn::Parameter*> PromptLatentGenerator::parameters() const { return projection_.parameters(); }

AutoencoderTrainer::AutoencoderTrainer(ImageAutoencoder& model, ImageTrainerConfig config)
    : model_(model), config_(config), optimizer_(model.parameters(), config.optimizer) {}

float AutoencoderTrainer::evaluate(const vision::RgbImage& image) const {
    const Tensor patches = vision::patchify(vision::image_to_tensor(image), model_.config().patch_size);
    const Tensor prediction = model_.decoder().forward(model_.encoder().forward(patches));
    return mean_squared_error(prediction, patches);
}

float AutoencoderTrainer::train_step(const vision::RgbImage& image) {
    const Tensor patches = vision::patchify(vision::image_to_tensor(image), model_.config().patch_size);
    const Tensor latent = model_.encoder().forward(patches);
    const Tensor prediction = model_.decoder().forward(latent);
    const float loss = mean_squared_error(prediction, patches);

    optimizer_.zero_grad();
    const Tensor grad_prediction = mse_gradient(prediction, patches);
    const Tensor grad_latent = linear_backward(model_.decoder(), latent, grad_prediction);
    (void)linear_backward(model_.encoder(), patches, grad_latent);
    auto parameters = model_.parameters();
    train::clip_grad_norm(parameters, config_.max_grad_norm);
    optimizer_.step();
    return loss;
}

PromptGeneratorTrainer::PromptGeneratorTrainer(
    PromptLatentGenerator& generator,
    const ImageAutoencoder& target_encoder,
    ImageTrainerConfig config)
    : generator_(generator),
      target_encoder_(target_encoder),
      config_(config),
      optimizer_(generator.parameters(), config.optimizer) {
    if (generator_.config().latent_dim != target_encoder_.config().latent_dim) {
        throw std::invalid_argument("prompt generator latent dimension must match autoencoder");
    }
}

float PromptGeneratorTrainer::evaluate(std::string_view prompt, const vision::RgbImage& image) const {
    const std::size_t patch = target_encoder_.config().patch_size;
    if (image.width() % patch != 0 || image.height() % patch != 0) throw std::invalid_argument("image dimensions must align to patch size");
    const std::size_t grid_height = image.height() / patch;
    const std::size_t grid_width = image.width() / patch;
    return mean_squared_error(generator_.predict(prompt, grid_height, grid_width), target_encoder_.encode(image));
}

float PromptGeneratorTrainer::train_step(std::string_view prompt, const vision::RgbImage& image) {
    const std::size_t patch = target_encoder_.config().patch_size;
    if (image.width() % patch != 0 || image.height() % patch != 0) throw std::invalid_argument("image dimensions must align to patch size");
    const std::size_t grid_height = image.height() / patch;
    const std::size_t grid_width = image.width() / patch;
    const Tensor input = build_conditioning(prompt, generator_.config(), grid_height, grid_width);
    const Tensor target = target_encoder_.encode(image);
    const Tensor prediction = generator_.projection().forward(input);
    const float loss = mean_squared_error(prediction, target);

    optimizer_.zero_grad();
    const Tensor grad_prediction = mse_gradient(prediction, target);
    (void)linear_backward(generator_.projection(), input, grad_prediction);
    auto parameters = generator_.parameters();
    train::clip_grad_norm(parameters, config_.max_grad_norm);
    optimizer_.step();
    return loss;
}

vision::RgbImage SpiralImageGenerator::generate(
    std::string_view prompt,
    std::size_t grid_height,
    std::size_t grid_width) const {
    return autoencoder_.decode(generator_.predict(prompt, grid_height, grid_width), grid_height, grid_width);
}

void save_image_bundle(
    const ImageAutoencoder& autoencoder,
    const PromptLatentGenerator& generator,
    const std::string& path) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("failed to open Spiral image bundle for writing");
    const char magic[] = "SPIRALIMG1";
    out.write(magic, sizeof(magic) - 1);
    write_u64(out, autoencoder.config().patch_size);
    write_u64(out, autoencoder.config().latent_dim);
    write_u64(out, generator.config().text_feature_dim);
    write_u64(out, generator.config().latent_dim);
    write_parameters(out, autoencoder.parameters());
    write_parameters(out, generator.parameters());
}

void load_image_bundle(
    ImageAutoencoder& autoencoder,
    PromptLatentGenerator& generator,
    const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("failed to open Spiral image bundle for reading");
    char magic[10]{};
    in.read(magic, 10);
    if (!in || std::string(magic, 10) != "SPIRALIMG1") throw std::runtime_error("invalid Spiral image bundle");
    if (read_u64(in) != autoencoder.config().patch_size ||
        read_u64(in) != autoencoder.config().latent_dim ||
        read_u64(in) != generator.config().text_feature_dim ||
        read_u64(in) != generator.config().latent_dim) {
        throw std::runtime_error("Spiral image bundle configuration mismatch");
    }
    read_parameters(in, autoencoder.parameters());
    read_parameters(in, generator.parameters());
}

} // namespace spiral::multimodal
