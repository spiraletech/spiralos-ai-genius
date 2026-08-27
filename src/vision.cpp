#include "spiral/vision.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace spiral::vision {
namespace {

std::size_t checked_image_bytes(std::size_t width, std::size_t height) {
    if (width == 0 || height == 0) throw std::invalid_argument("image dimensions must be non-zero");
    if (width > std::numeric_limits<std::size_t>::max() / height) throw std::overflow_error("image dimensions overflow");
    const std::size_t pixels = width * height;
    if (pixels > std::numeric_limits<std::size_t>::max() / 3) throw std::overflow_error("image storage overflow");
    return pixels * 3;
}

std::string read_ppm_token(std::istream& stream) {
    std::string token;
    char ch = 0;
    while (stream.get(ch)) {
        if (ch == '#') {
            std::string ignored;
            std::getline(stream, ignored);
            continue;
        }
        if (!std::isspace(static_cast<unsigned char>(ch))) {
            token.push_back(ch);
            break;
        }
    }
    while (stream.get(ch)) {
        if (std::isspace(static_cast<unsigned char>(ch))) break;
        if (ch == '#') {
            std::string ignored;
            std::getline(stream, ignored);
            break;
        }
        token.push_back(ch);
    }
    return token;
}

std::size_t parse_size_token(const std::string& token, const char* field) {
    if (token.empty()) throw std::runtime_error(std::string("missing PPM ") + field);
    std::size_t consumed = 0;
    unsigned long long value = 0;
    try {
        value = std::stoull(token, &consumed);
    } catch (...) {
        throw std::runtime_error(std::string("invalid PPM ") + field);
    }
    if (consumed != token.size() || value == 0 || value > std::numeric_limits<std::size_t>::max()) {
        throw std::runtime_error(std::string("invalid PPM ") + field);
    }
    return static_cast<std::size_t>(value);
}

void append_parameters(std::vector<nn::Parameter*>& out, std::vector<nn::Parameter*> values) {
    out.insert(out.end(), values.begin(), values.end());
}

void append_parameters(std::vector<const nn::Parameter*>& out, std::vector<const nn::Parameter*> values) {
    out.insert(out.end(), values.begin(), values.end());
}

float positional_component(std::size_t position, std::size_t component, std::size_t dimensions) {
    if (dimensions == 0) return 0.0F;
    const std::size_t pair = component / 2;
    const double exponent = (2.0 * static_cast<double>(pair)) / static_cast<double>(dimensions);
    const double angle = static_cast<double>(position) / std::pow(10000.0, exponent);
    return static_cast<float>((component % 2 == 0) ? std::sin(angle) : std::cos(angle));
}

} // namespace

RgbImage::RgbImage(std::size_t width, std::size_t height, std::uint8_t fill)
    : width_(width), height_(height), pixels_(checked_image_bytes(width, height), fill) {}

RgbImage::RgbImage(std::size_t width, std::size_t height, std::vector<std::uint8_t> pixels)
    : width_(width), height_(height), pixels_(std::move(pixels)) {
    if (pixels_.size() != checked_image_bytes(width, height)) {
        throw std::invalid_argument("RGB pixel buffer size mismatch");
    }
}

std::size_t RgbImage::offset(std::size_t x, std::size_t y, std::size_t channel) const {
    if (x >= width_ || y >= height_ || channel >= 3) throw std::out_of_range("RGB pixel coordinate out of range");
    return (y * width_ + x) * 3 + channel;
}

std::uint8_t RgbImage::at(std::size_t x, std::size_t y, std::size_t channel) const {
    return pixels_[offset(x, y, channel)];
}

std::uint8_t& RgbImage::at(std::size_t x, std::size_t y, std::size_t channel) {
    return pixels_[offset(x, y, channel)];
}

void RgbImage::save_ppm(const std::string& path) const {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) throw std::runtime_error("failed to open PPM for writing");
    stream << "P6\n" << width_ << ' ' << height_ << "\n255\n";
    stream.write(reinterpret_cast<const char*>(pixels_.data()), static_cast<std::streamsize>(pixels_.size()));
    if (!stream) throw std::runtime_error("failed while writing PPM");
}

RgbImage RgbImage::load_ppm(const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("failed to open PPM for reading");
    if (read_ppm_token(stream) != "P6") throw std::runtime_error("only binary P6 PPM is supported");
    const std::size_t width = parse_size_token(read_ppm_token(stream), "width");
    const std::size_t height = parse_size_token(read_ppm_token(stream), "height");
    if (parse_size_token(read_ppm_token(stream), "max value") != 255) {
        throw std::runtime_error("PPM max value must be 255");
    }
    std::vector<std::uint8_t> pixels(checked_image_bytes(width, height));
    stream.read(reinterpret_cast<char*>(pixels.data()), static_cast<std::streamsize>(pixels.size()));
    if (stream.gcount() != static_cast<std::streamsize>(pixels.size())) throw std::runtime_error("PPM pixel data is truncated");
    return RgbImage(width, height, std::move(pixels));
}

Tensor image_to_tensor(const RgbImage& image) {
    std::vector<float> values;
    values.reserve(image.pixels().size());
    for (const auto value : image.pixels()) values.push_back(static_cast<float>(value) / 255.0F);
    return Tensor({image.height(), image.width(), 3}, std::move(values));
}

RgbImage tensor_to_image(const Tensor& tensor) {
    if (tensor.rank() != 3 || tensor.shape()[2] != 3 || tensor.shape()[0] == 0 || tensor.shape()[1] == 0) {
        throw std::invalid_argument("tensor_to_image requires [height,width,3]");
    }
    RgbImage image(tensor.shape()[1], tensor.shape()[0]);
    for (std::size_t i = 0; i < tensor.numel(); ++i) {
        const float clamped = std::clamp(tensor.data()[i], 0.0F, 1.0F);
        image.pixels()[i] = static_cast<std::uint8_t>(std::lround(clamped * 255.0F));
    }
    return image;
}

Tensor patchify(const Tensor& image, std::size_t patch_size) {
    if (image.rank() != 3 || image.shape()[2] != 3) throw std::invalid_argument("patchify requires [height,width,3]");
    if (patch_size == 0) throw std::invalid_argument("patch_size must be non-zero");
    const std::size_t height = image.shape()[0];
    const std::size_t width = image.shape()[1];
    if (height % patch_size != 0 || width % patch_size != 0) {
        throw std::invalid_argument("image dimensions must be divisible by patch_size");
    }
    const std::size_t grid_h = height / patch_size;
    const std::size_t grid_w = width / patch_size;
    const std::size_t patch_dim = patch_size * patch_size * 3;
    Tensor patches({grid_h * grid_w, patch_dim});
    for (std::size_t gy = 0; gy < grid_h; ++gy) {
        for (std::size_t gx = 0; gx < grid_w; ++gx) {
            const std::size_t patch_index = gy * grid_w + gx;
            std::size_t feature = 0;
            for (std::size_t py = 0; py < patch_size; ++py) {
                for (std::size_t px = 0; px < patch_size; ++px) {
                    const std::size_t y = gy * patch_size + py;
                    const std::size_t x = gx * patch_size + px;
                    const std::size_t image_base = (y * width + x) * 3;
                    for (std::size_t channel = 0; channel < 3; ++channel) {
                        patches.data()[patch_index * patch_dim + feature++] = image.data()[image_base + channel];
                    }
                }
            }
        }
    }
    return patches;
}

void add_2d_sincos_position(Tensor& tokens, std::size_t grid_height, std::size_t grid_width) {
    if (tokens.rank() != 2 || grid_height == 0 || grid_width == 0 || tokens.shape()[0] != grid_height * grid_width) {
        throw std::invalid_argument("2D position encoding shape mismatch");
    }
    const std::size_t dim = tokens.shape()[1];
    const std::size_t y_dims = dim / 2;
    const std::size_t x_dims = dim - y_dims;
    for (std::size_t y = 0; y < grid_height; ++y) {
        for (std::size_t x = 0; x < grid_width; ++x) {
            const std::size_t row = y * grid_width + x;
            for (std::size_t d = 0; d < y_dims; ++d) {
                tokens.data()[row * dim + d] += positional_component(y, d, y_dims);
            }
            for (std::size_t d = 0; d < x_dims; ++d) {
                tokens.data()[row * dim + y_dims + d] += positional_component(x, d, x_dims);
            }
        }
    }
}

VisionSelfAttention::VisionSelfAttention(std::size_t model_dim, std::size_t num_heads, Random& rng, bool use_bias)
    : model_dim_(model_dim),
      num_heads_(num_heads),
      head_dim_(num_heads == 0 ? 0 : model_dim / num_heads),
      q_proj_(model_dim, model_dim, rng, use_bias),
      k_proj_(model_dim, model_dim, rng, use_bias),
      v_proj_(model_dim, model_dim, rng, use_bias),
      out_proj_(model_dim, model_dim, rng, use_bias) {
    if (model_dim == 0 || num_heads == 0 || model_dim % num_heads != 0) {
        throw std::invalid_argument("vision attention dimensions must be non-zero and divisible by head count");
    }
}

Tensor VisionSelfAttention::forward(const Tensor& input) const {
    if (input.rank() != 2 || input.shape()[1] != model_dim_ || input.shape()[0] == 0) {
        throw std::invalid_argument("VisionSelfAttention requires [sequence,model_dim]");
    }
    const Tensor q = q_proj_.forward(input);
    const Tensor k = k_proj_.forward(input);
    const Tensor v = v_proj_.forward(input);
    const std::size_t sequence = input.shape()[0];
    Tensor context({sequence, model_dim_});
    const float scale = 1.0F / std::sqrt(static_cast<float>(head_dim_));

    std::vector<float> scores(sequence);
    std::vector<float> probabilities(sequence);
    for (std::size_t head = 0; head < num_heads_; ++head) {
        const std::size_t head_offset = head * head_dim_;
        for (std::size_t query = 0; query < sequence; ++query) {
            float max_score = -std::numeric_limits<float>::infinity();
            for (std::size_t key = 0; key < sequence; ++key) {
                float dot = 0.0F;
                for (std::size_t d = 0; d < head_dim_; ++d) {
                    dot += q.data()[query * model_dim_ + head_offset + d] *
                           k.data()[key * model_dim_ + head_offset + d];
                }
                scores[key] = dot * scale;
                max_score = std::max(max_score, scores[key]);
            }
            double sum = 0.0;
            for (std::size_t key = 0; key < sequence; ++key) {
                probabilities[key] = static_cast<float>(std::exp(static_cast<double>(scores[key] - max_score)));
                sum += probabilities[key];
            }
            if (!(sum > 0.0) || !std::isfinite(sum)) throw std::runtime_error("vision attention softmax failed");
            for (std::size_t key = 0; key < sequence; ++key) probabilities[key] /= static_cast<float>(sum);

            for (std::size_t d = 0; d < head_dim_; ++d) {
                float value = 0.0F;
                for (std::size_t key = 0; key < sequence; ++key) {
                    value += probabilities[key] * v.data()[key * model_dim_ + head_offset + d];
                }
                context.data()[query * model_dim_ + head_offset + d] = value;
            }
        }
    }
    return out_proj_.forward(context);
}

std::vector<nn::Parameter*> VisionSelfAttention::parameters() {
    std::vector<nn::Parameter*> out;
    append_parameters(out, q_proj_.parameters());
    append_parameters(out, k_proj_.parameters());
    append_parameters(out, v_proj_.parameters());
    append_parameters(out, out_proj_.parameters());
    return out;
}

std::vector<const nn::Parameter*> VisionSelfAttention::parameters() const {
    std::vector<const nn::Parameter*> out;
    append_parameters(out, q_proj_.parameters());
    append_parameters(out, k_proj_.parameters());
    append_parameters(out, v_proj_.parameters());
    append_parameters(out, out_proj_.parameters());
    return out;
}

VisionBlock::VisionBlock(
    std::size_t model_dim,
    std::size_t num_heads,
    std::size_t ffn_hidden_dim,
    Random& rng,
    float norm_epsilon)
    : attention_norm_(model_dim, norm_epsilon),
      attention_(model_dim, num_heads, rng, false),
      feed_forward_norm_(model_dim, norm_epsilon),
      feed_forward_(model_dim, ffn_hidden_dim, rng, false) {}

Tensor VisionBlock::forward(const Tensor& input) const {
    const Tensor attention_out = attention_.forward(attention_norm_.forward(input));
    const Tensor residual = input.add(attention_out);
    const Tensor feed_forward_out = feed_forward_.forward(feed_forward_norm_.forward(residual));
    return residual.add(feed_forward_out);
}

std::vector<nn::Parameter*> VisionBlock::parameters() {
    std::vector<nn::Parameter*> out;
    append_parameters(out, attention_norm_.parameters());
    append_parameters(out, attention_.parameters());
    append_parameters(out, feed_forward_norm_.parameters());
    append_parameters(out, feed_forward_.parameters());
    return out;
}

std::vector<const nn::Parameter*> VisionBlock::parameters() const {
    std::vector<const nn::Parameter*> out;
    append_parameters(out, attention_norm_.parameters());
    append_parameters(out, attention_.parameters());
    append_parameters(out, feed_forward_norm_.parameters());
    append_parameters(out, feed_forward_.parameters());
    return out;
}

VisionEncoder::VisionEncoder(VisionConfig config, Random& rng)
    : config_(config),
      patch_projection_(config.patch_size * config.patch_size * 3, config.model_dim, rng, true),
      final_norm_(config.model_dim, config.norm_epsilon),
      output_projection_(config.model_dim, config.embedding_dim, rng, false) {
    if (config_.patch_size == 0 || config_.model_dim == 0 || config_.num_heads == 0 ||
        config_.num_layers == 0 || config_.ffn_hidden_dim == 0 || config_.embedding_dim == 0 ||
        config_.model_dim % config_.num_heads != 0) {
        throw std::invalid_argument("invalid VisionConfig");
    }
    blocks_.reserve(config_.num_layers);
    for (std::size_t i = 0; i < config_.num_layers; ++i) {
        blocks_.push_back(std::make_unique<VisionBlock>(
            config_.model_dim, config_.num_heads, config_.ffn_hidden_dim, rng, config_.norm_epsilon));
    }
}

Tensor VisionEncoder::encode_tokens(const RgbImage& image) const {
    if (image.width() % config_.patch_size != 0 || image.height() % config_.patch_size != 0) {
        throw std::invalid_argument("image dimensions must be divisible by vision patch size");
    }
    const Tensor image_tensor = image_to_tensor(image);
    const Tensor patches = patchify(image_tensor, config_.patch_size);
    Tensor hidden = patch_projection_.forward(patches);
    add_2d_sincos_position(hidden, image.height() / config_.patch_size, image.width() / config_.patch_size);
    for (const auto& block : blocks_) hidden = block->forward(hidden);
    hidden = final_norm_.forward(hidden);
    return output_projection_.forward(hidden);
}

Tensor VisionEncoder::encode_pooled(const RgbImage& image) const {
    const Tensor tokens = encode_tokens(image);
    Tensor pooled({tokens.shape()[1]});
    for (std::size_t row = 0; row < tokens.shape()[0]; ++row) {
        for (std::size_t col = 0; col < tokens.shape()[1]; ++col) {
            pooled.data()[col] += tokens.data()[row * tokens.shape()[1] + col];
        }
    }
    const float inverse = 1.0F / static_cast<float>(tokens.shape()[0]);
    for (auto& value : pooled.data()) value *= inverse;
    return pooled;
}

std::vector<nn::Parameter*> VisionEncoder::parameters() {
    std::vector<nn::Parameter*> out;
    append_parameters(out, patch_projection_.parameters());
    for (auto& block : blocks_) append_parameters(out, block->parameters());
    append_parameters(out, final_norm_.parameters());
    append_parameters(out, output_projection_.parameters());
    return out;
}

std::vector<const nn::Parameter*> VisionEncoder::parameters() const {
    std::vector<const nn::Parameter*> out;
    append_parameters(out, patch_projection_.parameters());
    for (const auto& block : blocks_) append_parameters(out, block->parameters());
    append_parameters(out, final_norm_.parameters());
    append_parameters(out, output_projection_.parameters());
    return out;
}

CrossModalProjector::CrossModalProjector(
    std::size_t text_dim,
    std::size_t vision_dim,
    std::size_t shared_dim,
    Random& rng)
    : text_projection_(text_dim, shared_dim, rng, false),
      vision_projection_(vision_dim, shared_dim, rng, false) {
    if (text_dim == 0 || vision_dim == 0 || shared_dim == 0) throw std::invalid_argument("cross-modal dimensions must be non-zero");
}

Tensor CrossModalProjector::project_text(const Tensor& text_features) const {
    return text_projection_.forward(text_features);
}

Tensor CrossModalProjector::project_vision(const Tensor& vision_features) const {
    return vision_projection_.forward(vision_features);
}

std::vector<nn::Parameter*> CrossModalProjector::parameters() {
    std::vector<nn::Parameter*> out;
    append_parameters(out, text_projection_.parameters());
    append_parameters(out, vision_projection_.parameters());
    return out;
}

std::vector<const nn::Parameter*> CrossModalProjector::parameters() const {
    std::vector<const nn::Parameter*> out;
    append_parameters(out, text_projection_.parameters());
    append_parameters(out, vision_projection_.parameters());
    return out;
}

LatentRasterDecoder::LatentRasterDecoder(LatentDecoderConfig config, Random& rng)
    : config_(config),
      latent_to_patch_(config.latent_channels, config.patch_size * config.patch_size * 3, rng, true) {
    if (config_.latent_channels == 0 || config_.patch_size == 0) throw std::invalid_argument("invalid latent decoder config");
}

RgbImage LatentRasterDecoder::decode(const Tensor& latent_grid) const {
    if (latent_grid.rank() != 3 || latent_grid.shape()[0] == 0 || latent_grid.shape()[1] == 0 ||
        latent_grid.shape()[2] != config_.latent_channels) {
        throw std::invalid_argument("latent decoder requires [grid_h,grid_w,latent_channels]");
    }
    const std::size_t grid_h = latent_grid.shape()[0];
    const std::size_t grid_w = latent_grid.shape()[1];
    const std::size_t cells = grid_h * grid_w;
    Tensor flattened({cells, config_.latent_channels}, latent_grid.data());
    const Tensor patches = latent_to_patch_.forward(flattened);
    RgbImage image(grid_w * config_.patch_size, grid_h * config_.patch_size);
    const std::size_t patch_dim = config_.patch_size * config_.patch_size * 3;
    for (std::size_t gy = 0; gy < grid_h; ++gy) {
        for (std::size_t gx = 0; gx < grid_w; ++gx) {
            const std::size_t cell = gy * grid_w + gx;
            std::size_t feature = 0;
            for (std::size_t py = 0; py < config_.patch_size; ++py) {
                for (std::size_t px = 0; px < config_.patch_size; ++px) {
                    for (std::size_t channel = 0; channel < 3; ++channel) {
                        const float raw = patches.data()[cell * patch_dim + feature++];
                        const float normalized = 0.5F + 0.5F * std::tanh(raw);
                        image.at(gx * config_.patch_size + px, gy * config_.patch_size + py, channel) =
                            static_cast<std::uint8_t>(std::lround(std::clamp(normalized, 0.0F, 1.0F) * 255.0F));
                    }
                }
            }
        }
    }
    return image;
}

Tensor LatentRasterDecoder::sample_latent(std::size_t grid_height, std::size_t grid_width, Random& rng) const {
    if (grid_height == 0 || grid_width == 0) throw std::invalid_argument("latent grid dimensions must be non-zero");
    Tensor latent({grid_height, grid_width, config_.latent_channels});
    rng.fill_normal(latent, 0.0F, 1.0F);
    return latent;
}

std::vector<nn::Parameter*> LatentRasterDecoder::parameters() {
    return latent_to_patch_.parameters();
}

std::vector<const nn::Parameter*> LatentRasterDecoder::parameters() const {
    return latent_to_patch_.parameters();
}

} // namespace spiral::vision
