#pragma once

#include "spiral/model.hpp"
#include "spiral/nn.hpp"
#include "spiral/random.hpp"
#include "spiral/tensor.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace spiral::vision {

class RgbImage {
public:
    RgbImage() = default;
    RgbImage(std::size_t width, std::size_t height, std::uint8_t fill = 0);
    RgbImage(std::size_t width, std::size_t height, std::vector<std::uint8_t> pixels);

    [[nodiscard]] std::size_t width() const noexcept { return width_; }
    [[nodiscard]] std::size_t height() const noexcept { return height_; }
    [[nodiscard]] const std::vector<std::uint8_t>& pixels() const noexcept { return pixels_; }
    [[nodiscard]] std::vector<std::uint8_t>& pixels() noexcept { return pixels_; }
    [[nodiscard]] std::uint8_t at(std::size_t x, std::size_t y, std::size_t channel) const;
    std::uint8_t& at(std::size_t x, std::size_t y, std::size_t channel);

    void save_ppm(const std::string& path) const;
    [[nodiscard]] static RgbImage load_ppm(const std::string& path);

private:
    [[nodiscard]] std::size_t offset(std::size_t x, std::size_t y, std::size_t channel) const;
    std::size_t width_ = 0;
    std::size_t height_ = 0;
    std::vector<std::uint8_t> pixels_;
};

[[nodiscard]] Tensor image_to_tensor(const RgbImage& image);
[[nodiscard]] RgbImage tensor_to_image(const Tensor& tensor);
[[nodiscard]] Tensor patchify(const Tensor& image, std::size_t patch_size);
void add_2d_sincos_position(Tensor& tokens, std::size_t grid_height, std::size_t grid_width);

inline void append_parameters(
    std::vector<const nn::Parameter*>& out,
    std::vector<nn::Parameter*> values) {
    out.reserve(out.size() + values.size());
    for (auto* value : values) out.push_back(value);
}

class VisionSelfAttention final : public nn::Module {
public:
    VisionSelfAttention(std::size_t model_dim, std::size_t num_heads, Random& rng, bool use_bias = false);

    [[nodiscard]] Tensor forward(const Tensor& input) const override;
    [[nodiscard]] std::vector<nn::Parameter*> parameters() override;
    [[nodiscard]] std::vector<const nn::Parameter*> parameters() const override;

private:
    std::size_t model_dim_;
    std::size_t num_heads_;
    std::size_t head_dim_;
    nn::Linear q_proj_;
    nn::Linear k_proj_;
    nn::Linear v_proj_;
    nn::Linear out_proj_;
};

class VisionBlock final : public nn::Module {
public:
    VisionBlock(
        std::size_t model_dim,
        std::size_t num_heads,
        std::size_t ffn_hidden_dim,
        Random& rng,
        float norm_epsilon = 1.0e-5F);

    [[nodiscard]] Tensor forward(const Tensor& input) const override;
    [[nodiscard]] std::vector<nn::Parameter*> parameters() override;
    [[nodiscard]] std::vector<const nn::Parameter*> parameters() const override;

private:
    nn::RMSNorm attention_norm_;
    VisionSelfAttention attention_;
    nn::RMSNorm feed_forward_norm_;
    nn::GatedFeedForward feed_forward_;
};

struct VisionConfig {
    std::size_t patch_size = 4;
    std::size_t model_dim = 64;
    std::size_t num_heads = 4;
    std::size_t num_layers = 2;
    std::size_t ffn_hidden_dim = 128;
    std::size_t embedding_dim = 64;
    float norm_epsilon = 1.0e-5F;
};

class VisionEncoder final {
public:
    VisionEncoder(VisionConfig config, Random& rng);

    [[nodiscard]] Tensor encode_tokens(const RgbImage& image) const;
    [[nodiscard]] Tensor encode_pooled(const RgbImage& image) const;
    [[nodiscard]] std::vector<nn::Parameter*> parameters();
    [[nodiscard]] std::vector<const nn::Parameter*> parameters() const;
    [[nodiscard]] const VisionConfig& config() const noexcept { return config_; }

private:
    VisionConfig config_;
    nn::Linear patch_projection_;
    std::vector<std::unique_ptr<VisionBlock>> blocks_;
    nn::RMSNorm final_norm_;
    nn::Linear output_projection_;
};

class CrossModalProjector final {
public:
    CrossModalProjector(std::size_t text_dim, std::size_t vision_dim, std::size_t shared_dim, Random& rng);

    [[nodiscard]] Tensor project_text(const Tensor& text_features) const;
    [[nodiscard]] Tensor project_vision(const Tensor& vision_features) const;
    [[nodiscard]] std::vector<nn::Parameter*> parameters();
    [[nodiscard]] std::vector<const nn::Parameter*> parameters() const;

private:
    nn::Linear text_projection_;
    nn::Linear vision_projection_;
};

struct LatentDecoderConfig {
    std::size_t latent_channels = 8;
    std::size_t patch_size = 4;
};

class LatentRasterDecoder final {
public:
    LatentRasterDecoder(LatentDecoderConfig config, Random& rng);

    [[nodiscard]] RgbImage decode(const Tensor& latent_grid) const;
    [[nodiscard]] Tensor sample_latent(std::size_t grid_height, std::size_t grid_width, Random& rng) const;
    [[nodiscard]] std::vector<nn::Parameter*> parameters();
    [[nodiscard]] std::vector<const nn::Parameter*> parameters() const;

private:
    LatentDecoderConfig config_;
    nn::Linear latent_to_patch_;
};

} // namespace spiral::vision
