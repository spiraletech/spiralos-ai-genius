#include "spiral/nn.hpp"

#include <cmath>
#include <stdexcept>

namespace spiral::nn {
namespace {

void require_last_dimension(const Tensor& input, std::size_t expected, const char* layer_name) {
    if (input.rank() == 0 || input.shape().back() != expected) {
        throw std::invalid_argument(std::string(layer_name) + " input feature dimension mismatch");
    }
}

std::size_t outer_rows(const Tensor& input, std::size_t feature_size) {
    if (feature_size == 0 || input.numel() % feature_size != 0) {
        throw std::invalid_argument("Normalization input is not divisible by feature size");
    }
    return input.numel() / feature_size;
}

} // namespace

std::vector<Parameter*> Module::parameters() { return {}; }
std::vector<const Parameter*> Module::parameters() const { return {}; }

Linear::Linear(std::size_t in_features, std::size_t out_features, Random& rng, bool use_bias)
    : in_features_(in_features),
      out_features_(out_features),
      use_bias_(use_bias),
      weight_{"weight", Tensor({in_features, out_features}), true},
      bias_{"bias", Tensor({out_features}), use_bias} {
    if (in_features == 0 || out_features == 0) {
        throw std::invalid_argument("Linear dimensions must be non-zero");
    }
    const float limit = std::sqrt(6.0F / static_cast<float>(in_features + out_features));
    rng.fill_uniform(weight_.value, -limit, limit);
}

Tensor Linear::forward(const Tensor& input) const {
    require_last_dimension(input, in_features_, "Linear");

    Tensor matrix;
    bool collapse = false;
    if (input.rank() == 1) {
        matrix = Tensor({1, in_features_}, input.data());
        collapse = true;
    } else if (input.rank() == 2) {
        matrix = input;
    } else {
        throw std::invalid_argument("Linear currently accepts rank-1 or rank-2 tensors");
    }

    Tensor out = matrix.matmul(weight_.value);
    if (use_bias_) {
        for (std::size_t row = 0; row < out.shape()[0]; ++row) {
            for (std::size_t col = 0; col < out_features_; ++col) {
                out.data()[row * out_features_ + col] += bias_.value.data()[col];
            }
        }
    }

    if (collapse) {
        return Tensor({out_features_}, out.data());
    }
    return out;
}

std::vector<Parameter*> Linear::parameters() {
    if (use_bias_) return {&weight_, &bias_};
    return {&weight_};
}

std::vector<const Parameter*> Linear::parameters() const {
    if (use_bias_) return {&weight_, &bias_};
    return {&weight_};
}

Embedding::Embedding(std::size_t vocabulary_size, std::size_t embedding_dim, Random& rng)
    : vocabulary_size_(vocabulary_size),
      embedding_dim_(embedding_dim),
      table_{"embedding", Tensor({vocabulary_size, embedding_dim}), true} {
    if (vocabulary_size == 0 || embedding_dim == 0) {
        throw std::invalid_argument("Embedding dimensions must be non-zero");
    }
    rng.fill_normal(table_.value, 0.0F, 0.02F);
}

Tensor Embedding::forward(std::span<const std::uint32_t> token_ids) const {
    Tensor out({token_ids.size(), embedding_dim_});
    for (std::size_t row = 0; row < token_ids.size(); ++row) {
        const auto token = static_cast<std::size_t>(token_ids[row]);
        if (token >= vocabulary_size_) {
            throw std::out_of_range("Embedding token id exceeds vocabulary");
        }
        const std::size_t source = token * embedding_dim_;
        const std::size_t dest = row * embedding_dim_;
        for (std::size_t col = 0; col < embedding_dim_; ++col) {
            out.data()[dest + col] = table_.value.data()[source + col];
        }
    }
    return out;
}

RMSNorm::RMSNorm(std::size_t feature_size, float epsilon)
    : feature_size_(feature_size), epsilon_(epsilon), scale_{"scale", Tensor::ones({feature_size}), true} {
    if (feature_size == 0 || epsilon <= 0.0F) {
        throw std::invalid_argument("RMSNorm requires positive feature size and epsilon");
    }
}

Tensor RMSNorm::forward(const Tensor& input) const {
    require_last_dimension(input, feature_size_, "RMSNorm");
    const std::size_t rows = outer_rows(input, feature_size_);
    Tensor out(input.shape());

    for (std::size_t row = 0; row < rows; ++row) {
        const std::size_t base = row * feature_size_;
        float mean_square = 0.0F;
        for (std::size_t col = 0; col < feature_size_; ++col) {
            const float value = input.data()[base + col];
            mean_square += value * value;
        }
        mean_square /= static_cast<float>(feature_size_);
        const float inv_rms = 1.0F / std::sqrt(mean_square + epsilon_);
        for (std::size_t col = 0; col < feature_size_; ++col) {
            out.data()[base + col] = input.data()[base + col] * inv_rms * scale_.value.data()[col];
        }
    }
    return out;
}

std::vector<Parameter*> RMSNorm::parameters() { return {&scale_}; }
std::vector<const Parameter*> RMSNorm::parameters() const { return {&scale_}; }

LayerNorm::LayerNorm(std::size_t feature_size, float epsilon)
    : feature_size_(feature_size), epsilon_(epsilon),
      scale_{"scale", Tensor::ones({feature_size}), true},
      bias_{"bias", Tensor::zeros({feature_size}), true} {
    if (feature_size == 0 || epsilon <= 0.0F) {
        throw std::invalid_argument("LayerNorm requires positive feature size and epsilon");
    }
}

Tensor LayerNorm::forward(const Tensor& input) const {
    require_last_dimension(input, feature_size_, "LayerNorm");
    const std::size_t rows = outer_rows(input, feature_size_);
    Tensor out(input.shape());

    for (std::size_t row = 0; row < rows; ++row) {
        const std::size_t base = row * feature_size_;
        float mean = 0.0F;
        for (std::size_t col = 0; col < feature_size_; ++col) mean += input.data()[base + col];
        mean /= static_cast<float>(feature_size_);

        float variance = 0.0F;
        for (std::size_t col = 0; col < feature_size_; ++col) {
            const float centered = input.data()[base + col] - mean;
            variance += centered * centered;
        }
        variance /= static_cast<float>(feature_size_);
        const float inv_std = 1.0F / std::sqrt(variance + epsilon_);

        for (std::size_t col = 0; col < feature_size_; ++col) {
            const float normalized = (input.data()[base + col] - mean) * inv_std;
            out.data()[base + col] = normalized * scale_.value.data()[col] + bias_.value.data()[col];
        }
    }
    return out;
}

std::vector<Parameter*> LayerNorm::parameters() { return {&scale_, &bias_}; }
std::vector<const Parameter*> LayerNorm::parameters() const { return {&scale_, &bias_}; }

void Sequential::add(std::unique_ptr<Module> module) {
    if (!module) throw std::invalid_argument("Sequential cannot add null module");
    modules_.push_back(std::move(module));
}

Tensor Sequential::forward(const Tensor& input) const {
    Tensor value = input;
    for (const auto& module : modules_) value = module->forward(value);
    return value;
}

std::vector<Parameter*> Sequential::parameters() {
    std::vector<Parameter*> out;
    for (auto& module : modules_) {
        auto child = module->parameters();
        out.insert(out.end(), child.begin(), child.end());
    }
    return out;
}

std::vector<const Parameter*> Sequential::parameters() const {
    std::vector<const Parameter*> out;
    for (const auto& module : modules_) {
        auto child = module->parameters();
        out.insert(out.end(), child.begin(), child.end());
    }
    return out;
}

} // namespace spiral::nn
