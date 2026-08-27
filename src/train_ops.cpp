#include "train_detail.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace spiral::train::detail {

void require_same_shape(const Tensor& a, const Tensor& b, const char* what) {
    if (a.shape() != b.shape()) {
        throw std::invalid_argument(std::string(what) + " shape mismatch");
    }
}

void add_inplace(Tensor& destination, const Tensor& source) {
    require_same_shape(destination, source, "tensor add_inplace");
    for (std::size_t i = 0; i < destination.numel(); ++i) {
        destination.data()[i] += source.data()[i];
    }
}

void accumulate_grad(nn::Parameter& parameter, const Tensor& gradient) {
    if (!parameter.trainable) return;
    require_same_shape(parameter.value, gradient, "parameter gradient");
    parameter.ensure_grad();
    add_inplace(parameter.grad, gradient);
}

Tensor linear_backward(nn::Linear& layer, const Tensor& input, const Tensor& grad_output) {
    if (input.rank() != 2 || grad_output.rank() != 2) {
        throw std::invalid_argument("linear_backward expects rank-2 tensors");
    }
    const auto rows = input.shape()[0];
    const auto in_features = layer.in_features();
    const auto out_features = layer.out_features();
    if (input.shape()[1] != in_features || grad_output.shape()[0] != rows
        || grad_output.shape()[1] != out_features) {
        throw std::invalid_argument("linear_backward dimension mismatch");
    }

    Tensor grad_input({rows, in_features});
    Tensor grad_weight({in_features, out_features});
    Tensor grad_bias({out_features});
    const auto& weight = layer.weight().value.data();

    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t out = 0; out < out_features; ++out) {
            const float go = grad_output.data()[row * out_features + out];
            if (layer.uses_bias()) grad_bias.data()[out] += go;
            for (std::size_t in = 0; in < in_features; ++in) {
                grad_weight.data()[in * out_features + out] += input.data()[row * in_features + in] * go;
                grad_input.data()[row * in_features + in] += weight[in * out_features + out] * go;
            }
        }
    }

    accumulate_grad(layer.weight(), grad_weight);
    if (layer.uses_bias()) accumulate_grad(layer.bias(), grad_bias);
    return grad_input;
}

Tensor rmsnorm_backward(nn::RMSNorm& norm, const Tensor& input, const Tensor& grad_output) {
    require_same_shape(input, grad_output, "RMSNorm backward");
    const std::size_t width = norm.feature_size();
    if (input.rank() == 0 || input.shape().back() != width) {
        throw std::invalid_argument("RMSNorm backward feature mismatch");
    }

    const std::size_t rows = input.numel() / width;
    Tensor grad_input(input.shape());
    Tensor grad_scale({width});
    const auto& scale = norm.scale().value.data();

    for (std::size_t row = 0; row < rows; ++row) {
        const std::size_t base = row * width;
        float mean_square = 0.0F;
        for (std::size_t col = 0; col < width; ++col) {
            const float x = input.data()[base + col];
            mean_square += x * x;
        }
        mean_square /= static_cast<float>(width);
        const float inv = 1.0F / std::sqrt(mean_square + norm.epsilon());

        float dot = 0.0F;
        for (std::size_t col = 0; col < width; ++col) {
            const float x = input.data()[base + col];
            const float go = grad_output.data()[base + col];
            dot += go * scale[col] * x;
            grad_scale.data()[col] += go * x * inv;
        }

        const float correction = (inv * inv * inv / static_cast<float>(width)) * dot;
        for (std::size_t col = 0; col < width; ++col) {
            const float x = input.data()[base + col];
            const float go = grad_output.data()[base + col];
            grad_input.data()[base + col] = go * scale[col] * inv - x * correction;
        }
    }

    accumulate_grad(norm.scale(), grad_scale);
    return grad_input;
}

float silu(float x) {
    return x / (1.0F + std::exp(-x));
}

float silu_derivative(float x) {
    const float sigmoid = 1.0F / (1.0F + std::exp(-x));
    return sigmoid * (1.0F + x * (1.0F - sigmoid));
}

void apply_rotary_inverse_to_gradient(Tensor& gradient, float base) {
    if (gradient.rank() != 3) throw std::invalid_argument("rotary gradient expects rank-3 tensor");
    const auto heads = gradient.shape()[0];
    const auto sequence = gradient.shape()[1];
    const auto dim = gradient.shape()[2];

    for (std::size_t head = 0; head < heads; ++head) {
        for (std::size_t position = 0; position < sequence; ++position) {
            for (std::size_t pair = 0; pair < dim; pair += 2) {
                const float exponent = static_cast<float>(pair) / static_cast<float>(dim);
                const float theta = static_cast<float>(position) / std::pow(base, exponent);
                const float cosine = std::cos(theta);
                const float sine = std::sin(theta);
                const std::size_t offset = (head * sequence + position) * dim + pair;
                const float gx = gradient.data()[offset];
                const float gy = gradient.data()[offset + 1];
                gradient.data()[offset] = gx * cosine + gy * sine;
                gradient.data()[offset + 1] = -gx * sine + gy * cosine;
            }
        }
    }
}

Tensor split_heads(const Tensor& projected, std::size_t heads, std::size_t head_dim) {
    const auto sequence = projected.shape()[0];
    const auto model_dim = projected.shape()[1];
    Tensor out({heads, sequence, head_dim});
    for (std::size_t token = 0; token < sequence; ++token) {
        for (std::size_t head = 0; head < heads; ++head) {
            for (std::size_t dim = 0; dim < head_dim; ++dim) {
                out.data()[(head * sequence + token) * head_dim + dim] =
                    projected.data()[token * model_dim + head * head_dim + dim];
            }
        }
    }
    return out;
}

Tensor merge_heads(const Tensor& input) {
    const auto heads = input.shape()[0];
    const auto sequence = input.shape()[1];
    const auto head_dim = input.shape()[2];
    const auto model_dim = heads * head_dim;
    Tensor out({sequence, model_dim});
    for (std::size_t token = 0; token < sequence; ++token) {
        for (std::size_t head = 0; head < heads; ++head) {
            for (std::size_t dim = 0; dim < head_dim; ++dim) {
                out.data()[token * model_dim + head * head_dim + dim] =
                    input.data()[(head * sequence + token) * head_dim + dim];
            }
        }
    }
    return out;
}

} // namespace spiral::train::detail
