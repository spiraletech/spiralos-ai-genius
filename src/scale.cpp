#include "spiral/scale.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

namespace spiral::scale {
namespace {

void append_parameters(std::vector<nn::Parameter*>& out, std::vector<nn::Parameter*> values) {
    out.insert(out.end(), values.begin(), values.end());
}

void append_parameters(std::vector<const nn::Parameter*>& out, std::vector<const nn::Parameter*> values) {
    out.insert(out.end(), values.begin(), values.end());
}

Tensor add_scaled(const Tensor& lhs, const Tensor& rhs, float scale) {
    if (lhs.shape() != rhs.shape()) throw std::invalid_argument("scaled residual shape mismatch");
    Tensor out(lhs.shape());
    for (std::size_t i = 0; i < lhs.numel(); ++i) out.data()[i] = lhs.data()[i] + scale * rhs.data()[i];
    return out;
}

void add_inplace(Tensor& lhs, const Tensor& rhs) {
    if (lhs.shape() != rhs.shape()) throw std::invalid_argument("gradient add shape mismatch");
    for (std::size_t i = 0; i < lhs.numel(); ++i) lhs.data()[i] += rhs.data()[i];
}

void scale_inplace(Tensor& tensor, float scale) {
    for (float& value : tensor.data()) value *= scale;
}

Tensor silu(const Tensor& input) {
    Tensor out(input.shape());
    for (std::size_t i = 0; i < input.numel(); ++i) {
        const float x = input.data()[i];
        const float s = 1.0F / (1.0F + std::exp(-x));
        out.data()[i] = x * s;
    }
    return out;
}

Tensor silu_backward(const Tensor& input, const Tensor& grad_output) {
    if (input.shape() != grad_output.shape()) throw std::invalid_argument("SiLU backward shape mismatch");
    Tensor grad(input.shape());
    for (std::size_t i = 0; i < input.numel(); ++i) {
        const float x = input.data()[i];
        const float s = 1.0F / (1.0F + std::exp(-x));
        grad.data()[i] = grad_output.data()[i] * s * (1.0F + x * (1.0F - s));
    }
    return grad;
}

Tensor mse_gradient(const Tensor& prediction, const Tensor& target) {
    if (prediction.shape() != target.shape() || prediction.numel() == 0) {
        throw std::invalid_argument("MSE gradient requires equal non-empty tensors");
    }
    Tensor grad(prediction.shape());
    const float factor = 2.0F / static_cast<float>(prediction.numel());
    for (std::size_t i = 0; i < prediction.numel(); ++i) {
        grad.data()[i] = factor * (prediction.data()[i] - target.data()[i]);
    }
    return grad;
}

Tensor linear_backward(nn::Linear& layer, const Tensor& input, const Tensor& grad_output) {
    if (input.rank() != 2 || grad_output.rank() != 2) throw std::invalid_argument("linear backward requires rank-2 tensors");
    const std::size_t rows = input.shape()[0];
    const std::size_t in_features = input.shape()[1];
    const std::size_t out_features = grad_output.shape()[1];
    if (rows != grad_output.shape()[0] || in_features != layer.in_features() || out_features != layer.out_features()) {
        throw std::invalid_argument("linear backward shape mismatch");
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
            const float g = grad_output.data()[row * out_features + out];
            if (bias != nullptr) bias->grad.data()[out] += g;
            for (std::size_t in = 0; in < in_features; ++in) {
                weight.grad.data()[in * out_features + out] += input.data()[row * in_features + in] * g;
                grad_input.data()[row * in_features + in] += weight.value.data()[in * out_features + out] * g;
            }
        }
    }
    return grad_input;
}

Tensor rmsnorm_backward(nn::RMSNorm& norm, const Tensor& input, const Tensor& grad_output) {
    if (input.shape() != grad_output.shape() || input.rank() == 0 || input.shape().back() != norm.feature_size()) {
        throw std::invalid_argument("RMSNorm backward shape mismatch");
    }
    auto& scale = norm.scale();
    scale.ensure_grad();
    const std::size_t features = norm.feature_size();
    const std::size_t rows = input.numel() / features;
    Tensor grad_input(input.shape());

    for (std::size_t row = 0; row < rows; ++row) {
        const std::size_t base = row * features;
        float mean_square = 0.0F;
        for (std::size_t d = 0; d < features; ++d) {
            const float x = input.data()[base + d];
            mean_square += x * x;
        }
        mean_square /= static_cast<float>(features);
        const float inv_rms = 1.0F / std::sqrt(mean_square + norm.epsilon());
        float weighted_dot = 0.0F;
        for (std::size_t d = 0; d < features; ++d) {
            const float x = input.data()[base + d];
            const float g = grad_output.data()[base + d];
            const float weight = scale.value.data()[d];
            scale.grad.data()[d] += g * x * inv_rms;
            weighted_dot += g * weight * x;
        }
        const float correction = inv_rms * inv_rms * inv_rms * weighted_dot / static_cast<float>(features);
        for (std::size_t d = 0; d < features; ++d) {
            const float x = input.data()[base + d];
            const float g = grad_output.data()[base + d];
            const float weight = scale.value.data()[d];
            grad_input.data()[base + d] = g * weight * inv_rms - x * correction;
        }
    }
    return grad_input;
}

struct AttentionCache {
    Tensor query_input;
    Tensor context_input;
    Tensor q;
    Tensor k;
    Tensor v;
    Tensor weights;
    Tensor merged;
};

Tensor attention_forward(
    latent::MultiHeadAttention& attention,
    const Tensor& query,
    const Tensor& context,
    AttentionCache& cache) {
    if (query.rank() != 2 || context.rank() != 2 ||
        query.shape()[1] != attention.model_dim() || context.shape()[1] != attention.model_dim() ||
        query.shape()[0] == 0 || context.shape()[0] == 0) {
        throw std::invalid_argument("scale attention requires non-empty [sequence,model_dim]");
    }
    cache.query_input = query;
    cache.context_input = context;
    cache.q = attention.q_proj().forward(query);
    cache.k = attention.k_proj().forward(context);
    cache.v = attention.v_proj().forward(context);

    const std::size_t qseq = query.shape()[0];
    const std::size_t kseq = context.shape()[0];
    const std::size_t model_dim = attention.model_dim();
    const std::size_t heads = attention.num_heads();
    const std::size_t head_dim = attention.head_dim();
    const float attention_scale = 1.0F / std::sqrt(static_cast<float>(head_dim));
    cache.weights = Tensor({heads, qseq, kseq});
    cache.merged = Tensor({qseq, model_dim});
    std::vector<float> scores(kseq);

    for (std::size_t head = 0; head < heads; ++head) {
        const std::size_t offset = head * head_dim;
        for (std::size_t qi = 0; qi < qseq; ++qi) {
            float maximum = -std::numeric_limits<float>::infinity();
            for (std::size_t ki = 0; ki < kseq; ++ki) {
                float dot = 0.0F;
                for (std::size_t d = 0; d < head_dim; ++d) {
                    dot += cache.q.data()[qi * model_dim + offset + d] * cache.k.data()[ki * model_dim + offset + d];
                }
                scores[ki] = dot * attention_scale;
                maximum = std::max(maximum, scores[ki]);
            }
            float sum = 0.0F;
            const std::size_t base = (head * qseq + qi) * kseq;
            for (std::size_t ki = 0; ki < kseq; ++ki) {
                const float value = std::exp(scores[ki] - maximum);
                cache.weights.data()[base + ki] = value;
                sum += value;
            }
            if (!(sum > 0.0F) || !std::isfinite(sum)) throw std::runtime_error("scale attention softmax failed");
            for (std::size_t ki = 0; ki < kseq; ++ki) cache.weights.data()[base + ki] /= sum;
            for (std::size_t d = 0; d < head_dim; ++d) {
                float value = 0.0F;
                for (std::size_t ki = 0; ki < kseq; ++ki) {
                    value += cache.weights.data()[base + ki] * cache.v.data()[ki * model_dim + offset + d];
                }
                cache.merged.data()[qi * model_dim + offset + d] = value;
            }
        }
    }
    return attention.out_proj().forward(cache.merged);
}

std::pair<Tensor, Tensor> attention_backward(
    latent::MultiHeadAttention& attention,
    const AttentionCache& cache,
    const Tensor& grad_output) {
    const std::size_t qseq = cache.query_input.shape()[0];
    const std::size_t kseq = cache.context_input.shape()[0];
    const std::size_t model_dim = attention.model_dim();
    const std::size_t heads = attention.num_heads();
    const std::size_t head_dim = attention.head_dim();
    const float attention_scale = 1.0F / std::sqrt(static_cast<float>(head_dim));

    const Tensor grad_merged = linear_backward(attention.out_proj(), cache.merged, grad_output);
    Tensor grad_weights({heads, qseq, kseq});
    Tensor grad_v({kseq, model_dim});
    for (std::size_t head = 0; head < heads; ++head) {
        const std::size_t offset = head * head_dim;
        for (std::size_t qi = 0; qi < qseq; ++qi) {
            const std::size_t base = (head * qseq + qi) * kseq;
            for (std::size_t ki = 0; ki < kseq; ++ki) {
                float dot = 0.0F;
                for (std::size_t d = 0; d < head_dim; ++d) {
                    const float g = grad_merged.data()[qi * model_dim + offset + d];
                    dot += g * cache.v.data()[ki * model_dim + offset + d];
                    grad_v.data()[ki * model_dim + offset + d] += cache.weights.data()[base + ki] * g;
                }
                grad_weights.data()[base + ki] = dot;
            }
        }
    }

    Tensor grad_scores({heads, qseq, kseq});
    for (std::size_t head = 0; head < heads; ++head) {
        for (std::size_t qi = 0; qi < qseq; ++qi) {
            const std::size_t base = (head * qseq + qi) * kseq;
            float weighted = 0.0F;
            for (std::size_t ki = 0; ki < kseq; ++ki) weighted += grad_weights.data()[base + ki] * cache.weights.data()[base + ki];
            for (std::size_t ki = 0; ki < kseq; ++ki) {
                const float w = cache.weights.data()[base + ki];
                grad_scores.data()[base + ki] = w * (grad_weights.data()[base + ki] - weighted);
            }
        }
    }

    Tensor grad_q({qseq, model_dim});
    Tensor grad_k({kseq, model_dim});
    for (std::size_t head = 0; head < heads; ++head) {
        const std::size_t offset = head * head_dim;
        for (std::size_t qi = 0; qi < qseq; ++qi) {
            const std::size_t base = (head * qseq + qi) * kseq;
            for (std::size_t ki = 0; ki < kseq; ++ki) {
                const float gs = grad_scores.data()[base + ki] * attention_scale;
                for (std::size_t d = 0; d < head_dim; ++d) {
                    grad_q.data()[qi * model_dim + offset + d] += gs * cache.k.data()[ki * model_dim + offset + d];
                    grad_k.data()[ki * model_dim + offset + d] += gs * cache.q.data()[qi * model_dim + offset + d];
                }
            }
        }
    }
    Tensor grad_query = linear_backward(attention.q_proj(), cache.query_input, grad_q);
    Tensor grad_context = linear_backward(attention.k_proj(), cache.context_input, grad_k);
    add_inplace(grad_context, linear_backward(attention.v_proj(), cache.context_input, grad_v));
    return {std::move(grad_query), std::move(grad_context)};
}

struct BlockCache {
    Tensor input;
    Tensor self_normed;
    AttentionCache self_attention;
    Tensor after_self;
    Tensor cross_normed;
    AttentionCache cross_attention;
    Tensor after_cross;
    Tensor ffn_normed;
    Tensor ffn_pre;
    Tensor ffn_active;
};

Tensor block_forward(
    StableLatentTransformerBlock& block,
    const Tensor& input,
    const Tensor& prompt,
    BlockCache& cache) {
    cache.input = input;
    cache.self_normed = block.self_norm().forward(input);
    const Tensor self = attention_forward(block.self_attention(), cache.self_normed, cache.self_normed, cache.self_attention);
    cache.after_self = add_scaled(input, self, block.residual_scale());

    cache.cross_normed = block.cross_norm().forward(cache.after_self);
    const Tensor cross = attention_forward(block.cross_attention(), cache.cross_normed, prompt, cache.cross_attention);
    cache.after_cross = add_scaled(cache.after_self, cross, block.residual_scale());

    cache.ffn_normed = block.ffn_norm().forward(cache.after_cross);
    cache.ffn_pre = block.ffn_in().forward(cache.ffn_normed);
    cache.ffn_active = silu(cache.ffn_pre);
    return add_scaled(cache.after_cross, block.ffn_out().forward(cache.ffn_active), block.residual_scale());
}

std::pair<Tensor, Tensor> block_backward(
    StableLatentTransformerBlock& block,
    const BlockCache& cache,
    const Tensor& grad_output) {
    Tensor grad_after_cross = grad_output;
    Tensor grad_ffn_branch = grad_output;
    scale_inplace(grad_ffn_branch, block.residual_scale());
    const Tensor grad_ffn_active = linear_backward(block.ffn_out(), cache.ffn_active, grad_ffn_branch);
    const Tensor grad_ffn_pre = silu_backward(cache.ffn_pre, grad_ffn_active);
    const Tensor grad_ffn_normed = linear_backward(block.ffn_in(), cache.ffn_normed, grad_ffn_pre);
    add_inplace(grad_after_cross, rmsnorm_backward(block.ffn_norm(), cache.after_cross, grad_ffn_normed));

    Tensor grad_cross_branch = grad_after_cross;
    scale_inplace(grad_cross_branch, block.residual_scale());
    auto [grad_cross_query_normed, grad_prompt] = attention_backward(block.cross_attention(), cache.cross_attention, grad_cross_branch);
    Tensor grad_after_self = grad_after_cross;
    add_inplace(grad_after_self, rmsnorm_backward(block.cross_norm(), cache.after_self, grad_cross_query_normed));

    Tensor grad_self_branch = grad_after_self;
    scale_inplace(grad_self_branch, block.residual_scale());
    auto [grad_self_query_normed, grad_self_context_normed] = attention_backward(block.self_attention(), cache.self_attention, grad_self_branch);
    add_inplace(grad_self_query_normed, grad_self_context_normed);
    Tensor grad_input = grad_after_self;
    add_inplace(grad_input, rmsnorm_backward(block.self_norm(), cache.input, grad_self_query_normed));
    return {std::move(grad_input), std::move(grad_prompt)};
}

struct ModelCache {
    Tensor raw_latent_conditioning;
    Tensor raw_prompt_tokens;
    Tensor prompt_tokens;
    std::vector<BlockCache> blocks;
    Tensor pre_final_norm;
    Tensor final_tokens;
};

Tensor build_latent_conditioning(
    const StableLatentTransformerConfig& config,
    const Tensor& noisy_latent,
    float time,
    std::size_t grid_height,
    std::size_t grid_width) {
    if (grid_height == 0 || grid_width == 0 || noisy_latent.rank() != 2 ||
        noisy_latent.shape()[0] != grid_height * grid_width || noisy_latent.shape()[1] != config.latent_dim) {
        throw std::invalid_argument("stable latent transformer noisy latent shape mismatch");
    }
    const Tensor time_features = flow::timestep_features(time, config.time_feature_dim);
    const std::size_t width = config.latent_dim + 2 + config.time_feature_dim;
    Tensor matrix({grid_height * grid_width, width});
    for (std::size_t y = 0; y < grid_height; ++y) {
        for (std::size_t x = 0; x < grid_width; ++x) {
            const std::size_t row = y * grid_width + x;
            const std::size_t base = row * width;
            for (std::size_t d = 0; d < config.latent_dim; ++d) matrix.data()[base + d] = noisy_latent.data()[row * config.latent_dim + d];
            matrix.data()[base + config.latent_dim] = grid_width == 1 ? 0.0F : 2.0F * static_cast<float>(x) / static_cast<float>(grid_width - 1) - 1.0F;
            matrix.data()[base + config.latent_dim + 1] = grid_height == 1 ? 0.0F : 2.0F * static_cast<float>(y) / static_cast<float>(grid_height - 1) - 1.0F;
            for (std::size_t d = 0; d < config.time_feature_dim; ++d) matrix.data()[base + config.latent_dim + 2 + d] = time_features.data()[d];
        }
    }
    return matrix;
}

Tensor model_forward_cached(
    StableLatentTransformerDenoiser& model,
    const Tensor& noisy_latent,
    std::string_view prompt,
    float time,
    std::size_t grid_height,
    std::size_t grid_width,
    ModelCache& cache) {
    const auto& config = model.config();
    cache.raw_latent_conditioning = build_latent_conditioning(config, noisy_latent, time, grid_height, grid_width);
    cache.raw_prompt_tokens = latent::prompt_token_features(prompt, config.prompt_tokens, config.text_feature_dim);
    Tensor tokens = model.input_projection().forward(cache.raw_latent_conditioning);
    cache.prompt_tokens = model.prompt_projection().forward(cache.raw_prompt_tokens);
    cache.blocks.clear();
    cache.blocks.resize(config.num_layers);
    for (std::size_t layer = 0; layer < config.num_layers; ++layer) {
        tokens = block_forward(model.block(layer), tokens, cache.prompt_tokens, cache.blocks[layer]);
    }
    cache.pre_final_norm = tokens;
    cache.final_tokens = model.final_norm().forward(tokens);
    return model.output_projection().forward(cache.final_tokens);
}

void model_backward(
    StableLatentTransformerDenoiser& model,
    const ModelCache& cache,
    const Tensor& grad_output) {
    Tensor grad_final = linear_backward(model.output_projection(), cache.final_tokens, grad_output);
    Tensor grad_tokens = rmsnorm_backward(model.final_norm(), cache.pre_final_norm, grad_final);
    Tensor grad_prompt(cache.prompt_tokens.shape());
    for (std::size_t layer = model.config().num_layers; layer-- > 0;) {
        auto [grad_input, grad_layer_prompt] = block_backward(model.block(layer), cache.blocks[layer], grad_tokens);
        grad_tokens = std::move(grad_input);
        add_inplace(grad_prompt, grad_layer_prompt);
    }
    (void)linear_backward(model.input_projection(), cache.raw_latent_conditioning, grad_tokens);
    (void)linear_backward(model.prompt_projection(), cache.raw_prompt_tokens, grad_prompt);
}

void require_aligned_image(const multimodal::ImageAutoencoder& autoencoder, const vision::RgbImage& image) {
    const std::size_t patch = autoencoder.config().patch_size;
    if (image.width() == 0 || image.height() == 0 || image.width() % patch != 0 || image.height() % patch != 0) {
        throw std::invalid_argument("scale-training image dimensions must align to autoencoder patch size");
    }
}

float evaluate_example(
    const StableLatentTransformerDenoiser& model,
    const multimodal::ImageAutoencoder& autoencoder,
    const flow::NoiseScheduler& scheduler,
    float training_time,
    const latent::ImagePromptExample& example,
    std::uint64_t noise_seed) {
    require_aligned_image(autoencoder, example.image);
    const std::size_t patch = autoencoder.config().patch_size;
    const std::size_t grid_h = example.image.height() / patch;
    const std::size_t grid_w = example.image.width() / patch;
    const Tensor target = autoencoder.encode(example.image);
    Random rng(noise_seed);
    const Tensor noise = flow::gaussian_noise(target.shape(), rng);
    const Tensor noisy = scheduler.add_noise(target, noise, training_time);
    return multimodal::mean_squared_error(model.predict(noisy, example.prompt, training_time, grid_h, grid_w), target);
}

float backward_example(
    StableLatentTransformerDenoiser& model,
    const multimodal::ImageAutoencoder& autoencoder,
    const flow::NoiseScheduler& scheduler,
    float training_time,
    const latent::ImagePromptExample& example,
    std::uint64_t noise_seed,
    float gradient_scale) {
    require_aligned_image(autoencoder, example.image);
    const std::size_t patch = autoencoder.config().patch_size;
    const std::size_t grid_h = example.image.height() / patch;
    const std::size_t grid_w = example.image.width() / patch;
    const Tensor target = autoencoder.encode(example.image);
    Random rng(noise_seed);
    const Tensor noise = flow::gaussian_noise(target.shape(), rng);
    const Tensor noisy = scheduler.add_noise(target, noise, training_time);
    ModelCache cache;
    const Tensor prediction = model_forward_cached(model, noisy, example.prompt, training_time, grid_h, grid_w, cache);
    const float loss = multimodal::mean_squared_error(prediction, target);
    Tensor grad = mse_gradient(prediction, target);
    scale_inplace(grad, gradient_scale);
    model_backward(model, cache, grad);
    return loss;
}

void write_u64(std::ostream& out, std::uint64_t value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(value));
    if (!out) throw std::runtime_error("failed while writing L12 checkpoint");
}

std::uint64_t read_u64(std::istream& in) {
    std::uint64_t value = 0;
    in.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!in) throw std::runtime_error("truncated L12 checkpoint");
    return value;
}

void write_f32(std::ostream& out, float value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(value));
    if (!out) throw std::runtime_error("failed while writing L12 checkpoint");
}

float read_f32(std::istream& in) {
    float value = 0.0F;
    in.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!in) throw std::runtime_error("truncated L12 checkpoint");
    return value;
}

void write_parameters(std::ostream& out, const std::vector<const nn::Parameter*>& parameters) {
    write_u64(out, static_cast<std::uint64_t>(parameters.size()));
    for (const auto* parameter : parameters) {
        write_u64(out, static_cast<std::uint64_t>(parameter->value.rank()));
        for (const auto dim : parameter->value.shape()) write_u64(out, static_cast<std::uint64_t>(dim));
        write_u64(out, static_cast<std::uint64_t>(parameter->value.numel()));
        out.write(reinterpret_cast<const char*>(parameter->value.data().data()),
                  static_cast<std::streamsize>(parameter->value.numel() * sizeof(float)));
        if (!out) throw std::runtime_error("failed while writing L12 model weights");
    }
}

void read_parameters(std::istream& in, const std::vector<nn::Parameter*>& parameters) {
    if (read_u64(in) != parameters.size()) throw std::runtime_error("L12 parameter count mismatch");
    for (auto* parameter : parameters) {
        if (read_u64(in) != parameter->value.rank()) throw std::runtime_error("L12 parameter rank mismatch");
        for (std::size_t d = 0; d < parameter->value.rank(); ++d) {
            if (read_u64(in) != parameter->value.shape()[d]) throw std::runtime_error("L12 parameter shape mismatch");
        }
        if (read_u64(in) != parameter->value.numel()) throw std::runtime_error("L12 parameter tensor size mismatch");
        in.read(reinterpret_cast<char*>(parameter->value.data().data()),
                static_cast<std::streamsize>(parameter->value.numel() * sizeof(float)));
        if (!in) throw std::runtime_error("truncated L12 model weights");
    }
}

void write_model_config(std::ostream& out, const StableLatentTransformerConfig& c) {
    const std::array<std::size_t, 8> dims{c.latent_dim, c.model_dim, c.num_heads, c.num_layers, c.ffn_dim, c.text_feature_dim, c.prompt_tokens, c.time_feature_dim};
    for (const auto dim : dims) write_u64(out, static_cast<std::uint64_t>(dim));
    write_f32(out, c.norm_epsilon);
    write_f32(out, c.residual_scale);
}

void verify_model_config(std::istream& in, const StableLatentTransformerConfig& c) {
    const std::array<std::size_t, 8> dims{c.latent_dim, c.model_dim, c.num_heads, c.num_layers, c.ffn_dim, c.text_feature_dim, c.prompt_tokens, c.time_feature_dim};
    for (const auto dim : dims) if (read_u64(in) != dim) throw std::runtime_error("L12 model configuration mismatch");
    if (read_f32(in) != c.norm_epsilon || read_f32(in) != c.residual_scale) throw std::runtime_error("L12 model float configuration mismatch");
}

void write_trainer_config(std::ostream& out, const ScaleTrainerConfig& c) {
    write_f32(out, c.optimizer.learning_rate);
    write_f32(out, c.optimizer.beta1);
    write_f32(out, c.optimizer.beta2);
    write_f32(out, c.optimizer.epsilon);
    write_f32(out, c.optimizer.weight_decay);
    write_f32(out, c.max_grad_norm);
    write_f32(out, c.training_time);
    write_u64(out, c.micro_batch_size);
    write_u64(out, c.gradient_accumulation_steps);
    write_u64(out, c.shuffle_seed);
    write_u64(out, c.noise_seed);
}

void verify_trainer_config(std::istream& in, const ScaleTrainerConfig& c) {
    if (read_f32(in) != c.optimizer.learning_rate || read_f32(in) != c.optimizer.beta1 ||
        read_f32(in) != c.optimizer.beta2 || read_f32(in) != c.optimizer.epsilon ||
        read_f32(in) != c.optimizer.weight_decay || read_f32(in) != c.max_grad_norm ||
        read_f32(in) != c.training_time || read_u64(in) != c.micro_batch_size ||
        read_u64(in) != c.gradient_accumulation_steps || read_u64(in) != c.shuffle_seed ||
        read_u64(in) != c.noise_seed) {
        throw std::runtime_error("L12 trainer configuration mismatch");
    }
}

} // namespace

StableLatentTransformerBlock::StableLatentTransformerBlock(
    std::size_t model_dim,
    std::size_t num_heads,
    std::size_t ffn_dim,
    float norm_epsilon,
    float residual_scale,
    Random& rng)
    : self_norm_(model_dim, norm_epsilon),
      self_attention_(model_dim, num_heads, rng),
      cross_norm_(model_dim, norm_epsilon),
      cross_attention_(model_dim, num_heads, rng),
      ffn_norm_(model_dim, norm_epsilon),
      ffn_in_(model_dim, ffn_dim, rng, true),
      ffn_out_(ffn_dim, model_dim, rng, true),
      residual_scale_(residual_scale) {
    if (ffn_dim == 0 || !std::isfinite(residual_scale_) || residual_scale_ <= 0.0F) {
        throw std::invalid_argument("stable transformer block configuration is invalid");
    }
}

Tensor StableLatentTransformerBlock::forward(const Tensor& latent_tokens, const Tensor& prompt_tokens) const {
    const Tensor self_normed = self_norm_.forward(latent_tokens);
    Tensor after_self = add_scaled(latent_tokens, self_attention_.forward(self_normed, self_normed), residual_scale_);
    const Tensor cross_normed = cross_norm_.forward(after_self);
    Tensor after_cross = add_scaled(after_self, cross_attention_.forward(cross_normed, prompt_tokens), residual_scale_);
    const Tensor ffn_normed = ffn_norm_.forward(after_cross);
    return add_scaled(after_cross, ffn_out_.forward(silu(ffn_in_.forward(ffn_normed))), residual_scale_);
}

std::vector<nn::Parameter*> StableLatentTransformerBlock::parameters() {
    std::vector<nn::Parameter*> out;
    append_parameters(out, self_norm_.parameters());
    append_parameters(out, self_attention_.parameters());
    append_parameters(out, cross_norm_.parameters());
    append_parameters(out, cross_attention_.parameters());
    append_parameters(out, ffn_norm_.parameters());
    append_parameters(out, ffn_in_.parameters());
    append_parameters(out, ffn_out_.parameters());
    return out;
}

std::vector<const nn::Parameter*> StableLatentTransformerBlock::parameters() const {
    std::vector<const nn::Parameter*> out;
    append_parameters(out, self_norm_.parameters());
    append_parameters(out, self_attention_.parameters());
    append_parameters(out, cross_norm_.parameters());
    append_parameters(out, cross_attention_.parameters());
    append_parameters(out, ffn_norm_.parameters());
    append_parameters(out, ffn_in_.parameters());
    append_parameters(out, ffn_out_.parameters());
    return out;
}

StableLatentTransformerDenoiser::StableLatentTransformerDenoiser(StableLatentTransformerConfig config, Random& rng)
    : config_(config),
      residual_scale_(config.residual_scale == 0.0F ? 1.0F / std::sqrt(static_cast<float>(config.num_layers)) : config.residual_scale),
      input_projection_(config.latent_dim + 2 + config.time_feature_dim, config.model_dim, rng, true),
      prompt_projection_(config.text_feature_dim, config.model_dim, rng, true),
      final_norm_(config.model_dim, config.norm_epsilon),
      output_projection_(config.model_dim, config.latent_dim, rng, true) {
    if (config_.latent_dim == 0 || config_.model_dim == 0 || config_.num_heads == 0 || config_.num_layers == 0 ||
        config_.ffn_dim == 0 || config_.text_feature_dim == 0 || config_.prompt_tokens == 0 || config_.time_feature_dim == 0 ||
        config_.model_dim % config_.num_heads != 0 || config_.norm_epsilon <= 0.0F ||
        !std::isfinite(residual_scale_) || residual_scale_ <= 0.0F) {
        throw std::invalid_argument("invalid stable latent transformer configuration");
    }
    blocks_.reserve(config_.num_layers);
    for (std::size_t layer = 0; layer < config_.num_layers; ++layer) {
        blocks_.emplace_back(config_.model_dim, config_.num_heads, config_.ffn_dim, config_.norm_epsilon, residual_scale_, rng);
    }
}

Tensor StableLatentTransformerDenoiser::latent_conditioning(
    const Tensor& noisy_latent,
    float time,
    std::size_t grid_height,
    std::size_t grid_width) const {
    return build_latent_conditioning(config_, noisy_latent, time, grid_height, grid_width);
}

Tensor StableLatentTransformerDenoiser::predict(
    const Tensor& noisy_latent,
    std::string_view prompt,
    float time,
    std::size_t grid_height,
    std::size_t grid_width) const {
    Tensor tokens = input_projection_.forward(latent_conditioning(noisy_latent, time, grid_height, grid_width));
    const Tensor projected_prompt = prompt_projection_.forward(latent::prompt_token_features(prompt, config_.prompt_tokens, config_.text_feature_dim));
    for (const auto& block_value : blocks_) tokens = block_value.forward(tokens, projected_prompt);
    return output_projection_.forward(final_norm_.forward(tokens));
}

std::vector<nn::Parameter*> StableLatentTransformerDenoiser::parameters() {
    std::vector<nn::Parameter*> out;
    append_parameters(out, input_projection_.parameters());
    append_parameters(out, prompt_projection_.parameters());
    for (auto& block_value : blocks_) append_parameters(out, block_value.parameters());
    append_parameters(out, final_norm_.parameters());
    append_parameters(out, output_projection_.parameters());
    return out;
}

std::vector<const nn::Parameter*> StableLatentTransformerDenoiser::parameters() const {
    std::vector<const nn::Parameter*> out;
    append_parameters(out, input_projection_.parameters());
    append_parameters(out, prompt_projection_.parameters());
    for (const auto& block_value : blocks_) append_parameters(out, block_value.parameters());
    append_parameters(out, final_norm_.parameters());
    append_parameters(out, output_projection_.parameters());
    return out;
}

StableLatentTransformerBlock& StableLatentTransformerDenoiser::block(std::size_t index) {
    if (index >= blocks_.size()) throw std::out_of_range("stable latent transformer block index out of range");
    return blocks_[index];
}

const StableLatentTransformerBlock& StableLatentTransformerDenoiser::block(std::size_t index) const {
    if (index >= blocks_.size()) throw std::out_of_range("stable latent transformer block index out of range");
    return blocks_[index];
}

latent::ImagePromptDataset load_manifest_tsv(const std::string& manifest_path) {
    std::ifstream in(manifest_path);
    if (!in) throw std::runtime_error("failed to open image-prompt manifest");
    const std::filesystem::path base = std::filesystem::path(manifest_path).parent_path();
    latent::ImagePromptDataset dataset;
    std::string line;
    std::size_t line_number = 0;
    while (std::getline(in, line)) {
        ++line_number;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line[0] == '#') continue;
        const auto tab = line.find('\t');
        if (tab == std::string::npos || tab == 0) throw std::runtime_error("manifest line requires image_path<TAB>prompt at line " + std::to_string(line_number));
        const std::filesystem::path image_path = base / line.substr(0, tab);
        dataset.add(line.substr(tab + 1), vision::RgbImage::load_ppm(image_path.lexically_normal().string()));
    }
    if (dataset.size() == 0) throw std::runtime_error("image-prompt manifest contains no examples");
    return dataset;
}

std::vector<std::size_t> shuffled_indices(std::size_t size, std::uint64_t seed) {
    std::vector<std::size_t> indices(size);
    std::iota(indices.begin(), indices.end(), 0);
    Random rng(seed);
    for (std::size_t i = size; i > 1; --i) {
        const std::size_t j = static_cast<std::size_t>(rng.next_u64() % i);
        std::swap(indices[i - 1], indices[j]);
    }
    return indices;
}

DatasetSplit split_dataset(const latent::ImagePromptDataset& dataset, float validation_fraction, std::uint64_t seed) {
    if (dataset.size() == 0) throw std::invalid_argument("cannot split empty dataset");
    if (!std::isfinite(validation_fraction) || validation_fraction < 0.0F || validation_fraction >= 1.0F) {
        throw std::invalid_argument("validation fraction must be in [0,1)");
    }
    auto indices = shuffled_indices(dataset.size(), seed);
    std::size_t validation_count = static_cast<std::size_t>(std::floor(static_cast<float>(dataset.size()) * validation_fraction));
    if (validation_fraction > 0.0F && dataset.size() > 1 && validation_count == 0) validation_count = 1;
    if (validation_count >= dataset.size()) validation_count = dataset.size() - 1;

    DatasetSplit split;
    for (std::size_t position = 0; position < indices.size(); ++position) {
        const auto& example = dataset.at(indices[position]);
        if (position < validation_count) split.validation.add(example.prompt, example.image);
        else split.train.add(example.prompt, example.image);
    }
    return split;
}

StatefulAdamW::StatefulAdamW(std::vector<nn::Parameter*> parameters, train::AdamWConfig config)
    : config_(config) {
    if (config_.learning_rate <= 0.0F || config_.beta1 < 0.0F || config_.beta1 >= 1.0F ||
        config_.beta2 < 0.0F || config_.beta2 >= 1.0F || config_.epsilon <= 0.0F || config_.weight_decay < 0.0F) {
        throw std::invalid_argument("StatefulAdamW configuration is invalid");
    }
    states_.reserve(parameters.size());
    for (auto* parameter : parameters) {
        if (parameter == nullptr || !parameter->trainable) continue;
        parameter->ensure_grad();
        states_.push_back({parameter, Tensor::zeros(parameter->value.shape()), Tensor::zeros(parameter->value.shape())});
    }
}

void StatefulAdamW::zero_grad() {
    for (auto& state : states_) state.parameter->zero_grad();
}

void StatefulAdamW::step() {
    ++step_count_;
    const float correction1 = 1.0F - std::pow(config_.beta1, static_cast<float>(step_count_));
    const float correction2 = 1.0F - std::pow(config_.beta2, static_cast<float>(step_count_));
    for (auto& state : states_) {
        auto& parameter = *state.parameter;
        parameter.ensure_grad();
        for (std::size_t i = 0; i < parameter.value.numel(); ++i) {
            const float g = parameter.grad.data()[i];
            state.first_moment.data()[i] = config_.beta1 * state.first_moment.data()[i] + (1.0F - config_.beta1) * g;
            state.second_moment.data()[i] = config_.beta2 * state.second_moment.data()[i] + (1.0F - config_.beta2) * g * g;
            const float m_hat = state.first_moment.data()[i] / correction1;
            const float v_hat = state.second_moment.data()[i] / correction2;
            const float update = m_hat / (std::sqrt(v_hat) + config_.epsilon) + config_.weight_decay * parameter.value.data()[i];
            parameter.value.data()[i] -= config_.learning_rate * update;
        }
    }
}

void StatefulAdamW::save(std::ostream& out) const {
    write_u64(out, step_count_);
    write_u64(out, static_cast<std::uint64_t>(states_.size()));
    for (const auto& state : states_) {
        write_u64(out, static_cast<std::uint64_t>(state.first_moment.numel()));
        out.write(reinterpret_cast<const char*>(state.first_moment.data().data()), static_cast<std::streamsize>(state.first_moment.numel() * sizeof(float)));
        out.write(reinterpret_cast<const char*>(state.second_moment.data().data()), static_cast<std::streamsize>(state.second_moment.numel() * sizeof(float)));
        if (!out) throw std::runtime_error("failed while writing L12 optimizer state");
    }
}

void StatefulAdamW::load(std::istream& in) {
    step_count_ = read_u64(in);
    if (read_u64(in) != states_.size()) throw std::runtime_error("L12 optimizer state count mismatch");
    for (auto& state : states_) {
        if (read_u64(in) != state.first_moment.numel()) throw std::runtime_error("L12 optimizer tensor size mismatch");
        in.read(reinterpret_cast<char*>(state.first_moment.data().data()), static_cast<std::streamsize>(state.first_moment.numel() * sizeof(float)));
        in.read(reinterpret_cast<char*>(state.second_moment.data().data()), static_cast<std::streamsize>(state.second_moment.numel() * sizeof(float)));
        if (!in) throw std::runtime_error("truncated L12 optimizer state");
    }
}

ScaleTrainer::ScaleTrainer(
    StableLatentTransformerDenoiser& model,
    const multimodal::ImageAutoencoder& autoencoder,
    flow::NoiseScheduler scheduler,
    ScaleTrainerConfig config)
    : model_(model),
      autoencoder_(autoencoder),
      scheduler_(scheduler),
      config_(config),
      optimizer_(model.parameters(), config.optimizer) {
    if (model_.config().latent_dim != autoencoder_.config().latent_dim) throw std::invalid_argument("L12 model/autoencoder latent dimensions must match");
    if (config_.max_grad_norm <= 0.0F || config_.training_time < 0.0F || config_.training_time > 1.0F ||
        config_.micro_batch_size == 0 || config_.gradient_accumulation_steps == 0) {
        throw std::invalid_argument("invalid L12 trainer configuration");
    }
}

float ScaleTrainer::evaluate_dataset(const latent::ImagePromptDataset& dataset, std::uint64_t noise_seed) const {
    if (dataset.size() == 0) return 0.0F;
    float total = 0.0F;
    for (std::size_t i = 0; i < dataset.size(); ++i) {
        total += evaluate_example(model_, autoencoder_, scheduler_, config_.training_time, dataset.at(i), noise_seed + i * 7919ULL);
    }
    return total / static_cast<float>(dataset.size());
}

ScaleMetrics ScaleTrainer::train_epoch(
    const latent::ImagePromptDataset& train_dataset,
    const latent::ImagePromptDataset& validation_dataset) {
    if (train_dataset.size() == 0) throw std::invalid_argument("cannot train empty L12 dataset");
    const auto order = shuffled_indices(train_dataset.size(), config_.shuffle_seed + epoch_ * 104729ULL);
    const std::size_t effective_batch = config_.micro_batch_size * config_.gradient_accumulation_steps;
    float total_loss = 0.0F;
    float last_grad_norm = 0.0F;

    for (std::size_t start = 0; start < order.size(); start += effective_batch) {
        const std::size_t count = std::min(effective_batch, order.size() - start);
        optimizer_.zero_grad();
        for (std::size_t local = 0; local < count; ++local) {
            const std::size_t dataset_index = order[start + local];
            const std::uint64_t noise_seed = config_.noise_seed + epoch_ * 1000003ULL + dataset_index * 7919ULL;
            total_loss += backward_example(
                model_, autoencoder_, scheduler_, config_.training_time, train_dataset.at(dataset_index), noise_seed,
                1.0F / static_cast<float>(count));
        }
        auto parameters = model_.parameters();
        last_grad_norm = train::global_grad_norm(parameters);
        train::clip_grad_norm(parameters, config_.max_grad_norm);
        optimizer_.step();
    }

    ++epoch_;
    ScaleMetrics metrics;
    metrics.epoch = epoch_;
    metrics.optimizer_steps = optimizer_.step_count();
    metrics.train_loss = total_loss / static_cast<float>(train_dataset.size());
    metrics.validation_loss = evaluate_dataset(validation_dataset, config_.noise_seed + epoch_ * 16127ULL);
    metrics.last_grad_norm = last_grad_norm;
    return metrics;
}

void save_training_checkpoint(
    const StableLatentTransformerDenoiser& model,
    const ScaleTrainer& trainer,
    const std::string& path) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("failed to open L12 checkpoint for writing");
    constexpr std::array<char, 8> magic{'S','P','S','C','A','L','E','1'};
    out.write(magic.data(), static_cast<std::streamsize>(magic.size()));
    write_model_config(out, model.config());
    write_parameters(out, model.parameters());
    write_trainer_config(out, trainer.config());
    write_u64(out, trainer.epoch());
    trainer.optimizer().save(out);
}

void load_training_checkpoint(
    StableLatentTransformerDenoiser& model,
    ScaleTrainer& trainer,
    const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("failed to open L12 checkpoint for reading");
    std::array<char, 8> magic{};
    in.read(magic.data(), static_cast<std::streamsize>(magic.size()));
    constexpr std::array<char, 8> expected{'S','P','S','C','A','L','E','1'};
    if (!in || magic != expected) throw std::runtime_error("invalid L12 checkpoint");
    verify_model_config(in, model.config());
    read_parameters(in, model.parameters());
    verify_trainer_config(in, trainer.config());
    trainer.set_epoch(read_u64(in));
    trainer.optimizer().load(in);
}

void append_metrics_csv(const std::string& path, const ScaleMetrics& metrics) {
    const bool needs_header = !std::filesystem::exists(path) || std::filesystem::file_size(path) == 0;
    std::ofstream out(path, std::ios::app);
    if (!out) throw std::runtime_error("failed to open L12 metrics CSV");
    if (needs_header) out << "epoch,optimizer_steps,train_loss,validation_loss,last_grad_norm\n";
    out << metrics.epoch << ',' << metrics.optimizer_steps << ',' << metrics.train_loss << ','
        << metrics.validation_loss << ',' << metrics.last_grad_norm << '\n';
    if (!out) throw std::runtime_error("failed while writing L12 metrics CSV");
}

} // namespace spiral::scale
