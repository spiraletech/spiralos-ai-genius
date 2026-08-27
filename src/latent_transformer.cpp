#include "spiral/latent_transformer.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <utility>

namespace spiral::latent {
namespace {

void append_parameters(std::vector<nn::Parameter*>& out, std::vector<nn::Parameter*> values) {
    out.insert(out.end(), values.begin(), values.end());
}

void append_parameters(std::vector<const nn::Parameter*>& out, std::vector<const nn::Parameter*> values) {
    out.insert(out.end(), values.begin(), values.end());
}

Tensor add_tensors(const Tensor& lhs, const Tensor& rhs) {
    if (lhs.shape() != rhs.shape()) throw std::invalid_argument("tensor add shape mismatch");
    Tensor out(lhs.shape());
    for (std::size_t i = 0; i < lhs.numel(); ++i) out.data()[i] = lhs.data()[i] + rhs.data()[i];
    return out;
}

void add_inplace(Tensor& lhs, const Tensor& rhs) {
    if (lhs.shape() != rhs.shape()) throw std::invalid_argument("tensor add_inplace shape mismatch");
    for (std::size_t i = 0; i < lhs.numel(); ++i) lhs.data()[i] += rhs.data()[i];
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
    if (input.shape() != grad_output.shape()) throw std::invalid_argument("silu backward shape mismatch");
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
    MultiHeadAttention& attention,
    const Tensor& query,
    const Tensor& context,
    AttentionCache& cache) {
    if (query.rank() != 2 || context.rank() != 2 ||
        query.shape()[1] != attention.model_dim() || context.shape()[1] != attention.model_dim() ||
        query.shape()[0] == 0 || context.shape()[0] == 0) {
        throw std::invalid_argument("attention requires non-empty [sequence,model_dim] matrices");
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
    const float scale = 1.0F / std::sqrt(static_cast<float>(head_dim));

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
                    dot += cache.q.data()[qi * model_dim + offset + d] *
                           cache.k.data()[ki * model_dim + offset + d];
                }
                scores[ki] = dot * scale;
                maximum = std::max(maximum, scores[ki]);
            }
            float sum = 0.0F;
            const std::size_t weight_base = (head * qseq + qi) * kseq;
            for (std::size_t ki = 0; ki < kseq; ++ki) {
                const float value = std::exp(scores[ki] - maximum);
                cache.weights.data()[weight_base + ki] = value;
                sum += value;
            }
            if (!(sum > 0.0F) || !std::isfinite(sum)) throw std::runtime_error("attention softmax failed");
            for (std::size_t ki = 0; ki < kseq; ++ki) cache.weights.data()[weight_base + ki] /= sum;

            for (std::size_t d = 0; d < head_dim; ++d) {
                float value = 0.0F;
                for (std::size_t ki = 0; ki < kseq; ++ki) {
                    value += cache.weights.data()[weight_base + ki] *
                             cache.v.data()[ki * model_dim + offset + d];
                }
                cache.merged.data()[qi * model_dim + offset + d] = value;
            }
        }
    }
    return attention.out_proj().forward(cache.merged);
}

std::pair<Tensor, Tensor> attention_backward(
    MultiHeadAttention& attention,
    const AttentionCache& cache,
    const Tensor& grad_output) {
    const std::size_t qseq = cache.query_input.shape()[0];
    const std::size_t kseq = cache.context_input.shape()[0];
    const std::size_t model_dim = attention.model_dim();
    const std::size_t heads = attention.num_heads();
    const std::size_t head_dim = attention.head_dim();
    const float scale = 1.0F / std::sqrt(static_cast<float>(head_dim));

    const Tensor grad_merged = linear_backward(attention.out_proj(), cache.merged, grad_output);
    Tensor grad_weights({heads, qseq, kseq});
    Tensor grad_v({kseq, model_dim});

    for (std::size_t head = 0; head < heads; ++head) {
        const std::size_t offset = head * head_dim;
        for (std::size_t qi = 0; qi < qseq; ++qi) {
            const std::size_t weight_base = (head * qseq + qi) * kseq;
            for (std::size_t ki = 0; ki < kseq; ++ki) {
                float dot = 0.0F;
                for (std::size_t d = 0; d < head_dim; ++d) {
                    const float g = grad_merged.data()[qi * model_dim + offset + d];
                    dot += g * cache.v.data()[ki * model_dim + offset + d];
                    grad_v.data()[ki * model_dim + offset + d] += cache.weights.data()[weight_base + ki] * g;
                }
                grad_weights.data()[weight_base + ki] = dot;
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
                const float gs = grad_scores.data()[base + ki] * scale;
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
    AttentionCache self_attention;
    Tensor after_self;
    AttentionCache cross_attention;
    Tensor after_cross;
    Tensor ffn_pre;
    Tensor ffn_active;
};

Tensor block_forward(
    LatentTransformerBlock& block,
    const Tensor& input,
    const Tensor& prompt,
    BlockCache& cache) {
    cache.input = input;
    const Tensor self = attention_forward(block.self_attention(), input, input, cache.self_attention);
    cache.after_self = add_tensors(input, self);
    const Tensor cross = attention_forward(block.cross_attention(), cache.after_self, prompt, cache.cross_attention);
    cache.after_cross = add_tensors(cache.after_self, cross);
    cache.ffn_pre = block.ffn_in().forward(cache.after_cross);
    cache.ffn_active = silu(cache.ffn_pre);
    return add_tensors(cache.after_cross, block.ffn_out().forward(cache.ffn_active));
}

std::pair<Tensor, Tensor> block_backward(
    LatentTransformerBlock& block,
    const BlockCache& cache,
    const Tensor& grad_output) {
    Tensor grad_after_cross = grad_output;
    const Tensor grad_ffn_active = linear_backward(block.ffn_out(), cache.ffn_active, grad_output);
    const Tensor grad_ffn_pre = silu_backward(cache.ffn_pre, grad_ffn_active);
    add_inplace(grad_after_cross, linear_backward(block.ffn_in(), cache.after_cross, grad_ffn_pre));

    auto [grad_cross_query, grad_prompt] = attention_backward(block.cross_attention(), cache.cross_attention, grad_after_cross);
    Tensor grad_after_self = grad_after_cross;
    add_inplace(grad_after_self, grad_cross_query);

    auto [grad_self_query, grad_self_context] = attention_backward(block.self_attention(), cache.self_attention, grad_after_self);
    Tensor grad_input = grad_after_self;
    add_inplace(grad_input, grad_self_query);
    add_inplace(grad_input, grad_self_context);
    return {std::move(grad_input), std::move(grad_prompt)};
}

struct ModelCache {
    Tensor raw_latent_conditioning;
    Tensor raw_prompt_tokens;
    Tensor prompt_tokens;
    std::vector<BlockCache> blocks;
    Tensor final_tokens;
};

Tensor model_forward_cached(
    LatentTransformerDenoiser& model,
    const Tensor& noisy_latent,
    std::string_view prompt,
    float time,
    std::size_t grid_height,
    std::size_t grid_width,
    ModelCache& cache) {
    const auto& config = model.config();
    if (noisy_latent.rank() != 2 || noisy_latent.shape()[0] != grid_height * grid_width || noisy_latent.shape()[1] != config.latent_dim) {
        throw std::invalid_argument("latent transformer noisy latent shape mismatch");
    }

    const Tensor time_features = flow::timestep_features(time, config.time_feature_dim);
    const std::size_t input_dim = config.latent_dim + 2 + config.time_feature_dim;
    cache.raw_latent_conditioning = Tensor({grid_height * grid_width, input_dim});
    for (std::size_t y = 0; y < grid_height; ++y) {
        for (std::size_t x = 0; x < grid_width; ++x) {
            const std::size_t row = y * grid_width + x;
            const std::size_t base = row * input_dim;
            for (std::size_t d = 0; d < config.latent_dim; ++d) {
                cache.raw_latent_conditioning.data()[base + d] = noisy_latent.data()[row * config.latent_dim + d];
            }
            cache.raw_latent_conditioning.data()[base + config.latent_dim] =
                grid_width == 1 ? 0.0F : 2.0F * static_cast<float>(x) / static_cast<float>(grid_width - 1) - 1.0F;
            cache.raw_latent_conditioning.data()[base + config.latent_dim + 1] =
                grid_height == 1 ? 0.0F : 2.0F * static_cast<float>(y) / static_cast<float>(grid_height - 1) - 1.0F;
            for (std::size_t d = 0; d < config.time_feature_dim; ++d) {
                cache.raw_latent_conditioning.data()[base + config.latent_dim + 2 + d] = time_features.data()[d];
            }
        }
    }

    cache.raw_prompt_tokens = prompt_token_features(prompt, config.prompt_tokens, config.text_feature_dim);
    Tensor tokens = model.input_projection().forward(cache.raw_latent_conditioning);
    cache.prompt_tokens = model.prompt_projection().forward(cache.raw_prompt_tokens);
    cache.blocks.clear();
    cache.blocks.resize(config.num_layers);
    for (std::size_t layer = 0; layer < config.num_layers; ++layer) {
        tokens = block_forward(model.block(layer), tokens, cache.prompt_tokens, cache.blocks[layer]);
    }
    cache.final_tokens = tokens;
    return model.output_projection().forward(tokens);
}

void model_backward(
    LatentTransformerDenoiser& model,
    const ModelCache& cache,
    const Tensor& grad_output) {
    Tensor grad_tokens = linear_backward(model.output_projection(), cache.final_tokens, grad_output);
    Tensor grad_prompt(cache.prompt_tokens.shape());
    for (std::size_t layer = model.config().num_layers; layer-- > 0;) {
        auto [grad_input, grad_layer_prompt] = block_backward(model.block(layer), cache.blocks[layer], grad_tokens);
        grad_tokens = std::move(grad_input);
        add_inplace(grad_prompt, grad_layer_prompt);
    }
    (void)linear_backward(model.input_projection(), cache.raw_latent_conditioning, grad_tokens);
    (void)linear_backward(model.prompt_projection(), cache.raw_prompt_tokens, grad_prompt);
}

void write_u64(std::ostream& out, std::uint64_t value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

std::uint64_t read_u64(std::istream& in) {
    std::uint64_t value = 0;
    in.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!in) throw std::runtime_error("truncated Spiral latent transformer checkpoint");
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
        if (!out) throw std::runtime_error("failed while writing Spiral latent transformer checkpoint");
    }
}

void read_parameters(std::istream& in, const std::vector<nn::Parameter*>& parameters) {
    if (read_u64(in) != parameters.size()) throw std::runtime_error("latent transformer parameter count mismatch");
    for (auto* parameter : parameters) {
        if (read_u64(in) != parameter->value.rank()) throw std::runtime_error("latent transformer rank mismatch");
        for (std::size_t i = 0; i < parameter->value.rank(); ++i) {
            if (read_u64(in) != parameter->value.shape()[i]) throw std::runtime_error("latent transformer shape mismatch");
        }
        if (read_u64(in) != parameter->value.numel()) throw std::runtime_error("latent transformer tensor size mismatch");
        in.read(reinterpret_cast<char*>(parameter->value.data().data()),
                static_cast<std::streamsize>(parameter->value.numel() * sizeof(float)));
        if (!in) throw std::runtime_error("truncated Spiral latent transformer weights");
    }
}

} // namespace

Tensor prompt_token_features(std::string_view prompt, std::size_t token_count, std::size_t feature_dim) {
    if (token_count == 0 || feature_dim == 0) throw std::invalid_argument("prompt token dimensions must be non-zero");
    Tensor tokens({token_count, feature_dim});
    for (std::size_t i = 0; i < prompt.size(); ++i) {
        const auto byte = static_cast<unsigned char>(prompt[i]);
        const std::size_t token = i % token_count;
        const std::size_t primary = (static_cast<std::size_t>(byte) + i * 17U) % feature_dim;
        const std::size_t secondary = (static_cast<std::size_t>(byte) * 131U + i * 7U + 11U) % feature_dim;
        tokens.data()[token * feature_dim + primary] += 1.0F;
        tokens.data()[token * feature_dim + secondary] += 0.5F;
    }
    for (std::size_t token = 0; token < token_count; ++token) {
        double norm_sq = 0.0;
        for (std::size_t d = 0; d < feature_dim; ++d) {
            const float value = tokens.data()[token * feature_dim + d];
            norm_sq += static_cast<double>(value) * value;
        }
        if (norm_sq > 0.0) {
            const float inv = 1.0F / static_cast<float>(std::sqrt(norm_sq));
            for (std::size_t d = 0; d < feature_dim; ++d) tokens.data()[token * feature_dim + d] *= inv;
        }
    }
    return tokens;
}

MultiHeadAttention::MultiHeadAttention(std::size_t model_dim, std::size_t num_heads, Random& rng)
    : model_dim_(model_dim),
      num_heads_(num_heads),
      head_dim_(num_heads == 0 ? 0 : model_dim / num_heads),
      q_proj_(model_dim, model_dim, rng, false),
      k_proj_(model_dim, model_dim, rng, false),
      v_proj_(model_dim, model_dim, rng, false),
      out_proj_(model_dim, model_dim, rng, false) {
    if (model_dim == 0 || num_heads == 0 || model_dim % num_heads != 0) {
        throw std::invalid_argument("latent attention dimensions must be non-zero and divisible by head count");
    }
}

Tensor MultiHeadAttention::forward(const Tensor& query, const Tensor& context) const {
    if (query.rank() != 2 || context.rank() != 2 ||
        query.shape()[1] != model_dim_ || context.shape()[1] != model_dim_ ||
        query.shape()[0] == 0 || context.shape()[0] == 0) {
        throw std::invalid_argument("MultiHeadAttention requires non-empty [sequence,model_dim] matrices");
    }
    const Tensor q = q_proj_.forward(query);
    const Tensor k = k_proj_.forward(context);
    const Tensor v = v_proj_.forward(context);
    const std::size_t qseq = query.shape()[0];
    const std::size_t kseq = context.shape()[0];
    const float scale = 1.0F / std::sqrt(static_cast<float>(head_dim_));
    Tensor merged({qseq, model_dim_});
    std::vector<float> scores(kseq);
    std::vector<float> weights(kseq);

    for (std::size_t head = 0; head < num_heads_; ++head) {
        const std::size_t offset = head * head_dim_;
        for (std::size_t qi = 0; qi < qseq; ++qi) {
            float maximum = -std::numeric_limits<float>::infinity();
            for (std::size_t ki = 0; ki < kseq; ++ki) {
                float dot = 0.0F;
                for (std::size_t d = 0; d < head_dim_; ++d) {
                    dot += q.data()[qi * model_dim_ + offset + d] * k.data()[ki * model_dim_ + offset + d];
                }
                scores[ki] = dot * scale;
                maximum = std::max(maximum, scores[ki]);
            }
            float sum = 0.0F;
            for (std::size_t ki = 0; ki < kseq; ++ki) {
                weights[ki] = std::exp(scores[ki] - maximum);
                sum += weights[ki];
            }
            if (!(sum > 0.0F)) throw std::runtime_error("latent attention softmax failed");
            for (float& weight : weights) weight /= sum;
            for (std::size_t d = 0; d < head_dim_; ++d) {
                float value = 0.0F;
                for (std::size_t ki = 0; ki < kseq; ++ki) {
                    value += weights[ki] * v.data()[ki * model_dim_ + offset + d];
                }
                merged.data()[qi * model_dim_ + offset + d] = value;
            }
        }
    }
    return out_proj_.forward(merged);
}

std::vector<nn::Parameter*> MultiHeadAttention::parameters() {
    std::vector<nn::Parameter*> out;
    append_parameters(out, q_proj_.parameters());
    append_parameters(out, k_proj_.parameters());
    append_parameters(out, v_proj_.parameters());
    append_parameters(out, out_proj_.parameters());
    return out;
}

std::vector<const nn::Parameter*> MultiHeadAttention::parameters() const {
    std::vector<const nn::Parameter*> out;
    append_parameters(out, q_proj_.parameters());
    append_parameters(out, k_proj_.parameters());
    append_parameters(out, v_proj_.parameters());
    append_parameters(out, out_proj_.parameters());
    return out;
}

LatentTransformerBlock::LatentTransformerBlock(
    std::size_t model_dim,
    std::size_t num_heads,
    std::size_t ffn_dim,
    Random& rng)
    : self_attention_(model_dim, num_heads, rng),
      cross_attention_(model_dim, num_heads, rng),
      ffn_in_(model_dim, ffn_dim, rng, true),
      ffn_out_(ffn_dim, model_dim, rng, true) {
    if (ffn_dim == 0) throw std::invalid_argument("latent transformer FFN dimension must be non-zero");
}

Tensor LatentTransformerBlock::forward(const Tensor& latent_tokens, const Tensor& prompt_tokens) const {
    Tensor after_self = add_tensors(latent_tokens, self_attention_.forward(latent_tokens, latent_tokens));
    Tensor after_cross = add_tensors(after_self, cross_attention_.forward(after_self, prompt_tokens));
    return add_tensors(after_cross, ffn_out_.forward(silu(ffn_in_.forward(after_cross))));
}

std::vector<nn::Parameter*> LatentTransformerBlock::parameters() {
    std::vector<nn::Parameter*> out;
    append_parameters(out, self_attention_.parameters());
    append_parameters(out, cross_attention_.parameters());
    append_parameters(out, ffn_in_.parameters());
    append_parameters(out, ffn_out_.parameters());
    return out;
}

std::vector<const nn::Parameter*> LatentTransformerBlock::parameters() const {
    std::vector<const nn::Parameter*> out;
    append_parameters(out, self_attention_.parameters());
    append_parameters(out, cross_attention_.parameters());
    append_parameters(out, ffn_in_.parameters());
    append_parameters(out, ffn_out_.parameters());
    return out;
}

LatentTransformerDenoiser::LatentTransformerDenoiser(LatentTransformerConfig config, Random& rng)
    : config_(config),
      input_projection_(config.latent_dim + 2 + config.time_feature_dim, config.model_dim, rng, true),
      prompt_projection_(config.text_feature_dim, config.model_dim, rng, true),
      output_projection_(config.model_dim, config.latent_dim, rng, true) {
    if (config_.latent_dim == 0 || config_.model_dim == 0 || config_.num_heads == 0 || config_.num_layers == 0 ||
        config_.ffn_dim == 0 || config_.text_feature_dim == 0 || config_.prompt_tokens == 0 || config_.time_feature_dim == 0 ||
        config_.model_dim % config_.num_heads != 0) {
        throw std::invalid_argument("invalid latent transformer configuration");
    }
    blocks_.reserve(config_.num_layers);
    for (std::size_t i = 0; i < config_.num_layers; ++i) {
        blocks_.emplace_back(config_.model_dim, config_.num_heads, config_.ffn_dim, rng);
    }
}

Tensor LatentTransformerDenoiser::latent_conditioning(
    const Tensor& noisy_latent,
    float time,
    std::size_t grid_height,
    std::size_t grid_width) const {
    if (noisy_latent.rank() != 2 || noisy_latent.shape()[0] != grid_height * grid_width || noisy_latent.shape()[1] != config_.latent_dim) {
        throw std::invalid_argument("latent transformer noisy latent shape mismatch");
    }
    const Tensor time_features = flow::timestep_features(time, config_.time_feature_dim);
    const std::size_t input_dim = config_.latent_dim + 2 + config_.time_feature_dim;
    Tensor matrix({grid_height * grid_width, input_dim});
    for (std::size_t y = 0; y < grid_height; ++y) {
        for (std::size_t x = 0; x < grid_width; ++x) {
            const std::size_t row = y * grid_width + x;
            const std::size_t base = row * input_dim;
            for (std::size_t d = 0; d < config_.latent_dim; ++d) matrix.data()[base + d] = noisy_latent.data()[row * config_.latent_dim + d];
            matrix.data()[base + config_.latent_dim] = grid_width == 1 ? 0.0F : 2.0F * static_cast<float>(x) / static_cast<float>(grid_width - 1) - 1.0F;
            matrix.data()[base + config_.latent_dim + 1] = grid_height == 1 ? 0.0F : 2.0F * static_cast<float>(y) / static_cast<float>(grid_height - 1) - 1.0F;
            for (std::size_t d = 0; d < config_.time_feature_dim; ++d) matrix.data()[base + config_.latent_dim + 2 + d] = time_features.data()[d];
        }
    }
    return matrix;
}

Tensor LatentTransformerDenoiser::predict(
    const Tensor& noisy_latent,
    std::string_view prompt,
    float time,
    std::size_t grid_height,
    std::size_t grid_width) const {
    Tensor tokens = input_projection_.forward(latent_conditioning(noisy_latent, time, grid_height, grid_width));
    const Tensor prompt_tokens = prompt_projection_.forward(prompt_token_features(prompt, config_.prompt_tokens, config_.text_feature_dim));
    for (const auto& block_value : blocks_) tokens = block_value.forward(tokens, prompt_tokens);
    return output_projection_.forward(tokens);
}

std::vector<nn::Parameter*> LatentTransformerDenoiser::parameters() {
    std::vector<nn::Parameter*> out;
    append_parameters(out, input_projection_.parameters());
    append_parameters(out, prompt_projection_.parameters());
    for (auto& block_value : blocks_) append_parameters(out, block_value.parameters());
    append_parameters(out, output_projection_.parameters());
    return out;
}

std::vector<const nn::Parameter*> LatentTransformerDenoiser::parameters() const {
    std::vector<const nn::Parameter*> out;
    append_parameters(out, input_projection_.parameters());
    append_parameters(out, prompt_projection_.parameters());
    for (const auto& block_value : blocks_) append_parameters(out, block_value.parameters());
    append_parameters(out, output_projection_.parameters());
    return out;
}

LatentTransformerBlock& LatentTransformerDenoiser::block(std::size_t index) {
    if (index >= blocks_.size()) throw std::out_of_range("latent transformer block index out of range");
    return blocks_[index];
}

const LatentTransformerBlock& LatentTransformerDenoiser::block(std::size_t index) const {
    if (index >= blocks_.size()) throw std::out_of_range("latent transformer block index out of range");
    return blocks_[index];
}

void ImagePromptDataset::add(std::string prompt, vision::RgbImage image) {
    if (image.width() == 0 || image.height() == 0) throw std::invalid_argument("dataset image must be non-empty");
    examples_.push_back({std::move(prompt), std::move(image)});
}

const ImagePromptExample& ImagePromptDataset::at(std::size_t index) const {
    if (index >= examples_.size()) throw std::out_of_range("dataset index out of range");
    return examples_[index];
}

LatentTransformerTrainer::LatentTransformerTrainer(
    LatentTransformerDenoiser& model,
    const multimodal::ImageAutoencoder& autoencoder,
    flow::NoiseScheduler scheduler,
    LatentTransformerTrainerConfig config)
    : model_(model),
      autoencoder_(autoencoder),
      scheduler_(scheduler),
      config_(config),
      optimizer_(model.parameters(), config.optimizer) {
    if (model_.config().latent_dim != autoencoder_.config().latent_dim) {
        throw std::invalid_argument("latent transformer and autoencoder latent dimensions must match");
    }
    if (config_.training_time < 0.0F || config_.training_time > 1.0F) throw std::invalid_argument("training time must be in [0,1]");
}

float LatentTransformerTrainer::evaluate_example(const ImagePromptExample& example, std::uint64_t noise_seed) const {
    const std::size_t patch = autoencoder_.config().patch_size;
    if (example.image.width() % patch != 0 || example.image.height() % patch != 0) {
        throw std::invalid_argument("dataset image dimensions must align to autoencoder patch size");
    }
    const std::size_t grid_h = example.image.height() / patch;
    const std::size_t grid_w = example.image.width() / patch;
    const Tensor target = autoencoder_.encode(example.image);
    Random rng(noise_seed);
    const Tensor noise = flow::gaussian_noise(target.shape(), rng);
    const Tensor noisy = scheduler_.add_noise(target, noise, config_.training_time);
    return multimodal::mean_squared_error(model_.predict(noisy, example.prompt, config_.training_time, grid_h, grid_w), target);
}

float LatentTransformerTrainer::evaluate_dataset(const ImagePromptDataset& dataset, std::uint64_t noise_seed) const {
    if (dataset.size() == 0) throw std::invalid_argument("cannot evaluate empty image-prompt dataset");
    float total = 0.0F;
    for (std::size_t i = 0; i < dataset.size(); ++i) total += evaluate_example(dataset.at(i), noise_seed + i * 7919ULL);
    return total / static_cast<float>(dataset.size());
}

float LatentTransformerTrainer::train_example(const ImagePromptExample& example, std::uint64_t noise_seed) {
    const std::size_t patch = autoencoder_.config().patch_size;
    if (example.image.width() % patch != 0 || example.image.height() % patch != 0) {
        throw std::invalid_argument("dataset image dimensions must align to autoencoder patch size");
    }
    const std::size_t grid_h = example.image.height() / patch;
    const std::size_t grid_w = example.image.width() / patch;
    const Tensor target = autoencoder_.encode(example.image);
    Random rng(noise_seed);
    const Tensor noise = flow::gaussian_noise(target.shape(), rng);
    const Tensor noisy = scheduler_.add_noise(target, noise, config_.training_time);

    optimizer_.zero_grad();
    ModelCache cache;
    const Tensor prediction = model_forward_cached(model_, noisy, example.prompt, config_.training_time, grid_h, grid_w, cache);
    const float loss = multimodal::mean_squared_error(prediction, target);
    model_backward(model_, cache, mse_gradient(prediction, target));
    auto parameters = model_.parameters();
    train::clip_grad_norm(parameters, config_.max_grad_norm);
    optimizer_.step();
    return loss;
}

float LatentTransformerTrainer::train_batch(const ImagePromptDataset& dataset, std::uint64_t noise_seed) {
    if (dataset.size() == 0) throw std::invalid_argument("cannot train empty image-prompt dataset");
    optimizer_.zero_grad();
    float total = 0.0F;
    for (std::size_t i = 0; i < dataset.size(); ++i) {
        const auto& example = dataset.at(i);
        const std::size_t patch = autoencoder_.config().patch_size;
        if (example.image.width() % patch != 0 || example.image.height() % patch != 0) {
            throw std::invalid_argument("dataset image dimensions must align to autoencoder patch size");
        }
        const std::size_t grid_h = example.image.height() / patch;
        const std::size_t grid_w = example.image.width() / patch;
        const Tensor target = autoencoder_.encode(example.image);
        Random rng(noise_seed + i * 7919ULL);
        const Tensor noise = flow::gaussian_noise(target.shape(), rng);
        const Tensor noisy = scheduler_.add_noise(target, noise, config_.training_time);
        ModelCache cache;
        const Tensor prediction = model_forward_cached(model_, noisy, example.prompt, config_.training_time, grid_h, grid_w, cache);
        total += multimodal::mean_squared_error(prediction, target);
        Tensor grad = mse_gradient(prediction, target);
        const float inv_batch = 1.0F / static_cast<float>(dataset.size());
        for (float& value : grad.data()) value *= inv_batch;
        model_backward(model_, cache, grad);
    }
    auto parameters = model_.parameters();
    train::clip_grad_norm(parameters, config_.max_grad_norm);
    optimizer_.step();
    return total / static_cast<float>(dataset.size());
}

float LatentTransformerTrainer::train_epoch(const ImagePromptDataset& dataset) {
    const std::uint64_t seed = config_.noise_seed + epoch_ * 104729ULL;
    ++epoch_;
    return train_batch(dataset, seed);
}

void save_latent_transformer(const LatentTransformerDenoiser& model, const std::string& path) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("failed to open latent transformer checkpoint for writing");
    constexpr std::array<char, 8> magic{'S','P','L','T','R','F','1','\0'};
    out.write(magic.data(), static_cast<std::streamsize>(magic.size()));
    const auto& c = model.config();
    const std::array<std::size_t, 8> dims{c.latent_dim, c.model_dim, c.num_heads, c.num_layers, c.ffn_dim, c.text_feature_dim, c.prompt_tokens, c.time_feature_dim};
    for (const auto dim : dims) write_u64(out, static_cast<std::uint64_t>(dim));
    write_parameters(out, model.parameters());
}

void load_latent_transformer(LatentTransformerDenoiser& model, const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("failed to open latent transformer checkpoint for reading");
    std::array<char, 8> magic{};
    in.read(magic.data(), static_cast<std::streamsize>(magic.size()));
    constexpr std::array<char, 8> expected{'S','P','L','T','R','F','1','\0'};
    if (!in || magic != expected) throw std::runtime_error("invalid Spiral latent transformer checkpoint");
    const auto& c = model.config();
    const std::array<std::size_t, 8> dims{c.latent_dim, c.model_dim, c.num_heads, c.num_layers, c.ffn_dim, c.text_feature_dim, c.prompt_tokens, c.time_feature_dim};
    for (const auto dim : dims) {
        if (read_u64(in) != dim) throw std::runtime_error("latent transformer checkpoint configuration mismatch");
    }
    read_parameters(in, model.parameters());
}

} // namespace spiral::latent
