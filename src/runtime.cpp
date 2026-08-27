#include "spiral/runtime.hpp"

#include "spiral/random.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <utility>

namespace spiral::runtime {
namespace {

constexpr std::uint32_t bundle_magic = 0x53414947U;
constexpr std::uint32_t bundle_version = 1U;

void write_u32(std::ostream& out, std::uint32_t value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(value));
}
void write_u64(std::ostream& out, std::uint64_t value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(value));
}
std::uint32_t read_u32(std::istream& in) {
    std::uint32_t value = 0;
    in.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!in) throw std::runtime_error("model bundle ended unexpectedly");
    return value;
}
std::uint64_t read_u64(std::istream& in) {
    std::uint64_t value = 0;
    in.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!in) throw std::runtime_error("model bundle ended unexpectedly");
    return value;
}

Tensor rotate_row(const Tensor& projected, std::size_t num_heads, std::size_t head_dim, std::size_t position) {
    if (projected.rank() != 2 || projected.shape()[0] != 1 || projected.shape()[1] != num_heads * head_dim) {
        throw std::invalid_argument("incremental rotary projection shape mismatch");
    }
    Tensor out({num_heads, head_dim});
    constexpr float base = 10000.0F;
    for (std::size_t head = 0; head < num_heads; ++head) {
        for (std::size_t pair = 0; pair < head_dim; pair += 2) {
            const float exponent = static_cast<float>(pair) / static_cast<float>(head_dim);
            const float theta = static_cast<float>(position) / std::pow(base, exponent);
            const float c = std::cos(theta);
            const float s = std::sin(theta);
            const auto src = head * head_dim + pair;
            const float x = projected.data()[src];
            const float y = projected.data()[src + 1];
            out.data()[src] = x * c - y * s;
            out.data()[src + 1] = x * s + y * c;
        }
    }
    return out;
}

Tensor split_row(const Tensor& projected, std::size_t num_heads, std::size_t head_dim) {
    if (projected.rank() != 2 || projected.shape()[0] != 1 || projected.shape()[1] != num_heads * head_dim) {
        throw std::invalid_argument("incremental projection shape mismatch");
    }
    return Tensor({num_heads, head_dim}, projected.data());
}

Tensor attention_step(
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    LayerKVCache& cache) {
    const auto heads = cache.num_heads;
    const auto dim = cache.head_dim;
    const std::size_t prior = cache.token_count();
    cache.keys.insert(cache.keys.end(), k.data().begin(), k.data().end());
    cache.values.insert(cache.values.end(), v.data().begin(), v.data().end());
    const std::size_t count = prior + 1;

    Tensor merged({1, heads * dim});
    const float scale = 1.0F / std::sqrt(static_cast<float>(dim));
    for (std::size_t head = 0; head < heads; ++head) {
        std::vector<float> scores(count, 0.0F);
        float max_score = -std::numeric_limits<float>::infinity();
        for (std::size_t t = 0; t < count; ++t) {
            float dot = 0.0F;
            for (std::size_t d = 0; d < dim; ++d) {
                const std::size_t cache_index = (t * heads + head) * dim + d;
                dot += q.data()[head * dim + d] * cache.keys[cache_index];
            }
            scores[t] = dot * scale;
            max_score = std::max(max_score, scores[t]);
        }
        float denom = 0.0F;
        for (auto& score : scores) {
            score = std::exp(score - max_score);
            denom += score;
        }
        if (!(denom > 0.0F) || !std::isfinite(denom)) {
            throw std::runtime_error("incremental attention normalization failed");
        }
        for (std::size_t d = 0; d < dim; ++d) {
            float sum = 0.0F;
            for (std::size_t t = 0; t < count; ++t) {
                const std::size_t cache_index = (t * heads + head) * dim + d;
                sum += (scores[t] / denom) * cache.values[cache_index];
            }
            merged.data()[head * dim + d] = sum;
        }
    }
    return merged;
}

} // namespace

std::size_t LayerKVCache::token_count() const noexcept {
    const std::size_t stride = num_heads * head_dim;
    return stride == 0 ? 0 : keys.size() / stride;
}

void LayerKVCache::clear() noexcept {
    keys.clear();
    values.clear();
}

InferenceSession::InferenceSession(const nn::SpiralLanguageModel& model)
    : model_(model) {
    caches_.reserve(model_.blocks().size());
    for (const auto& block : model_.blocks()) {
        const auto& attention = block->attention();
        caches_.push_back(LayerKVCache{attention.num_heads(), attention.head_dim(), {}, {}});
    }
}

void InferenceSession::reset() {
    tokens_.clear();
    for (auto& cache : caches_) cache.clear();
}

Tensor InferenceSession::prefill(std::span<const std::uint32_t> tokens) {
    if (tokens.empty()) throw std::invalid_argument("InferenceSession prefill requires tokens");
    reset();
    Tensor logits;
    for (const auto token : tokens) logits = append(token);
    return logits;
}

Tensor InferenceSession::append(std::uint32_t token) {
    if (static_cast<std::size_t>(token) >= model_.config().vocabulary_size) {
        throw std::out_of_range("InferenceSession token exceeds vocabulary");
    }
    const std::size_t position = tokens_.size();
    Tensor logits = forward_token(token, position);
    tokens_.push_back(token);
    return logits;
}

Tensor InferenceSession::forward_token(std::uint32_t token, std::size_t position) {
    const std::uint32_t id[] = {token};
    Tensor hidden = model_.token_embedding().forward(id);

    for (std::size_t layer = 0; layer < model_.blocks().size(); ++layer) {
        const auto& block = *model_.blocks()[layer];
        const Tensor normed = block.attention_norm().forward(hidden);
        const auto& attention = block.attention();
        Tensor q = rotate_row(attention.q_proj().forward(normed), attention.num_heads(), attention.head_dim(), position);
        Tensor k = rotate_row(attention.k_proj().forward(normed), attention.num_heads(), attention.head_dim(), position);
        Tensor v = split_row(attention.v_proj().forward(normed), attention.num_heads(), attention.head_dim());
        Tensor context = attention_step(q, k, v, caches_[layer]);
        Tensor attention_output = attention.out_proj().forward(context);
        Tensor residual = hidden.add(attention_output);
        Tensor ff_input = block.feed_forward_norm().forward(residual);
        Tensor ff_output = block.feed_forward().forward(ff_input);
        hidden = residual.add(ff_output);
    }

    hidden = model_.final_norm().forward(hidden);
    const Tensor logits2d = model_.lm_head().forward(hidden);
    return Tensor({model_.config().vocabulary_size}, logits2d.data());
}

void save_model_bundle(const nn::SpiralLanguageModel& model, const std::string& path) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("failed to open model bundle for writing");
    write_u32(out, bundle_magic);
    write_u32(out, bundle_version);
    const auto& c = model.config();
    write_u64(out, c.vocabulary_size);
    write_u64(out, c.model_dim);
    write_u64(out, c.num_heads);
    write_u64(out, c.num_layers);
    write_u64(out, c.ffn_hidden_dim);
    out.write(reinterpret_cast<const char*>(&c.norm_epsilon), sizeof(c.norm_epsilon));

    const auto parameters = model.parameters();
    write_u64(out, parameters.size());
    for (const auto* p : parameters) {
        write_u64(out, p->value.rank());
        for (const auto dim : p->value.shape()) write_u64(out, dim);
        write_u64(out, p->value.numel());
        out.write(reinterpret_cast<const char*>(p->value.data().data()),
                  static_cast<std::streamsize>(p->value.numel() * sizeof(float)));
    }
    if (!out) throw std::runtime_error("failed while writing model bundle");
}

LoadedModelBundle load_model_bundle(const std::string& path, std::uint64_t initialization_seed) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("failed to open model bundle for reading");
    if (read_u32(in) != bundle_magic || read_u32(in) != bundle_version) {
        throw std::runtime_error("unsupported Spiral model bundle");
    }
    nn::ModelConfig config;
    config.vocabulary_size = static_cast<std::size_t>(read_u64(in));
    config.model_dim = static_cast<std::size_t>(read_u64(in));
    config.num_heads = static_cast<std::size_t>(read_u64(in));
    config.num_layers = static_cast<std::size_t>(read_u64(in));
    config.ffn_hidden_dim = static_cast<std::size_t>(read_u64(in));
    in.read(reinterpret_cast<char*>(&config.norm_epsilon), sizeof(config.norm_epsilon));
    if (!in) throw std::runtime_error("model bundle ended before config completed");

    Random rng(initialization_seed);
    auto model = std::make_unique<nn::SpiralLanguageModel>(config, rng);
    auto parameters = model->parameters();
    if (read_u64(in) != parameters.size()) throw std::runtime_error("model bundle parameter count mismatch");
    for (auto* p : parameters) {
        if (read_u64(in) != p->value.rank()) throw std::runtime_error("model bundle parameter rank mismatch");
        for (std::size_t axis = 0; axis < p->value.rank(); ++axis) {
            if (read_u64(in) != p->value.shape()[axis]) throw std::runtime_error("model bundle parameter shape mismatch");
        }
        if (read_u64(in) != p->value.numel()) throw std::runtime_error("model bundle parameter size mismatch");
        in.read(reinterpret_cast<char*>(p->value.data().data()),
                static_cast<std::streamsize>(p->value.numel() * sizeof(float)));
        if (!in) throw std::runtime_error("model bundle ended unexpectedly");
    }
    return LoadedModelBundle{config, std::move(model)};
}

} // namespace spiral::runtime
