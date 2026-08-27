#include "spiral/temporal_generation.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <utility>

namespace spiral::temporal_generation {
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

Tensor linear_backward(nn::Linear& layer, const Tensor& input, const Tensor& grad_output) {
    if (input.rank() != 2 || grad_output.rank() != 2 || input.shape()[0] != grad_output.shape()[0] ||
        input.shape()[1] != layer.in_features() || grad_output.shape()[1] != layer.out_features()) {
        throw std::invalid_argument("temporal linear backward shape mismatch");
    }
    const std::size_t rows = input.shape()[0];
    const std::size_t in_features = layer.in_features();
    const std::size_t out_features = layer.out_features();
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
    Tensor input;
    Tensor q;
    Tensor k;
    Tensor v;
    Tensor weights;
    Tensor merged;
};

Tensor attention_forward_cached(CausalTemporalAttention& attention, const Tensor& input, AttentionCache& cache) {
    if (input.rank() != 2 || input.shape()[0] == 0 || input.shape()[1] != attention.model_dim()) {
        throw std::invalid_argument("causal attention requires [sequence,model_dim]");
    }
    cache.input = input;
    cache.q = attention.q_proj().forward(input);
    cache.k = attention.k_proj().forward(input);
    cache.v = attention.v_proj().forward(input);
    const std::size_t seq = input.shape()[0];
    const std::size_t model_dim = attention.model_dim();
    const std::size_t heads = attention.num_heads();
    const std::size_t head_dim = attention.head_dim();
    const float scale = 1.0F / std::sqrt(static_cast<float>(head_dim));
    cache.weights = Tensor({heads, seq, seq});
    cache.merged = Tensor({seq, model_dim});
    std::vector<float> scores(seq);

    for (std::size_t head = 0; head < heads; ++head) {
        const std::size_t offset = head * head_dim;
        for (std::size_t qi = 0; qi < seq; ++qi) {
            float maximum = -std::numeric_limits<float>::infinity();
            for (std::size_t ki = 0; ki <= qi; ++ki) {
                float dot = 0.0F;
                for (std::size_t d = 0; d < head_dim; ++d) {
                    dot += cache.q.data()[qi * model_dim + offset + d] * cache.k.data()[ki * model_dim + offset + d];
                }
                scores[ki] = dot * scale;
                maximum = std::max(maximum, scores[ki]);
            }
            float sum = 0.0F;
            const std::size_t base = (head * seq + qi) * seq;
            for (std::size_t ki = 0; ki <= qi; ++ki) {
                const float e = std::exp(scores[ki] - maximum);
                cache.weights.data()[base + ki] = e;
                sum += e;
            }
            if (!(sum > 0.0F) || !std::isfinite(sum)) throw std::runtime_error("causal attention softmax failed");
            for (std::size_t ki = 0; ki <= qi; ++ki) cache.weights.data()[base + ki] /= sum;
            for (std::size_t d = 0; d < head_dim; ++d) {
                float mixed = 0.0F;
                for (std::size_t ki = 0; ki <= qi; ++ki) {
                    mixed += cache.weights.data()[base + ki] * cache.v.data()[ki * model_dim + offset + d];
                }
                cache.merged.data()[qi * model_dim + offset + d] = mixed;
            }
        }
    }
    return attention.out_proj().forward(cache.merged);
}

Tensor attention_backward(CausalTemporalAttention& attention, const AttentionCache& cache, const Tensor& grad_output) {
    const std::size_t seq = cache.input.shape()[0];
    const std::size_t model_dim = attention.model_dim();
    const std::size_t heads = attention.num_heads();
    const std::size_t head_dim = attention.head_dim();
    const float scale = 1.0F / std::sqrt(static_cast<float>(head_dim));
    const Tensor grad_merged = linear_backward(attention.out_proj(), cache.merged, grad_output);
    Tensor grad_weights({heads, seq, seq});
    Tensor grad_v({seq, model_dim});

    for (std::size_t head = 0; head < heads; ++head) {
        const std::size_t offset = head * head_dim;
        for (std::size_t qi = 0; qi < seq; ++qi) {
            const std::size_t base = (head * seq + qi) * seq;
            for (std::size_t ki = 0; ki <= qi; ++ki) {
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

    Tensor grad_scores({heads, seq, seq});
    for (std::size_t head = 0; head < heads; ++head) {
        for (std::size_t qi = 0; qi < seq; ++qi) {
            const std::size_t base = (head * seq + qi) * seq;
            float weighted = 0.0F;
            for (std::size_t ki = 0; ki <= qi; ++ki) weighted += grad_weights.data()[base + ki] * cache.weights.data()[base + ki];
            for (std::size_t ki = 0; ki <= qi; ++ki) {
                const float w = cache.weights.data()[base + ki];
                grad_scores.data()[base + ki] = w * (grad_weights.data()[base + ki] - weighted);
            }
        }
    }

    Tensor grad_q({seq, model_dim});
    Tensor grad_k({seq, model_dim});
    for (std::size_t head = 0; head < heads; ++head) {
        const std::size_t offset = head * head_dim;
        for (std::size_t qi = 0; qi < seq; ++qi) {
            const std::size_t base = (head * seq + qi) * seq;
            for (std::size_t ki = 0; ki <= qi; ++ki) {
                const float gs = grad_scores.data()[base + ki] * scale;
                for (std::size_t d = 0; d < head_dim; ++d) {
                    grad_q.data()[qi * model_dim + offset + d] += gs * cache.k.data()[ki * model_dim + offset + d];
                    grad_k.data()[ki * model_dim + offset + d] += gs * cache.q.data()[qi * model_dim + offset + d];
                }
            }
        }
    }

    Tensor grad_input = linear_backward(attention.q_proj(), cache.input, grad_q);
    add_inplace(grad_input, linear_backward(attention.k_proj(), cache.input, grad_k));
    add_inplace(grad_input, linear_backward(attention.v_proj(), cache.input, grad_v));
    return grad_input;
}

struct BlockCache {
    Tensor input;
    AttentionCache attention;
    Tensor after_attention;
    Tensor ffn_pre;
    Tensor ffn_active;
};

Tensor block_forward_cached(CausalTemporalBlock& block, const Tensor& input, BlockCache& cache) {
    cache.input = input;
    cache.after_attention = add_tensors(input, attention_forward_cached(block.attention(), input, cache.attention));
    cache.ffn_pre = block.ffn_in().forward(cache.after_attention);
    cache.ffn_active = silu(cache.ffn_pre);
    return add_tensors(cache.after_attention, block.ffn_out().forward(cache.ffn_active));
}

Tensor block_backward(CausalTemporalBlock& block, const BlockCache& cache, const Tensor& grad_output) {
    Tensor grad_after_attention = grad_output;
    const Tensor grad_active = linear_backward(block.ffn_out(), cache.ffn_active, grad_output);
    const Tensor grad_pre = silu_backward(cache.ffn_pre, grad_active);
    add_inplace(grad_after_attention, linear_backward(block.ffn_in(), cache.after_attention, grad_pre));
    Tensor grad_input = grad_after_attention;
    add_inplace(grad_input, attention_backward(block.attention(), cache.attention, grad_after_attention));
    return grad_input;
}

struct ModelCache {
    Tensor input;
    Tensor projected;
    std::vector<BlockCache> blocks;
    Tensor hidden;
};

Tensor model_forward_cached(CausalTemporalPredictor& model, const Tensor& sequence, ModelCache& cache) {
    const auto& config = model.config();
    if (sequence.rank() != 2 || sequence.shape()[0] == 0 || sequence.shape()[1] != config.input_dim) {
        throw std::invalid_argument("causal temporal model requires [sequence,input_dim]");
    }
    cache.input = sequence;
    cache.projected = model.input_projection().forward(sequence);
    temporal::add_temporal_sincos_position(cache.projected);
    cache.blocks.resize(config.num_layers);
    Tensor hidden = cache.projected;
    for (std::size_t i = 0; i < config.num_layers; ++i) hidden = block_forward_cached(model.block(i), hidden, cache.blocks[i]);
    cache.hidden = hidden;
    return model.output_projection().forward(hidden);
}

float next_latent_loss(const Tensor& prediction, const Tensor& sequence) {
    if (prediction.rank() != 2 || sequence.rank() != 2 || sequence.shape()[0] < 2 ||
        prediction.shape()[0] != sequence.shape()[0] || prediction.shape()[1] != sequence.shape()[1]) {
        throw std::invalid_argument("next-latent loss requires equal [sequence,dim] tensors with sequence >= 2");
    }
    const std::size_t rows = sequence.shape()[0] - 1;
    const std::size_t dim = sequence.shape()[1];
    double sum = 0.0;
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t d = 0; d < dim; ++d) {
            const double delta = static_cast<double>(prediction.data()[row * dim + d]) -
                                 static_cast<double>(sequence.data()[(row + 1) * dim + d]);
            sum += delta * delta;
        }
    }
    return static_cast<float>(sum / static_cast<double>(rows * dim));
}

Tensor next_latent_gradient(const Tensor& prediction, const Tensor& sequence) {
    Tensor grad(prediction.shape());
    const std::size_t rows = sequence.shape()[0] - 1;
    const std::size_t dim = sequence.shape()[1];
    const float scale = 2.0F / static_cast<float>(rows * dim);
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t d = 0; d < dim; ++d) {
            grad.data()[row * dim + d] = scale * (prediction.data()[row * dim + d] - sequence.data()[(row + 1) * dim + d]);
        }
    }
    return grad;
}

Tensor append_context_row(const Tensor& context, const Tensor& row, std::size_t max_context) {
    if (context.rank() != 2 || row.rank() != 1 || context.shape()[1] != row.shape()[0]) {
        throw std::invalid_argument("append_context_row shape mismatch");
    }
    const std::size_t dim = context.shape()[1];
    const std::size_t old_rows = context.shape()[0];
    const std::size_t keep_old = std::min(old_rows, max_context > 0 ? max_context - 1 : old_rows);
    const std::size_t start = old_rows - keep_old;
    Tensor out({keep_old + 1, dim});
    for (std::size_t r = 0; r < keep_old; ++r) {
        std::copy_n(context.data().begin() + static_cast<std::ptrdiff_t>((start + r) * dim), dim,
                    out.data().begin() + static_cast<std::ptrdiff_t>(r * dim));
    }
    std::copy(row.data().begin(), row.data().end(), out.data().begin() + static_cast<std::ptrdiff_t>(keep_old * dim));
    return out;
}

void write_u64(std::ofstream& out, std::uint64_t value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

std::uint64_t read_u64(std::ifstream& in) {
    std::uint64_t value = 0;
    in.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!in) throw std::runtime_error("truncated temporal checkpoint");
    return value;
}

} // namespace

CausalTemporalAttention::CausalTemporalAttention(std::size_t model_dim, std::size_t num_heads, Random& rng)
    : model_dim_(model_dim), num_heads_(num_heads), head_dim_(num_heads == 0 ? 0 : model_dim / num_heads),
      q_proj_(model_dim, model_dim, rng, false), k_proj_(model_dim, model_dim, rng, false),
      v_proj_(model_dim, model_dim, rng, false), out_proj_(model_dim, model_dim, rng, false) {
    if (model_dim_ == 0 || num_heads_ == 0 || model_dim_ % num_heads_ != 0) {
        throw std::invalid_argument("invalid causal temporal attention dimensions");
    }
}

Tensor CausalTemporalAttention::forward(const Tensor& tokens) const {
    if (tokens.rank() != 2 || tokens.shape()[0] == 0 || tokens.shape()[1] != model_dim_) {
        throw std::invalid_argument("causal temporal attention requires [sequence,model_dim]");
    }
    const Tensor q = q_proj_.forward(tokens);
    const Tensor k = k_proj_.forward(tokens);
    const Tensor v = v_proj_.forward(tokens);
    const std::size_t seq = tokens.shape()[0];
    const float scale = 1.0F / std::sqrt(static_cast<float>(head_dim_));
    Tensor merged({seq, model_dim_});
    std::vector<float> scores(seq);
    std::vector<float> weights(seq);
    for (std::size_t head = 0; head < num_heads_; ++head) {
        const std::size_t offset = head * head_dim_;
        for (std::size_t qi = 0; qi < seq; ++qi) {
            float maximum = -std::numeric_limits<float>::infinity();
            for (std::size_t ki = 0; ki <= qi; ++ki) {
                float dot = 0.0F;
                for (std::size_t d = 0; d < head_dim_; ++d) {
                    dot += q.data()[qi * model_dim_ + offset + d] * k.data()[ki * model_dim_ + offset + d];
                }
                scores[ki] = dot * scale;
                maximum = std::max(maximum, scores[ki]);
            }
            float sum = 0.0F;
            for (std::size_t ki = 0; ki <= qi; ++ki) {
                weights[ki] = std::exp(scores[ki] - maximum);
                sum += weights[ki];
            }
            for (std::size_t ki = 0; ki <= qi; ++ki) weights[ki] /= sum;
            for (std::size_t d = 0; d < head_dim_; ++d) {
                float value = 0.0F;
                for (std::size_t ki = 0; ki <= qi; ++ki) value += weights[ki] * v.data()[ki * model_dim_ + offset + d];
                merged.data()[qi * model_dim_ + offset + d] = value;
            }
        }
    }
    return out_proj_.forward(merged);
}

std::vector<nn::Parameter*> CausalTemporalAttention::parameters() {
    std::vector<nn::Parameter*> out;
    append_parameters(out, q_proj_.parameters()); append_parameters(out, k_proj_.parameters());
    append_parameters(out, v_proj_.parameters()); append_parameters(out, out_proj_.parameters());
    return out;
}

std::vector<const nn::Parameter*> CausalTemporalAttention::parameters() const {
    std::vector<const nn::Parameter*> out;
    append_parameters(out, q_proj_.parameters()); append_parameters(out, k_proj_.parameters());
    append_parameters(out, v_proj_.parameters()); append_parameters(out, out_proj_.parameters());
    return out;
}

CausalTemporalBlock::CausalTemporalBlock(std::size_t model_dim, std::size_t num_heads, std::size_t ffn_dim, Random& rng)
    : attention_(model_dim, num_heads, rng), ffn_in_(model_dim, ffn_dim, rng, true), ffn_out_(ffn_dim, model_dim, rng, true) {
    if (ffn_dim == 0) throw std::invalid_argument("temporal FFN dimension must be non-zero");
}

Tensor CausalTemporalBlock::forward(const Tensor& tokens) const {
    const Tensor after_attention = add_tensors(tokens, attention_.forward(tokens));
    return add_tensors(after_attention, ffn_out_.forward(silu(ffn_in_.forward(after_attention))));
}

std::vector<nn::Parameter*> CausalTemporalBlock::parameters() {
    std::vector<nn::Parameter*> out;
    append_parameters(out, attention_.parameters()); append_parameters(out, ffn_in_.parameters()); append_parameters(out, ffn_out_.parameters());
    return out;
}

std::vector<const nn::Parameter*> CausalTemporalBlock::parameters() const {
    std::vector<const nn::Parameter*> out;
    append_parameters(out, attention_.parameters()); append_parameters(out, ffn_in_.parameters()); append_parameters(out, ffn_out_.parameters());
    return out;
}

CausalTemporalPredictor::CausalTemporalPredictor(CausalTemporalConfig config, Random& rng)
    : config_(config), input_projection_(config.input_dim, config.model_dim, rng, true), output_projection_(config.model_dim, config.output_dim, rng, true) {
    if (config_.input_dim == 0 || config_.model_dim == 0 || config_.num_heads == 0 || config_.num_layers == 0 ||
        config_.ffn_dim == 0 || config_.output_dim == 0 || config_.max_context == 0 || config_.model_dim % config_.num_heads != 0) {
        throw std::invalid_argument("invalid CausalTemporalConfig");
    }
    blocks_.reserve(config_.num_layers);
    for (std::size_t i = 0; i < config_.num_layers; ++i) blocks_.emplace_back(config_.model_dim, config_.num_heads, config_.ffn_dim, rng);
}

Tensor CausalTemporalPredictor::predict_all(const Tensor& ordered_features) const {
    if (ordered_features.rank() != 2 || ordered_features.shape()[0] == 0 || ordered_features.shape()[1] != config_.input_dim) {
        throw std::invalid_argument("predict_all requires [sequence,input_dim]");
    }
    Tensor hidden = input_projection_.forward(ordered_features);
    temporal::add_temporal_sincos_position(hidden);
    for (const auto& block : blocks_) hidden = block.forward(hidden);
    return output_projection_.forward(hidden);
}

Tensor CausalTemporalPredictor::predict_next(const Tensor& context) const {
    const Tensor all = predict_all(context);
    Tensor next({config_.output_dim});
    const std::size_t base = (all.shape()[0] - 1) * config_.output_dim;
    std::copy_n(all.data().begin() + static_cast<std::ptrdiff_t>(base), config_.output_dim, next.data().begin());
    return next;
}

Tensor CausalTemporalPredictor::generate(const Tensor& seed_context, std::size_t steps) const {
    if (config_.input_dim != config_.output_dim) throw std::invalid_argument("autoregressive generation requires input_dim == output_dim");
    if (seed_context.rank() != 2 || seed_context.shape()[0] == 0 || seed_context.shape()[1] != config_.input_dim) {
        throw std::invalid_argument("generate requires non-empty [sequence,input_dim] seed");
    }
    Tensor generated({steps, config_.output_dim});
    Tensor context = seed_context;
    if (context.shape()[0] > config_.max_context) {
        const std::size_t start = context.shape()[0] - config_.max_context;
        Tensor trimmed({config_.max_context, config_.input_dim});
        std::copy(context.data().begin() + static_cast<std::ptrdiff_t>(start * config_.input_dim), context.data().end(), trimmed.data().begin());
        context = std::move(trimmed);
    }
    for (std::size_t step = 0; step < steps; ++step) {
        const Tensor next = predict_next(context);
        std::copy(next.data().begin(), next.data().end(), generated.data().begin() + static_cast<std::ptrdiff_t>(step * config_.output_dim));
        context = append_context_row(context, next, config_.max_context);
    }
    return generated;
}

std::vector<nn::Parameter*> CausalTemporalPredictor::parameters() {
    std::vector<nn::Parameter*> out;
    append_parameters(out, input_projection_.parameters());
    for (auto& block : blocks_) append_parameters(out, block.parameters());
    append_parameters(out, output_projection_.parameters());
    return out;
}

std::vector<const nn::Parameter*> CausalTemporalPredictor::parameters() const {
    std::vector<const nn::Parameter*> out;
    append_parameters(out, input_projection_.parameters());
    for (const auto& block : blocks_) append_parameters(out, block.parameters());
    append_parameters(out, output_projection_.parameters());
    return out;
}

CausalTemporalBlock& CausalTemporalPredictor::block(std::size_t index) {
    if (index >= blocks_.size()) throw std::out_of_range("temporal block index out of range");
    return blocks_[index];
}

TemporalNextLatentTrainer::TemporalNextLatentTrainer(CausalTemporalPredictor& model, TemporalTrainerConfig config)
    : model_(model), config_(config), optimizer_(model.parameters(), config.optimizer) {
    if (model_.config().input_dim != model_.config().output_dim) throw std::invalid_argument("next-latent training requires equal input/output dimensions");
}

float TemporalNextLatentTrainer::evaluate(const Tensor& sequence) const {
    return next_latent_loss(model_.predict_all(sequence), sequence);
}

float TemporalNextLatentTrainer::train_step(const Tensor& sequence) {
    optimizer_.zero_grad();
    ModelCache cache;
    const Tensor prediction = model_forward_cached(model_, sequence, cache);
    const float loss = next_latent_loss(prediction, sequence);
    Tensor grad = next_latent_gradient(prediction, sequence);
    grad = linear_backward(model_.output_projection(), cache.hidden, grad);
    for (std::size_t i = model_.config().num_layers; i-- > 0;) grad = block_backward(model_.block(i), cache.blocks[i], grad);
    (void)linear_backward(model_.input_projection(), cache.input, grad);
    auto params = model_.parameters();
    train::clip_grad_norm(params, config_.max_grad_norm);
    optimizer_.step();
    return loss;
}

audio::AudioBuffer magnitude_to_audio_zero_phase(const Tensor& magnitude, std::uint32_t sample_rate, audio::StftConfig config) {
    if (magnitude.rank() != 2 || magnitude.shape()[0] == 0 || !dsp::is_power_of_two(config.frame_size) ||
        config.hop_size == 0 || config.hop_size > config.frame_size || magnitude.shape()[1] != config.frame_size / 2 + 1) {
        throw std::invalid_argument("invalid magnitude spectrogram or STFT config");
    }
    const std::size_t frames = magnitude.shape()[0];
    const std::size_t bins = magnitude.shape()[1];
    const std::size_t length = (frames - 1) * config.hop_size + config.frame_size;
    std::vector<float> output(length, 0.0F);
    std::vector<float> normalization(length, 0.0F);
    const float denom = static_cast<float>(config.frame_size - 1);
    for (std::size_t frame = 0; frame < frames; ++frame) {
        std::vector<std::complex<float>> spectrum(config.frame_size, {0.0F, 0.0F});
        for (std::size_t bin = 0; bin < bins; ++bin) spectrum[bin] = {std::max(0.0F, magnitude.data()[frame * bins + bin]), 0.0F};
        for (std::size_t bin = 1; bin + 1 < bins; ++bin) spectrum[config.frame_size - bin] = spectrum[bin];
        dsp::fft_inplace(spectrum, true);
        const std::size_t start = frame * config.hop_size;
        for (std::size_t i = 0; i < config.frame_size; ++i) {
            const float window = 0.5F - 0.5F * std::cos(2.0F * std::numbers::pi_v<float> * static_cast<float>(i) / denom);
            output[start + i] += spectrum[i].real() * window;
            normalization[start + i] += window * window;
        }
    }
    for (std::size_t i = 0; i < length; ++i) {
        if (normalization[i] > 1.0e-8F) output[i] /= normalization[i];
        if (!std::isfinite(output[i])) output[i] = 0.0F;
        output[i] = std::clamp(output[i], -1.0F, 1.0F);
    }
    return audio::AudioBuffer(sample_rate, 1, std::move(output));
}

AudioLatentGenerator::AudioLatentGenerator(AudioGenerationConfig config, const audio::AudioLatentCodec& codec, const CausalTemporalPredictor& predictor)
    : config_(config), codec_(&codec), predictor_(&predictor) {
    if (!dsp::is_power_of_two(config_.stft.frame_size) || config_.stft.hop_size == 0 || config_.frames_per_patch == 0) {
        throw std::invalid_argument("invalid AudioGenerationConfig");
    }
    if (codec.config().latent_dim != predictor.config().input_dim || predictor.config().input_dim != predictor.config().output_dim) {
        throw std::invalid_argument("audio codec latent dimension must match temporal predictor dimensions");
    }
    const std::size_t bins = config_.stft.frame_size / 2 + 1;
    if (codec.config().patch_dim != bins * config_.frames_per_patch) throw std::invalid_argument("audio codec patch dimension does not match STFT patching");
}

Tensor AudioLatentGenerator::encode_latents(const audio::AudioBuffer& input) const {
    const Tensor spectrum = dsp::stft_magnitude_fft(input, config_.stft);
    if (spectrum.shape()[0] == 0) throw std::invalid_argument("seed audio is shorter than one STFT frame");
    return codec_->encode(audio::spectral_patches(spectrum, config_.frames_per_patch));
}

Tensor AudioLatentGenerator::generate_latents(const audio::AudioBuffer& seed_audio, std::size_t steps) const {
    return predictor_->generate(encode_latents(seed_audio), steps);
}

audio::AudioBuffer AudioLatentGenerator::synthesize(const Tensor& latents, std::uint32_t sample_rate) const {
    if (latents.rank() != 2 || latents.shape()[0] == 0 || latents.shape()[1] != codec_->config().latent_dim) {
        throw std::invalid_argument("audio synthesize requires non-empty latent matrix");
    }
    const Tensor patches = codec_->decode(latents);
    const std::size_t bins = config_.stft.frame_size / 2 + 1;
    Tensor magnitude({patches.shape()[0] * config_.frames_per_patch, bins});
    for (std::size_t patch = 0; patch < patches.shape()[0]; ++patch) {
        for (std::size_t local = 0; local < config_.frames_per_patch; ++local) {
            for (std::size_t bin = 0; bin < bins; ++bin) {
                magnitude.data()[(patch * config_.frames_per_patch + local) * bins + bin] =
                    std::max(0.0F, patches.data()[patch * codec_->config().patch_dim + local * bins + bin]);
            }
        }
    }
    return magnitude_to_audio_zero_phase(magnitude, sample_rate, config_.stft);
}

VideoEmbeddingGenerator::VideoEmbeddingGenerator(const vision::VisionEncoder& vision_encoder, const CausalTemporalPredictor& predictor)
    : vision_encoder_(&vision_encoder), predictor_(&predictor) {
    if (vision_encoder.config().embedding_dim != predictor.config().input_dim || predictor.config().input_dim != predictor.config().output_dim) {
        throw std::invalid_argument("vision embedding dimension must match temporal predictor dimensions");
    }
}

Tensor VideoEmbeddingGenerator::frame_embeddings(const temporal::VideoFrameSequence& sequence) const {
    if (sequence.size() == 0) throw std::invalid_argument("video sequence cannot be empty");
    const std::size_t dim = vision_encoder_->config().embedding_dim;
    Tensor out({sequence.size(), dim});
    for (std::size_t frame = 0; frame < sequence.size(); ++frame) {
        const Tensor embedding = vision_encoder_->encode_pooled(sequence.at(frame));
        std::copy(embedding.data().begin(), embedding.data().end(), out.data().begin() + static_cast<std::ptrdiff_t>(frame * dim));
    }
    return out;
}

Tensor VideoEmbeddingGenerator::predict_next(const temporal::VideoFrameSequence& sequence) const {
    return predictor_->predict_next(frame_embeddings(sequence));
}

Tensor VideoEmbeddingGenerator::generate(const temporal::VideoFrameSequence& sequence, std::size_t steps) const {
    return predictor_->generate(frame_embeddings(sequence), steps);
}

void save_temporal_predictor(const CausalTemporalPredictor& model, const std::string& path) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("failed to open temporal checkpoint for writing");
    constexpr std::array<char, 8> magic{'S','P','T','E','M','P','1','5'};
    out.write(magic.data(), static_cast<std::streamsize>(magic.size()));
    const auto& c = model.config();
    for (const std::size_t value : {c.input_dim, c.model_dim, c.num_heads, c.num_layers, c.ffn_dim, c.output_dim, c.max_context}) write_u64(out, value);
    const auto params = model.parameters();
    write_u64(out, params.size());
    for (const auto* parameter : params) {
        write_u64(out, parameter->value.rank());
        for (const auto dim : parameter->value.shape()) write_u64(out, dim);
        write_u64(out, parameter->value.numel());
        out.write(reinterpret_cast<const char*>(parameter->value.data().data()), static_cast<std::streamsize>(parameter->value.numel() * sizeof(float)));
    }
    if (!out) throw std::runtime_error("failed while writing temporal checkpoint");
}

void load_temporal_predictor(CausalTemporalPredictor& model, const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("failed to open temporal checkpoint");
    std::array<char, 8> magic{};
    in.read(magic.data(), static_cast<std::streamsize>(magic.size()));
    constexpr std::array<char, 8> expected{'S','P','T','E','M','P','1','5'};
    if (!in || magic != expected) throw std::runtime_error("invalid temporal checkpoint magic");
    const auto& c = model.config();
    const std::array<std::size_t, 7> expected_config{c.input_dim, c.model_dim, c.num_heads, c.num_layers, c.ffn_dim, c.output_dim, c.max_context};
    for (const auto value : expected_config) if (read_u64(in) != value) throw std::runtime_error("temporal checkpoint config mismatch");
    auto params = model.parameters();
    if (read_u64(in) != params.size()) throw std::runtime_error("temporal checkpoint parameter count mismatch");
    for (auto* parameter : params) {
        const std::size_t rank = static_cast<std::size_t>(read_u64(in));
        if (rank != parameter->value.rank()) throw std::runtime_error("temporal checkpoint rank mismatch");
        for (std::size_t i = 0; i < rank; ++i) if (read_u64(in) != parameter->value.shape()[i]) throw std::runtime_error("temporal checkpoint shape mismatch");
        if (read_u64(in) != parameter->value.numel()) throw std::runtime_error("temporal checkpoint tensor size mismatch");
        in.read(reinterpret_cast<char*>(parameter->value.data().data()), static_cast<std::streamsize>(parameter->value.numel() * sizeof(float)));
        if (!in) throw std::runtime_error("truncated temporal checkpoint tensor");
    }
}

} // namespace spiral::temporal_generation
