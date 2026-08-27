#include "train_detail.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace spiral::train::detail {

Tensor feed_forward_forward(nn::GatedFeedForward& ffn, const Tensor& input, FeedForwardCache& cache) {
    cache.input = input;
    cache.gate = ffn.gate_proj().forward(input);
    cache.up = ffn.up_proj().forward(input);
    cache.mixed = Tensor(cache.gate.shape());
    for (std::size_t i = 0; i < cache.mixed.numel(); ++i) {
        cache.mixed.data()[i] = silu(cache.gate.data()[i]) * cache.up.data()[i];
    }
    return ffn.down_proj().forward(cache.mixed);
}

Tensor feed_forward_backward(nn::GatedFeedForward& ffn, const FeedForwardCache& cache, const Tensor& grad_output) {
    const Tensor grad_mixed = linear_backward(ffn.down_proj(), cache.mixed, grad_output);
    Tensor grad_gate(cache.gate.shape());
    Tensor grad_up(cache.up.shape());
    for (std::size_t i = 0; i < grad_mixed.numel(); ++i) {
        grad_gate.data()[i] = grad_mixed.data()[i] * cache.up.data()[i] * silu_derivative(cache.gate.data()[i]);
        grad_up.data()[i] = grad_mixed.data()[i] * silu(cache.gate.data()[i]);
    }
    Tensor grad_input = linear_backward(ffn.gate_proj(), cache.input, grad_gate);
    add_inplace(grad_input, linear_backward(ffn.up_proj(), cache.input, grad_up));
    return grad_input;
}

Tensor block_forward(nn::TransformerBlock& block, const Tensor& input, BlockCache& cache) {
    cache.input = input;
    cache.attention_input = block.attention_norm().forward(input);
    const Tensor attention_output = attention_forward(block.attention(), cache.attention_input, cache.attention);
    cache.residual = input.add(attention_output);
    cache.ffn_input = block.feed_forward_norm().forward(cache.residual);
    const Tensor ffn_output = feed_forward_forward(block.feed_forward(), cache.ffn_input, cache.ffn);
    cache.output = cache.residual.add(ffn_output);
    return cache.output;
}

Tensor block_backward(nn::TransformerBlock& block, const BlockCache& cache, const Tensor& grad_output) {
    Tensor grad_residual = grad_output;
    const Tensor grad_ffn_input = feed_forward_backward(block.feed_forward(), cache.ffn, grad_output);
    add_inplace(grad_residual, rmsnorm_backward(block.feed_forward_norm(), cache.residual, grad_ffn_input));

    Tensor grad_input = grad_residual;
    const Tensor grad_attention_input = attention_backward(block.attention(), cache.attention, grad_residual);
    add_inplace(grad_input, rmsnorm_backward(block.attention_norm(), cache.input, grad_attention_input));
    return grad_input;
}

Tensor logits_cross_entropy_gradient(
    const Tensor& logits,
    std::span<const std::uint32_t> targets,
    float& loss_out) {
    if (logits.rank() != 2 || logits.shape()[0] != targets.size()) {
        throw std::invalid_argument("cross entropy logits/targets shape mismatch");
    }
    const std::size_t rows = logits.shape()[0];
    const std::size_t vocabulary = logits.shape()[1];
    if (rows == 0 || vocabulary == 0) throw std::invalid_argument("cross entropy requires non-empty logits");

    Tensor gradient(logits.shape());
    double loss = 0.0;
    for (std::size_t row = 0; row < rows; ++row) {
        const auto target = static_cast<std::size_t>(targets[row]);
        if (target >= vocabulary) throw std::out_of_range("cross entropy target exceeds vocabulary");
        const std::size_t base = row * vocabulary;
        float max_value = logits.data()[base];
        for (std::size_t col = 1; col < vocabulary; ++col) {
            max_value = std::max(max_value, logits.data()[base + col]);
        }

        double sum = 0.0;
        for (std::size_t col = 0; col < vocabulary; ++col) {
            sum += std::exp(static_cast<double>(logits.data()[base + col] - max_value));
        }
        const double logsumexp = static_cast<double>(max_value) + std::log(sum);
        loss += logsumexp - static_cast<double>(logits.data()[base + target]);

        for (std::size_t col = 0; col < vocabulary; ++col) {
            const float probability = static_cast<float>(
                std::exp(static_cast<double>(logits.data()[base + col] - max_value)) / sum);
            gradient.data()[base + col] = probability;
        }
        gradient.data()[base + target] -= 1.0F;
    }

    const float inv_rows = 1.0F / static_cast<float>(rows);
    for (auto& value : gradient.data()) value *= inv_rows;
    loss_out = static_cast<float>(loss / static_cast<double>(rows));
    return gradient;
}

} // namespace spiral::train::detail
