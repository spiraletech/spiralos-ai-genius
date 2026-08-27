#include "spiral/model.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

bool near(float a, float b, float epsilon = 1.0e-5F) {
    return std::fabs(a - b) <= epsilon;
}

} // namespace

int main() {
    const spiral::nn::ModelConfig config{
        .vocabulary_size = 32,
        .model_dim = 8,
        .num_heads = 2,
        .num_layers = 2,
        .ffn_hidden_dim = 24,
        .norm_epsilon = 1.0e-5F,
    };

    spiral::Random rng_a(0x12345678ULL);
    spiral::nn::SpiralLanguageModel model_a(config, rng_a);

    const std::vector<std::uint32_t> tokens{1, 2, 3, 4};
    const auto logits = model_a.forward(tokens);
    assert(logits.shape() == std::vector<std::size_t>({4, 32}));
    assert(logits.numel() == 128);

    const auto last = model_a.last_token_logits(tokens);
    assert(last.shape() == std::vector<std::size_t>({32}));
    for (std::size_t i = 0; i < 32; ++i) {
        assert(near(last.data()[i], logits.data()[3 * 32 + i]));
    }

    const auto params = model_a.parameters();
    assert(params.size() == 21);
    assert(model_a.parameter_count() == 2216);

    spiral::Random rng_b(0x12345678ULL);
    spiral::nn::SpiralLanguageModel model_b(config, rng_b);
    const auto repeated = model_b.forward(tokens);
    assert(repeated.shape() == logits.shape());
    for (std::size_t i = 0; i < logits.numel(); ++i) {
        assert(near(repeated.data()[i], logits.data()[i]));
    }

    const std::vector<std::uint32_t> future_a{7, 8, 9, 10};
    const std::vector<std::uint32_t> future_b{7, 20, 21, 22};
    const auto causal_a = model_a.forward(future_a);
    const auto causal_b = model_a.forward(future_b);
    for (std::size_t vocab = 0; vocab < config.vocabulary_size; ++vocab) {
        assert(near(causal_a.data()[vocab], causal_b.data()[vocab]));
    }

    bool rejected_empty = false;
    try {
        const std::vector<std::uint32_t> empty;
        (void)model_a.forward(empty);
    } catch (const std::invalid_argument&) {
        rejected_empty = true;
    }
    assert(rejected_empty);

    std::cout << "spiral_model_tests: PASS\n";
    return 0;
}
