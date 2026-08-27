#include "spiral/generate.hpp"
#include "spiral/train.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

int main() {
    using spiral::Random;
    using spiral::Tensor;
    using namespace spiral::generate;

    const Tensor logits({4}, {1.0F, 4.0F, 3.0F, 2.0F});
    SamplingConfig greedy;
    greedy.temperature = 0.0F;
    greedy.top_k = 0;
    greedy.top_p = 1.0F;
    Random greedy_rng(1);
    assert(sample_token(logits, {}, greedy, greedy_rng) == 1U);

    SamplingConfig repeat = greedy;
    repeat.repetition_penalty = 10.0F;
    const std::uint32_t repeated[] = {1U};
    assert(sample_token(logits, repeated, repeat, greedy_rng) == 2U);

    SamplingConfig stochastic;
    stochastic.temperature = 1.0F;
    stochastic.top_k = 2;
    stochastic.top_p = 1.0F;
    Random a(77), b(77);
    for (int i = 0; i < 20; ++i) {
        const auto x = sample_token(logits, {}, stochastic, a);
        const auto y = sample_token(logits, {}, stochastic, b);
        assert(x == y);
        assert(x == 1U || x == 2U);
    }

    const std::vector<std::uint32_t> history{spiral::ByteTokenizer::bos_token, 10U, 11U, 12U, 13U, 14U};
    const auto context = build_context(history, 4, true);
    assert((context == std::vector<std::uint32_t>{spiral::ByteTokenizer::bos_token, 12U, 13U, 14U}));
    assert(is_stop_token(7U, std::vector<std::uint32_t>{6U, 7U}));

    const std::vector<std::uint32_t> corpus = {1,2,3,4,5,1,2,3,4,5,1,2,3,4,5};
    spiral::train::TokenDataset dataset(corpus, 4);
    spiral::nn::ModelConfig model_config{8, 4, 2, 1, 8, 1.0e-5F};
    Random train_rng(2026);
    spiral::nn::SpiralLanguageModel model(model_config, train_rng);

    spiral::train::TrainerConfig trainer_config;
    trainer_config.optimizer.learning_rate = 0.01F;
    trainer_config.optimizer.weight_decay = 0.0F;
    trainer_config.max_grad_norm = 1.0F;
    spiral::train::LanguageModelTrainer trainer(model, trainer_config);
    const auto& example = dataset.at(0);
    for (int step = 0; step < 80; ++step) {
        assert(std::isfinite(trainer.train_step(example.input, example.target)));
    }

    TokenGenerator generator(model);
    GenerationConfig generation;
    generation.sampling.temperature = 0.0F;
    generation.sampling.top_k = 0;
    generation.sampling.top_p = 1.0F;
    generation.max_new_tokens = 1;
    generation.max_context_tokens = 8;
    generation.stop_tokens.clear();
    const auto result = generator.generate(example.input, generation);
    assert(result.generated_tokens.size() == 1);
    assert(result.generated_tokens.front() == 5U);

    generation.max_new_tokens = 4;
    std::size_t callbacks = 0;
    const auto cancelled = generator.generate(example.input, generation, [&](std::uint32_t, std::size_t) {
        ++callbacks;
        return false;
    });
    assert(cancelled.cancelled);
    assert(cancelled.generated_tokens.size() == 1);
    assert(callbacks == 1);

    std::cout << "spiral_generate_tests: PASS next=" << result.generated_tokens.front() << '\n';
    return 0;
}
