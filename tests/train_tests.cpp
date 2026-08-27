#include "spiral/model.hpp"
#include "spiral/train.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <vector>

namespace {
bool near(float a, float b, float epsilon = 1.0e-5F) { return std::fabs(a - b) <= epsilon; }
}

int main() {
    using spiral::Random;
    using spiral::nn::ModelConfig;
    using spiral::nn::SpiralLanguageModel;
    using spiral::train::LanguageModelTrainer;
    using spiral::train::TokenDataset;
    using spiral::train::TrainerConfig;

    const std::vector<std::uint32_t> corpus = {1,2,3,4,5,1,2,3,4,5,1,2,3,4,5};
    TokenDataset dataset(corpus, 4);
    assert(dataset.size() == corpus.size() - 4);
    assert(dataset.at(0).input == std::vector<std::uint32_t>({1,2,3,4}));
    assert(dataset.at(0).target == std::vector<std::uint32_t>({2,3,4,5}));

    ModelConfig config{8, 4, 2, 1, 8, 1.0e-5F};
    Random rng(2026);
    SpiralLanguageModel model(config, rng);

    TrainerConfig trainer_config;
    trainer_config.optimizer.learning_rate = 0.01F;
    trainer_config.optimizer.weight_decay = 0.0F;
    trainer_config.max_grad_norm = 1.0F;
    LanguageModelTrainer trainer(model, trainer_config);

    const auto& example = dataset.at(0);
    const float initial_loss = trainer.evaluate(example.input, example.target);
    assert(std::isfinite(initial_loss));

    const float first_step_loss = trainer.train_step(example.input, example.target);
    assert(near(first_step_loss, initial_loss, 1.0e-4F));
    for (int step = 1; step < 80; ++step) {
        const float step_loss = trainer.train_step(example.input, example.target);
        assert(std::isfinite(step_loss));
    }
    const float trained_loss = trainer.evaluate(example.input, example.target);
    assert(trained_loss < initial_loss * 0.65F);
    assert(trainer.optimizer().step_count() == 80);

    auto parameters = model.parameters();
    const float norm = spiral::train::global_grad_norm(parameters);
    assert(std::isfinite(norm));

    const auto checkpoint_path = std::filesystem::temp_directory_path() / "spiral_l4_checkpoint.bin";
    spiral::train::save_checkpoint(model, checkpoint_path.string());
    const auto reference_logits = model.forward(example.input);

    Random other_rng(77);
    SpiralLanguageModel restored(config, other_rng);
    spiral::train::load_checkpoint(restored, checkpoint_path.string());
    const auto restored_logits = restored.forward(example.input);
    assert(reference_logits.shape() == restored_logits.shape());
    for (std::size_t i = 0; i < reference_logits.numel(); ++i) {
        assert(near(reference_logits.data()[i], restored_logits.data()[i]));
    }
    std::filesystem::remove(checkpoint_path);

    std::cout << "spiral_train_tests: PASS initial=" << initial_loss
              << " trained=" << trained_loss << '\n';
    return 0;
}
