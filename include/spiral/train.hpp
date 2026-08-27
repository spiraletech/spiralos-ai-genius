#pragma once

#include "spiral/model.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace spiral::train {

struct TrainingExample {
    std::vector<std::uint32_t> input;
    std::vector<std::uint32_t> target;
};

class TokenDataset {
public:
    TokenDataset(std::vector<std::uint32_t> tokens, std::size_t sequence_length);

    [[nodiscard]] std::size_t size() const noexcept { return examples_.size(); }
    [[nodiscard]] const TrainingExample& at(std::size_t index) const;
    [[nodiscard]] const std::vector<TrainingExample>& examples() const noexcept { return examples_; }

private:
    std::vector<TrainingExample> examples_;
};

struct AdamWConfig {
    float learning_rate = 3.0e-4F;
    float beta1 = 0.9F;
    float beta2 = 0.999F;
    float epsilon = 1.0e-8F;
    float weight_decay = 0.01F;
};

class AdamW {
public:
    explicit AdamW(std::vector<nn::Parameter*> parameters, AdamWConfig config = {});

    void step();
    void zero_grad();
    [[nodiscard]] std::uint64_t step_count() const noexcept { return step_count_; }

private:
    struct State {
        nn::Parameter* parameter = nullptr;
        Tensor first_moment;
        Tensor second_moment;
    };

    AdamWConfig config_;
    std::vector<State> states_;
    std::uint64_t step_count_ = 0;
};

[[nodiscard]] float cross_entropy_loss(
    const Tensor& logits,
    std::span<const std::uint32_t> targets);

[[nodiscard]] float global_grad_norm(std::span<nn::Parameter* const> parameters);
void clip_grad_norm(std::span<nn::Parameter* const> parameters, float max_norm);

struct TrainerConfig {
    AdamWConfig optimizer;
    float max_grad_norm = 1.0F;
};

class LanguageModelTrainer {
public:
    explicit LanguageModelTrainer(nn::SpiralLanguageModel& model, TrainerConfig config = {});

    [[nodiscard]] float evaluate(
        std::span<const std::uint32_t> input,
        std::span<const std::uint32_t> target) const;

    float train_step(
        std::span<const std::uint32_t> input,
        std::span<const std::uint32_t> target);

    float train_epoch(const TokenDataset& dataset);

    [[nodiscard]] AdamW& optimizer() noexcept { return optimizer_; }
    [[nodiscard]] const AdamW& optimizer() const noexcept { return optimizer_; }

private:
    float loss_and_backward(
        std::span<const std::uint32_t> input,
        std::span<const std::uint32_t> target);

    nn::SpiralLanguageModel& model_;
    TrainerConfig config_;
    AdamW optimizer_;
};

void save_checkpoint(const nn::SpiralLanguageModel& model, const std::string& path);
void load_checkpoint(nn::SpiralLanguageModel& model, const std::string& path);

} // namespace spiral::train
