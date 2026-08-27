#include "spiral/train.hpp"
#include "train_detail.hpp"

#include <cmath>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <utility>

namespace spiral::train {
namespace {

constexpr std::uint32_t checkpoint_magic = 0x5350524CU;
constexpr std::uint32_t checkpoint_version = 1U;

void write_u32(std::ofstream& stream, std::uint32_t value) {
    stream.write(reinterpret_cast<const char*>(&value), sizeof(value));
}
void write_u64(std::ofstream& stream, std::uint64_t value) {
    stream.write(reinterpret_cast<const char*>(&value), sizeof(value));
}
std::uint32_t read_u32(std::ifstream& stream) {
    std::uint32_t value{};
    stream.read(reinterpret_cast<char*>(&value), sizeof(value));
    return value;
}
std::uint64_t read_u64(std::ifstream& stream) {
    std::uint64_t value{};
    stream.read(reinterpret_cast<char*>(&value), sizeof(value));
    return value;
}

} // namespace

TokenDataset::TokenDataset(std::vector<std::uint32_t> tokens, std::size_t sequence_length) {
    if (sequence_length == 0) throw std::invalid_argument("TokenDataset sequence length must be non-zero");
    if (tokens.size() <= sequence_length) return;
    for (std::size_t start = 0; start + sequence_length < tokens.size(); ++start) {
        TrainingExample example;
        example.input.assign(tokens.begin() + static_cast<std::ptrdiff_t>(start),
                             tokens.begin() + static_cast<std::ptrdiff_t>(start + sequence_length));
        example.target.assign(tokens.begin() + static_cast<std::ptrdiff_t>(start + 1),
                              tokens.begin() + static_cast<std::ptrdiff_t>(start + sequence_length + 1));
        examples_.push_back(std::move(example));
    }
}

const TrainingExample& TokenDataset::at(std::size_t index) const {
    return examples_.at(index);
}

AdamW::AdamW(std::vector<nn::Parameter*> parameters, AdamWConfig config)
    : config_(config) {
    if (config_.learning_rate <= 0.0F || config_.beta1 < 0.0F || config_.beta1 >= 1.0F
        || config_.beta2 < 0.0F || config_.beta2 >= 1.0F || config_.epsilon <= 0.0F
        || config_.weight_decay < 0.0F) {
        throw std::invalid_argument("AdamW configuration is invalid");
    }
    states_.reserve(parameters.size());
    for (auto* parameter : parameters) {
        if (parameter == nullptr || !parameter->trainable) continue;
        parameter->ensure_grad();
        states_.push_back(State{
            parameter,
            Tensor::zeros(parameter->value.shape()),
            Tensor::zeros(parameter->value.shape())});
    }
}

void AdamW::zero_grad() {
    for (auto& state : states_) state.parameter->zero_grad();
}

void AdamW::step() {
    ++step_count_;
    const float beta1_correction = 1.0F - std::pow(config_.beta1, static_cast<float>(step_count_));
    const float beta2_correction = 1.0F - std::pow(config_.beta2, static_cast<float>(step_count_));

    for (auto& state : states_) {
        auto& parameter = *state.parameter;
        parameter.ensure_grad();
        for (std::size_t i = 0; i < parameter.value.numel(); ++i) {
            const float gradient = parameter.grad.data()[i];
            state.first_moment.data()[i] = config_.beta1 * state.first_moment.data()[i]
                + (1.0F - config_.beta1) * gradient;
            state.second_moment.data()[i] = config_.beta2 * state.second_moment.data()[i]
                + (1.0F - config_.beta2) * gradient * gradient;
            const float m_hat = state.first_moment.data()[i] / beta1_correction;
            const float v_hat = state.second_moment.data()[i] / beta2_correction;
            const float update = m_hat / (std::sqrt(v_hat) + config_.epsilon)
                + config_.weight_decay * parameter.value.data()[i];
            parameter.value.data()[i] -= config_.learning_rate * update;
        }
    }
}

float cross_entropy_loss(const Tensor& logits, std::span<const std::uint32_t> targets) {
    float loss = 0.0F;
    (void)detail::logits_cross_entropy_gradient(logits, targets, loss);
    return loss;
}

float global_grad_norm(std::span<nn::Parameter* const> parameters) {
    double sum = 0.0;
    for (auto* parameter : parameters) {
        if (parameter == nullptr || !parameter->trainable) continue;
        parameter->ensure_grad();
        for (const float value : parameter->grad.data()) {
            sum += static_cast<double>(value) * static_cast<double>(value);
        }
    }
    return static_cast<float>(std::sqrt(sum));
}

void clip_grad_norm(std::span<nn::Parameter* const> parameters, float max_norm) {
    if (max_norm <= 0.0F) throw std::invalid_argument("max gradient norm must be positive");
    const float norm = global_grad_norm(parameters);
    if (norm <= max_norm || norm == 0.0F) return;
    const float scale = max_norm / (norm + 1.0e-6F);
    for (auto* parameter : parameters) {
        if (parameter == nullptr || !parameter->trainable) continue;
        parameter->ensure_grad();
        for (auto& value : parameter->grad.data()) value *= scale;
    }
}

LanguageModelTrainer::LanguageModelTrainer(nn::SpiralLanguageModel& model, TrainerConfig config)
    : model_(model), config_(config), optimizer_(model.parameters(), config.optimizer) {
    if (config_.max_grad_norm <= 0.0F) {
        throw std::invalid_argument("trainer max_grad_norm must be positive");
    }
}

float LanguageModelTrainer::evaluate(
    std::span<const std::uint32_t> input,
    std::span<const std::uint32_t> target) const {
    if (input.size() != target.size()) throw std::invalid_argument("evaluate input/target length mismatch");
    return cross_entropy_loss(model_.forward(input), target);
}

float LanguageModelTrainer::loss_and_backward(
    std::span<const std::uint32_t> input,
    std::span<const std::uint32_t> target) {
    if (input.empty() || input.size() != target.size()) {
        throw std::invalid_argument("training input and target must be non-empty and equal length");
    }

    Tensor hidden = model_.token_embedding().forward(input);
    std::vector<detail::BlockCache> block_caches(model_.blocks().size());
    for (std::size_t layer = 0; layer < model_.blocks().size(); ++layer) {
        hidden = detail::block_forward(*model_.blocks()[layer], hidden, block_caches[layer]);
    }

    const Tensor final_norm_input = hidden;
    const Tensor final_norm_output = model_.final_norm().forward(hidden);
    const Tensor logits = model_.lm_head().forward(final_norm_output);

    float loss = 0.0F;
    const Tensor grad_logits = detail::logits_cross_entropy_gradient(logits, target, loss);
    Tensor grad_hidden = detail::linear_backward(model_.lm_head(), final_norm_output, grad_logits);
    grad_hidden = detail::rmsnorm_backward(model_.final_norm(), final_norm_input, grad_hidden);

    for (std::size_t layer = model_.blocks().size(); layer-- > 0;) {
        grad_hidden = detail::block_backward(*model_.blocks()[layer], block_caches[layer], grad_hidden);
    }

    auto& embedding = model_.token_embedding().table();
    embedding.ensure_grad();
    const std::size_t dim = model_.config().model_dim;
    for (std::size_t row = 0; row < input.size(); ++row) {
        const std::size_t token = static_cast<std::size_t>(input[row]);
        if (token >= model_.config().vocabulary_size) throw std::out_of_range("training token exceeds vocabulary");
        for (std::size_t col = 0; col < dim; ++col) {
            embedding.grad.data()[token * dim + col] += grad_hidden.data()[row * dim + col];
        }
    }
    return loss;
}

float LanguageModelTrainer::train_step(
    std::span<const std::uint32_t> input,
    std::span<const std::uint32_t> target) {
    optimizer_.zero_grad();
    const float loss = loss_and_backward(input, target);
    auto parameters = model_.parameters();
    clip_grad_norm(parameters, config_.max_grad_norm);
    optimizer_.step();
    return loss;
}

float LanguageModelTrainer::train_epoch(const TokenDataset& dataset) {
    if (dataset.size() == 0) throw std::invalid_argument("cannot train on empty dataset");
    double total = 0.0;
    for (const auto& example : dataset.examples()) total += train_step(example.input, example.target);
    return static_cast<float>(total / static_cast<double>(dataset.size()));
}

void save_checkpoint(const nn::SpiralLanguageModel& model, const std::string& path) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) throw std::runtime_error("failed to open checkpoint for writing");
    write_u32(stream, checkpoint_magic);
    write_u32(stream, checkpoint_version);
    const auto parameters = model.parameters();
    write_u64(stream, static_cast<std::uint64_t>(parameters.size()));
    for (const auto* parameter : parameters) {
        write_u64(stream, static_cast<std::uint64_t>(parameter->value.rank()));
        for (const auto dim : parameter->value.shape()) write_u64(stream, static_cast<std::uint64_t>(dim));
        write_u64(stream, static_cast<std::uint64_t>(parameter->value.numel()));
        stream.write(
            reinterpret_cast<const char*>(parameter->value.data().data()),
            static_cast<std::streamsize>(parameter->value.numel() * sizeof(float)));
    }
    if (!stream) throw std::runtime_error("failed while writing checkpoint");
}

void load_checkpoint(nn::SpiralLanguageModel& model, const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("failed to open checkpoint for reading");
    if (read_u32(stream) != checkpoint_magic || read_u32(stream) != checkpoint_version) {
        throw std::runtime_error("unsupported Spiral checkpoint format");
    }
    auto parameters = model.parameters();
    const auto count = read_u64(stream);
    if (count != parameters.size()) throw std::runtime_error("checkpoint parameter count mismatch");
    for (auto* parameter : parameters) {
        const auto rank = read_u64(stream);
        if (rank != parameter->value.rank()) throw std::runtime_error("checkpoint parameter rank mismatch");
        for (std::size_t axis = 0; axis < parameter->value.rank(); ++axis) {
            if (read_u64(stream) != parameter->value.shape()[axis]) {
                throw std::runtime_error("checkpoint parameter shape mismatch");
            }
        }
        const auto numel = read_u64(stream);
        if (numel != parameter->value.numel()) throw std::runtime_error("checkpoint parameter size mismatch");
        stream.read(
            reinterpret_cast<char*>(parameter->value.data().data()),
            static_cast<std::streamsize>(parameter->value.numel() * sizeof(float)));
        if (!stream) throw std::runtime_error("checkpoint ended unexpectedly");
    }
}

} // namespace spiral::train
