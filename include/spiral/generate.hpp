#pragma once

#include "spiral/model.hpp"
#include "spiral/random.hpp"
#include "spiral/tokenizer.hpp"
#include "spiral/tensor.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace spiral::generate {

struct SamplingConfig {
    float temperature = 0.8F;
    std::size_t top_k = 40;
    float top_p = 0.95F;
    float repetition_penalty = 1.0F;
    std::uint64_t seed = 0x53504952414CULL;
};

struct GenerationConfig {
    SamplingConfig sampling;
    std::size_t max_new_tokens = 128;
    std::size_t max_context_tokens = 512;
    bool preserve_bos = true;
    std::vector<std::uint32_t> stop_tokens = {ByteTokenizer::eos_token};
};

struct GenerationResult {
    std::vector<std::uint32_t> prompt_tokens;
    std::vector<std::uint32_t> generated_tokens;
    std::string text;
    bool stopped_by_token = false;
    bool cancelled = false;
};

using TokenCallback = std::function<bool(std::uint32_t token, std::size_t generated_index)>;
using TextCallback = std::function<bool(std::uint32_t token, std::string_view fragment, std::size_t generated_index)>;

[[nodiscard]] std::uint32_t sample_token(
    const Tensor& logits,
    std::span<const std::uint32_t> history,
    const SamplingConfig& config,
    Random& rng);

[[nodiscard]] std::vector<std::uint32_t> build_context(
    std::span<const std::uint32_t> history,
    std::size_t max_context_tokens,
    bool preserve_bos = true,
    std::uint32_t bos_token = ByteTokenizer::bos_token);

[[nodiscard]] bool is_stop_token(std::uint32_t token, std::span<const std::uint32_t> stop_tokens);

class TokenGenerator {
public:
    explicit TokenGenerator(const nn::SpiralLanguageModel& model) : model_(model) {}

    [[nodiscard]] GenerationResult generate(
        std::span<const std::uint32_t> prompt_tokens,
        const GenerationConfig& config = {},
        TokenCallback callback = {}) const;

private:
    const nn::SpiralLanguageModel& model_;
};

class ByteTextGenerator {
public:
    explicit ByteTextGenerator(const nn::SpiralLanguageModel& model);

    [[nodiscard]] GenerationResult generate(
        std::string_view prompt,
        const GenerationConfig& config = {},
        TextCallback callback = {}) const;

private:
    const nn::SpiralLanguageModel& model_;
    ByteTokenizer tokenizer_;
};

} // namespace spiral::generate
