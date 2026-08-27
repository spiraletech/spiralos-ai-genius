#include "spiral/generate.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace spiral::generate {
namespace {

struct Candidate {
    std::uint32_t token = 0;
    float logit = 0.0F;
    double probability = 0.0;
};

void validate_sampling_config(const SamplingConfig& config) {
    if (config.temperature < 0.0F || !std::isfinite(config.temperature)) {
        throw std::invalid_argument("sampling temperature must be finite and >= 0");
    }
    if (!(config.top_p > 0.0F && config.top_p <= 1.0F) || !std::isfinite(config.top_p)) {
        throw std::invalid_argument("sampling top_p must be finite in (0, 1]");
    }
    if (config.repetition_penalty <= 0.0F || !std::isfinite(config.repetition_penalty)) {
        throw std::invalid_argument("repetition penalty must be finite and > 0");
    }
}

std::vector<Candidate> prepare_candidates(
    const Tensor& logits,
    std::span<const std::uint32_t> history,
    const SamplingConfig& config) {
    if (logits.rank() != 1 || logits.numel() == 0) {
        throw std::invalid_argument("sample_token requires non-empty rank-1 logits");
    }

    std::vector<float> adjusted = logits.data();
    if (config.repetition_penalty != 1.0F) {
        std::vector<bool> seen(adjusted.size(), false);
        for (const auto token : history) {
            const auto index = static_cast<std::size_t>(token);
            if (index >= adjusted.size() || seen[index]) {
                continue;
            }
            seen[index] = true;
            float& value = adjusted[index];
            value = value >= 0.0F
                ? value / config.repetition_penalty
                : value * config.repetition_penalty;
        }
    }

    if (config.temperature > 0.0F && config.temperature != 1.0F) {
        for (auto& value : adjusted) {
            value /= config.temperature;
        }
    }

    std::vector<Candidate> candidates;
    candidates.reserve(adjusted.size());
    for (std::size_t i = 0; i < adjusted.size(); ++i) {
        candidates.push_back(Candidate{static_cast<std::uint32_t>(i), adjusted[i], 0.0});
    }

    std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
        if (a.logit == b.logit) return a.token < b.token;
        return a.logit > b.logit;
    });

    if (config.top_k > 0 && config.top_k < candidates.size()) {
        candidates.resize(config.top_k);
    }
    return candidates;
}

void normalize_and_apply_top_p(std::vector<Candidate>& candidates, float top_p) {
    const float max_logit = candidates.front().logit;
    double sum = 0.0;
    for (auto& candidate : candidates) {
        candidate.probability = std::exp(static_cast<double>(candidate.logit - max_logit));
        sum += candidate.probability;
    }
    if (!(sum > 0.0) || !std::isfinite(sum)) {
        throw std::runtime_error("sampling softmax normalization failed");
    }
    for (auto& candidate : candidates) {
        candidate.probability /= sum;
    }

    if (top_p < 1.0F && candidates.size() > 1) {
        double cumulative = 0.0;
        std::size_t keep = 0;
        for (; keep < candidates.size(); ++keep) {
            cumulative += candidates[keep].probability;
            if (cumulative >= static_cast<double>(top_p)) {
                ++keep;
                break;
            }
        }
        keep = std::max<std::size_t>(1, std::min(keep, candidates.size()));
        candidates.resize(keep);

        double retained = 0.0;
        for (const auto& candidate : candidates) retained += candidate.probability;
        for (auto& candidate : candidates) candidate.probability /= retained;
    }
}

} // namespace

std::uint32_t sample_token(
    const Tensor& logits,
    std::span<const std::uint32_t> history,
    const SamplingConfig& config,
    Random& rng) {
    validate_sampling_config(config);
    auto candidates = prepare_candidates(logits, history, config);

    if (config.temperature == 0.0F || candidates.size() == 1) {
        return candidates.front().token;
    }

    normalize_and_apply_top_p(candidates, config.top_p);
    const double draw = static_cast<double>(rng.uniform());
    double cumulative = 0.0;
    for (const auto& candidate : candidates) {
        cumulative += candidate.probability;
        if (draw < cumulative) {
            return candidate.token;
        }
    }
    return candidates.back().token;
}

std::vector<std::uint32_t> build_context(
    std::span<const std::uint32_t> history,
    std::size_t max_context_tokens,
    bool preserve_bos,
    std::uint32_t bos_token) {
    if (max_context_tokens == 0) {
        throw std::invalid_argument("max_context_tokens must be > 0");
    }
    if (history.empty()) {
        return {};
    }
    if (history.size() <= max_context_tokens) {
        return {history.begin(), history.end()};
    }

    if (preserve_bos && max_context_tokens >= 2 && history.front() == bos_token) {
        std::vector<std::uint32_t> out;
        out.reserve(max_context_tokens);
        out.push_back(bos_token);
        const std::size_t tail_count = max_context_tokens - 1;
        out.insert(out.end(), history.end() - static_cast<std::ptrdiff_t>(tail_count), history.end());
        return out;
    }

    return {history.end() - static_cast<std::ptrdiff_t>(max_context_tokens), history.end()};
}

bool is_stop_token(std::uint32_t token, std::span<const std::uint32_t> stop_tokens) {
    return std::find(stop_tokens.begin(), stop_tokens.end(), token) != stop_tokens.end();
}

GenerationResult TokenGenerator::generate(
    std::span<const std::uint32_t> prompt_tokens,
    const GenerationConfig& config,
    TokenCallback callback) const {
    if (prompt_tokens.empty()) {
        throw std::invalid_argument("TokenGenerator requires at least one prompt token");
    }
    if (config.max_context_tokens == 0) {
        throw std::invalid_argument("TokenGenerator max_context_tokens must be > 0");
    }
    validate_sampling_config(config.sampling);

    for (const auto token : prompt_tokens) {
        if (static_cast<std::size_t>(token) >= model_.config().vocabulary_size) {
            throw std::out_of_range("prompt token exceeds model vocabulary");
        }
    }
    for (const auto token : config.stop_tokens) {
        if (static_cast<std::size_t>(token) >= model_.config().vocabulary_size) {
            throw std::out_of_range("stop token exceeds model vocabulary");
        }
    }

    GenerationResult result;
    result.prompt_tokens.assign(prompt_tokens.begin(), prompt_tokens.end());
    std::vector<std::uint32_t> history = result.prompt_tokens;
    Random rng(config.sampling.seed);

    for (std::size_t index = 0; index < config.max_new_tokens; ++index) {
        const auto context = build_context(
            std::span<const std::uint32_t>(history.data(), history.size()),
            config.max_context_tokens,
            config.preserve_bos);
        const Tensor logits = model_.last_token_logits(
            std::span<const std::uint32_t>(context.data(), context.size()));
        const std::uint32_t token = sample_token(
            logits,
            std::span<const std::uint32_t>(history.data(), history.size()),
            config.sampling,
            rng);

        history.push_back(token);
        result.generated_tokens.push_back(token);

        if (is_stop_token(token, config.stop_tokens)) {
            result.stopped_by_token = true;
            break;
        }
        if (callback && !callback(token, index)) {
            result.cancelled = true;
            break;
        }
    }
    return result;
}

ByteTextGenerator::ByteTextGenerator(const nn::SpiralLanguageModel& model)
    : model_(model) {
    if (model_.config().vocabulary_size != ByteTokenizer::vocabulary_size) {
        throw std::invalid_argument("ByteTextGenerator requires the Spiral byte-token vocabulary");
    }
}

GenerationResult ByteTextGenerator::generate(
    std::string_view prompt,
    const GenerationConfig& config,
    TextCallback callback) const {
    const auto prompt_tokens = tokenizer_.encode(prompt, true, false);
    TokenGenerator generator(model_);

    auto result = generator.generate(
        std::span<const std::uint32_t>(prompt_tokens.data(), prompt_tokens.size()),
        config,
        [&](std::uint32_t token, std::size_t index) {
            if (!callback) return true;
            const std::vector<std::uint32_t> single{token};
            const std::string fragment = tokenizer_.decode(single);
            return callback(token, fragment, index);
        });
    result.text = tokenizer_.decode(result.generated_tokens);
    return result;
}

} // namespace spiral::generate
