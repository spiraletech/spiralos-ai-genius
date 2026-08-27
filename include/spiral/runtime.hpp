#pragma once

#include "spiral/model.hpp"
#include "spiral/tensor.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace spiral::runtime {

struct LayerKVCache {
    std::size_t num_heads = 0;
    std::size_t head_dim = 0;
    std::vector<float> keys;
    std::vector<float> values;

    [[nodiscard]] std::size_t token_count() const noexcept;
    void clear() noexcept;
};

class InferenceSession {
public:
    explicit InferenceSession(const nn::SpiralLanguageModel& model);

    void reset();
    [[nodiscard]] Tensor prefill(std::span<const std::uint32_t> tokens);
    [[nodiscard]] Tensor append(std::uint32_t token);

    [[nodiscard]] const std::vector<std::uint32_t>& tokens() const noexcept { return tokens_; }
    [[nodiscard]] std::size_t token_count() const noexcept { return tokens_.size(); }
    [[nodiscard]] const std::vector<LayerKVCache>& layer_caches() const noexcept { return caches_; }

private:
    [[nodiscard]] Tensor forward_token(std::uint32_t token, std::size_t position);

    const nn::SpiralLanguageModel& model_;
    std::vector<LayerKVCache> caches_;
    std::vector<std::uint32_t> tokens_;
};

struct LoadedModelBundle {
    nn::ModelConfig config;
    std::unique_ptr<nn::SpiralLanguageModel> model;
};

void save_model_bundle(const nn::SpiralLanguageModel& model, const std::string& path);
[[nodiscard]] LoadedModelBundle load_model_bundle(
    const std::string& path,
    std::uint64_t initialization_seed = 0x53504952414CULL);

} // namespace spiral::runtime
