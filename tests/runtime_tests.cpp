#include "spiral/memory.hpp"
#include "spiral/model.hpp"
#include "spiral/random.hpp"
#include "spiral/runtime.hpp"
#include "spiral/tools.hpp"

#include <cassert>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace {
bool near(float a, float b, float epsilon = 2.0e-4F) { return std::fabs(a - b) <= epsilon; }
}

int main() {
    using spiral::Random;
    using spiral::nn::ModelConfig;
    using spiral::nn::SpiralLanguageModel;

    ModelConfig config{16, 8, 2, 2, 16, 1.0e-5F};
    Random rng(6060);
    SpiralLanguageModel model(config, rng);
    const std::vector<std::uint32_t> prefix{1, 2, 3, 4};

    spiral::runtime::InferenceSession session(model);
    const auto cached_logits = session.prefill(prefix);
    const auto full_logits = model.last_token_logits(prefix);
    assert(cached_logits.shape() == full_logits.shape());
    for (std::size_t i = 0; i < full_logits.numel(); ++i) assert(near(cached_logits.data()[i], full_logits.data()[i]));
    assert(session.token_count() == prefix.size());
    assert(session.layer_caches().size() == config.num_layers);
    for (const auto& cache : session.layer_caches()) assert(cache.token_count() == prefix.size());

    const auto appended = session.append(5);
    const std::vector<std::uint32_t> longer{1, 2, 3, 4, 5};
    const auto full_longer = model.last_token_logits(longer);
    for (std::size_t i = 0; i < appended.numel(); ++i) assert(near(appended.data()[i], full_longer.data()[i]));

    const auto temp = std::filesystem::temp_directory_path();
    const auto bundle_path = (temp / "spiral_l6_bundle.saig").string();
    spiral::runtime::save_model_bundle(model, bundle_path);
    auto loaded = spiral::runtime::load_model_bundle(bundle_path, 1);
    assert(loaded.config.vocabulary_size == config.vocabulary_size);
    const auto loaded_logits = loaded.model->last_token_logits(prefix);
    for (std::size_t i = 0; i < full_logits.numel(); ++i) assert(near(loaded_logits.data()[i], full_logits.data()[i], 1.0e-6F));
    std::filesystem::remove(bundle_path);

    spiral::memory::MemoryStore memory;
    const auto music_id = memory.remember("EtherPlay uses a native audio engine and waveform UI", {"etherplay", "audio"}, 100);
    memory.remember("Hakui uses third person movement and social world simulation", {"hakui", "game"}, 200);
    const auto hits = memory.search("EtherPlay audio waveform", 2);
    assert(!hits.empty());
    assert(hits.front().record.id == music_id);

    const auto memory_path = (temp / "spiral_l6_memory.smem").string();
    memory.save(memory_path);
    spiral::memory::MemoryStore restored;
    restored.load(memory_path);
    assert(restored.records().size() == 2);
    assert(restored.search("third person Hakui", 1).front().record.text.find("Hakui") != std::string::npos);
    std::filesystem::remove(memory_path);

    spiral::tools::ToolRegistry tools;
    tools.register_tool({"echo", "Return the input"}, [](std::string_view input) {
        return spiral::tools::ToolResult::success(std::string(input));
    });
    tools.register_tool({"fail", "Return a controlled failure"}, [](std::string_view) {
        return spiral::tools::ToolResult::failure("expected");
    });
    assert(tools.contains("echo"));
    assert(tools.list().size() == 2);
    const auto echoed = tools.invoke("echo", "spiral");
    assert(echoed.ok && echoed.output == "spiral");
    const auto missing = tools.invoke("missing");
    assert(!missing.ok && !missing.error.empty());

    std::cout << "spiral_runtime_tests: PASS cache_tokens=" << session.token_count()
              << " memories=" << restored.records().size() << '\n';
    return 0;
}
