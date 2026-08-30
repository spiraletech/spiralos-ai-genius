#pragma once

#include "spiral/generate.hpp"
#include "spiral/gpu.hpp"
#include "spiral/gpu_compute.hpp"
#include "spiral/openai_backend.hpp"
#include "spiral/organic_ai.hpp"
#include "spiral/runtime.hpp"

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace spiral::genius {

enum class ShellMode {
    Native,
    Gpt,
};

enum class GptBackend {
    Auto,        // Native organic mind. Never calls a network API.
    OpenAI,      // Explicit optional bridge.
    SpiralLocal, // Explicit trained local model bundle.
};

struct ChatTurn {
    std::string role;
    std::string content;
};

struct ShellStatus {
    ShellMode mode = ShellMode::Gpt;
    GptBackend gpt_backend = GptBackend::Auto;
    bool gpu_platform_supported = false;
    bool gpu_available = false;
    bool gpu_hardware_accelerated = false;
    std::string gpu_adapter;
    std::string gpu_feature_level;
    bool openai_platform_supported = false;
    bool openai_key_present = false;
    std::string openai_model;
    bool model_loaded = false;
    std::string model_path;
    std::size_t conversation_turns = 0;
    std::size_t max_new_tokens = 128;
    float temperature = 0.8F;

    std::uint64_t organic_revision = 0;
    std::uint64_t organic_turns = 0;
    std::size_t organic_memories = 0;
    float organic_energy = 0.0F;
    float organic_focus = 0.0F;
    float organic_curiosity = 0.0F;
    float organic_confidence = 0.0F;
    float organic_warmth = 0.0F;
    float organic_novelty = 0.0F;
    float organic_coherence = 0.0F;
    std::string organic_topic;
};

class GeniusShell final {
public:
    GeniusShell();

    [[nodiscard]] ShellStatus status() const;
    [[nodiscard]] ShellMode mode() const noexcept { return mode_; }
    [[nodiscard]] GptBackend gpt_backend() const noexcept { return gpt_backend_; }
    [[nodiscard]] const std::vector<ChatTurn>& history() const noexcept { return history_; }
    [[nodiscard]] const std::string& system_context() const noexcept { return system_context_; }
    [[nodiscard]] const organic::OrganicMind& organic_mind() const noexcept { return organic_mind_; }

    void set_mode(ShellMode mode) noexcept { mode_ = mode; }
    void set_gpt_backend(GptBackend backend) noexcept { gpt_backend_ = backend; }
    void set_system_context(std::string context) { system_context_ = std::move(context); }
    void clear_history() noexcept { history_.clear(); }

    void set_organic_state_path(std::string path, bool load_existing = true) noexcept;
    [[nodiscard]] const std::string& organic_state_path() const noexcept { return organic_state_path_; }
    [[nodiscard]] bool save_organic_state(std::string* error = nullptr) const noexcept;
    [[nodiscard]] bool load_organic_state(std::string* error = nullptr) noexcept;
    void reset_organic_state() noexcept;

    [[nodiscard]] bool load_model(const std::string& path, std::string* error = nullptr) noexcept;
    void unload_model() noexcept;

    [[nodiscard]] std::string handle_line(std::string_view line);
    [[nodiscard]] std::string chat(std::string_view user_message);
    [[nodiscard]] std::string help_text() const;
    [[nodiscard]] std::string status_text() const;
    [[nodiscard]] std::string banner_text() const;

    [[nodiscard]] bool should_exit() const noexcept { return should_exit_; }

private:
    [[nodiscard]] std::string run_command(std::string_view line);
    [[nodiscard]] std::string native_reply(std::string_view user_message);
    [[nodiscard]] std::string gpt_reply();
    [[nodiscard]] std::string organic_reply();
    [[nodiscard]] std::string local_spiral_reply();
    [[nodiscard]] std::string openai_gpt_reply();
    [[nodiscard]] std::string build_chat_prompt() const;
    [[nodiscard]] std::string build_openai_input() const;
    [[nodiscard]] std::string openai_instructions() const;
    [[nodiscard]] static std::string trim(std::string_view text);
    [[nodiscard]] static std::optional<float> parse_float(std::string_view text);
    [[nodiscard]] static std::optional<std::size_t> parse_size(std::string_view text);

    ShellMode mode_ = ShellMode::Gpt;
    GptBackend gpt_backend_ = GptBackend::Auto;
    bool should_exit_ = false;
    std::vector<ChatTurn> history_;
    std::string system_context_;

    organic::OrganicMind organic_mind_;
    std::string organic_state_path_;

    std::unique_ptr<gpu::D3D11GpuDevice> gpu_device_;
    std::unique_ptr<gpu::D3D11ComputeEngine> gpu_compute_;
    std::string gpu_error_;

    openai::ResponsesBackend openai_backend_;
    std::string openai_model_;

    std::unique_ptr<runtime::LoadedModelBundle> model_bundle_;
    std::string model_path_;
    generate::GenerationConfig generation_;
};

[[nodiscard]] std::string shell_mode_name(ShellMode mode);
[[nodiscard]] std::string gpt_backend_name(GptBackend backend);

} // namespace spiral::genius
