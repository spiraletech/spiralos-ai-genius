#pragma once

#include "spiral/genius_shell.hpp"
#include "spiral/xenon_os.hpp"

#include <cstddef>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

namespace spiral::ether_ai {

enum class HostKind {
    StandaloneWindows,
    XenonOS,
    EtherPlay,
    Hakui,
    EtherBeat,
    Custom,
};

struct HostDescriptor {
    HostKind kind = HostKind::StandaloneWindows;
    std::string name = "Spiral Ether AI";
    std::string context;
};

struct Message {
    std::string role;
    std::string content;
};

struct Status {
    HostDescriptor host;
    genius::ShellStatus shell;
    bool xenon_online = true;
    bool xenon_local_cortex_loaded = false;
    std::string xenon_model_path;
    std::size_t xenon_tool_count = 0;
};

class Runtime final {
public:
    explicit Runtime(HostDescriptor host = {}, std::string organic_state_path = {});

    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;

    [[nodiscard]] std::string send(std::string_view text);
    [[nodiscard]] std::string command(std::string_view command_line);
    [[nodiscard]] Status status() const;
    [[nodiscard]] std::vector<Message> history() const;

    void set_host(HostDescriptor host);
    [[nodiscard]] HostDescriptor host() const;

    void set_backend(genius::GptBackend backend);
    [[nodiscard]] genius::GptBackend backend() const;

    void configure_local_generation(std::size_t max_new_tokens, float temperature) noexcept;
    [[nodiscard]] std::size_t local_max_new_tokens() const noexcept;
    [[nodiscard]] float local_temperature() const noexcept;

    [[nodiscard]] bool load_local_model(const std::string& path, std::string* error = nullptr) noexcept;
    void unload_local_model() noexcept;

    [[nodiscard]] xenon::ToolResult dispatch_tool(const xenon::ToolIntent& intent, bool allow_mutation = false);
    [[nodiscard]] std::vector<xenon::ToolDefinition> tool_capabilities() const;

    void clear();
    void reset_organic_state() noexcept;

private:
    void sync_host_context_locked();
    [[nodiscard]] xenon::SpiralContext xenon_context_locked(std::string_view pending_user_text = {}) const;

    mutable std::mutex mutex_;
    HostDescriptor host_;
    genius::GeniusShell shell_;
    xenon::LocalCortex local_cortex_;
    xenon::ToolBus tool_bus_;
    std::vector<xenon::ToolResult> recent_tool_results_;
    std::vector<Message> visible_history_;
    std::size_t local_max_new_tokens_ = 384;
    float local_temperature_ = 0.62F;
};

[[nodiscard]] std::string host_kind_name(HostKind kind);
[[nodiscard]] HostDescriptor standalone_host();
[[nodiscard]] HostDescriptor xenon_host();
[[nodiscard]] HostDescriptor etherplay_host();
[[nodiscard]] HostDescriptor hakui_host();
[[nodiscard]] HostDescriptor etherbeat_host();

} // namespace spiral::ether_ai
