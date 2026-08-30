#pragma once

#include "spiral/genius_shell.hpp"

#include <cstddef>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

namespace spiral::ether_ai {

enum class HostKind {
    StandaloneWindows,
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
};

class Runtime final {
public:
    explicit Runtime(HostDescriptor host = {});

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

    [[nodiscard]] bool load_local_model(const std::string& path, std::string* error = nullptr) noexcept;
    void unload_local_model() noexcept;
    void clear();

private:
    void sync_host_context_locked();

    mutable std::mutex mutex_;
    HostDescriptor host_;
    genius::GeniusShell shell_;
};

[[nodiscard]] std::string host_kind_name(HostKind kind);
[[nodiscard]] HostDescriptor standalone_host();
[[nodiscard]] HostDescriptor etherplay_host();
[[nodiscard]] HostDescriptor hakui_host();
[[nodiscard]] HostDescriptor etherbeat_host();

} // namespace spiral::ether_ai
