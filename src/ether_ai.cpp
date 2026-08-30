#include "spiral/ether_ai.hpp"

#include <utility>

namespace spiral::ether_ai {
namespace {

std::string context_for(const HostDescriptor& host) {
    if (!host.context.empty()) return host.context;
    switch (host.kind) {
        case HostKind::StandaloneWindows:
            return "You are Spiral Ether AI running as the standalone native Windows assistant. The same runtime can be embedded into EtherPlay, Hakui, EtherBeat, and other Spiral hosts.";
        case HostKind::EtherPlay:
            return "You are Spiral Ether AI embedded inside EtherPlay. Treat playback, library, metadata, waveform, audio analysis, and creative media actions as first-class host capabilities.";
        case HostKind::Hakui:
            return "You are Spiral Ether AI embedded inside Hakui. Treat the live world, avatar, scene, social runtime, media surfaces, and Spiral Units as first-class host capabilities.";
        case HostKind::EtherBeat:
            return "You are Spiral Ether AI embedded inside EtherBeat. Treat arrangement, generation, audio analysis, seams, stems, and song-building actions as first-class host capabilities.";
        case HostKind::Custom:
            return "You are Spiral Ether AI embedded in a custom Spiral host. Use the host context supplied by the application.";
    }
    return {};
}

} // namespace

Runtime::Runtime(HostDescriptor host) : host_(std::move(host)) {
    shell_.set_mode(genius::ShellMode::Gpt);
    shell_.set_gpt_backend(genius::GptBackend::Auto);
    sync_host_context_locked();
}

void Runtime::sync_host_context_locked() {
    shell_.set_system_context(context_for(host_));
}

std::string Runtime::internal_user_envelope(std::string_view visible_text) const {
    const std::string context = context_for(host_);
    std::string envelope;
    envelope.reserve(context.size() + visible_text.size() + 96);
    envelope += "[SPIRAL INTERNAL HOST CONTEXT]\n";
    envelope += context;
    envelope += "\n[END HOST CONTEXT]\n[VISIBLE USER MESSAGE]\n";
    envelope.append(visible_text.data(), visible_text.size());
    return envelope;
}

std::string Runtime::send(std::string_view text) {
    std::lock_guard lock(mutex_);
    const std::string visible(text);
    if (visible.empty()) return {};
    const std::string reply = shell_.chat(internal_user_envelope(text));
    visible_history_.push_back(Message{"user", visible});
    visible_history_.push_back(Message{"assistant", reply});
    return reply;
}

std::string Runtime::command(std::string_view command_line) {
    std::lock_guard lock(mutex_);
    return shell_.handle_line(command_line);
}

Status Runtime::status() const {
    std::lock_guard lock(mutex_);
    return Status{host_, shell_.status()};
}

std::vector<Message> Runtime::history() const {
    std::lock_guard lock(mutex_);
    return visible_history_;
}

void Runtime::set_host(HostDescriptor host) {
    std::lock_guard lock(mutex_);
    host_ = std::move(host);
    sync_host_context_locked();
}

HostDescriptor Runtime::host() const {
    std::lock_guard lock(mutex_);
    return host_;
}

void Runtime::set_backend(genius::GptBackend backend) {
    std::lock_guard lock(mutex_);
    shell_.set_gpt_backend(backend);
}

genius::GptBackend Runtime::backend() const {
    std::lock_guard lock(mutex_);
    return shell_.gpt_backend();
}

bool Runtime::load_local_model(const std::string& path, std::string* error) noexcept {
    try {
        std::lock_guard lock(mutex_);
        return shell_.load_model(path, error);
    } catch (const std::exception& exception) {
        if (error != nullptr) *error = exception.what();
        return false;
    } catch (...) {
        if (error != nullptr) *error = "unknown Ether AI model load failure";
        return false;
    }
}

void Runtime::unload_local_model() noexcept {
    try {
        std::lock_guard lock(mutex_);
        shell_.unload_model();
    } catch (...) {
    }
}

void Runtime::clear() {
    std::lock_guard lock(mutex_);
    shell_.clear_history();
    visible_history_.clear();
}

std::string host_kind_name(HostKind kind) {
    switch (kind) {
        case HostKind::StandaloneWindows: return "WINDOWS";
        case HostKind::EtherPlay: return "ETHERPLAY";
        case HostKind::Hakui: return "HAKUI";
        case HostKind::EtherBeat: return "ETHERBEAT";
        case HostKind::Custom: return "CUSTOM";
    }
    return "UNKNOWN";
}

HostDescriptor standalone_host() {
    return HostDescriptor{HostKind::StandaloneWindows, "Spiral Ether AI", {}};
}

HostDescriptor etherplay_host() {
    return HostDescriptor{HostKind::EtherPlay, "EtherPlay", {}};
}

HostDescriptor hakui_host() {
    return HostDescriptor{HostKind::Hakui, "Hakui", {}};
}

HostDescriptor etherbeat_host() {
    return HostDescriptor{HostKind::EtherBeat, "EtherBeat", {}};
}

} // namespace spiral::ether_ai
