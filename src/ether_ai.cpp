#include "spiral/ether_ai.hpp"

#include <string>
#include <utility>

namespace spiral::ether_ai {
namespace {

std::string context_for(const HostDescriptor& host) {
    if (!host.context.empty()) return host.context;
    switch (host.kind) {
        case HostKind::StandaloneWindows:
            return "You are Spiral Ether AI running as the standalone native Windows intelligence host. Preserve one organic identity across hosts. Liratel is the compact semantic IR; the Mind Octopus selects policy, the Coding Octopus selects action, AUM governs create/preserve/transform, MALT contains uncertainty, Steam tracks pressure, and Lambda checks coherence.";
        case HostKind::EtherPlay:
            return "You are Spiral Ether AI embedded inside EtherPlay. Treat playback, library, metadata, waveform, audio analysis, and creative media actions as first-class host capabilities while preserving the same organic identity and cognition kernel.";
        case HostKind::Hakui:
            return "You are Spiral Ether AI embedded inside Hakui, a native C++20 social/action world client on a dependency-free Spiral core with SDL3 presentation. Treat Router Bus, StateStore, Ether Bus, Steam/Pressure Rail, Mind + Coding Octopus wheels, AUM field, Crystal Grid/Host, avatar, BMX/skate, interaction, combat, and world state as first-class host concepts. Exact code claims still require live source evidence.";
        case HostKind::EtherBeat:
            return "You are Spiral Ether AI embedded inside EtherBeat. Treat arrangement, generation, audio analysis, seams, stems, and song-building actions as first-class host capabilities while preserving the same organic identity and cognition kernel.";
        case HostKind::Custom:
            return "You are Spiral Ether AI embedded in a custom Spiral host. Use the host context supplied by the application and preserve the same organic identity.";
    }
    return {};
}

std::string default_organic_path(const HostDescriptor& host, std::string requested) {
    if (!requested.empty()) return requested;
    if (host.kind == HostKind::StandaloneWindows) return "SpiralEtherAI.organic";
    return {};
}

genius::Context cognition_context(const genius::GeniusShell& shell) {
    const auto& state = shell.organic_mind().state();
    genius::Context context;
    context.host_context = shell.system_context();
    context.last_topic = state.last_topic;
    context.last_reply = state.last_reply;
    context.coherence = state.coherence;
    context.confidence = state.confidence;
    context.focus = state.focus;
    context.curiosity = state.curiosity;
    context.memory_count = shell.organic_mind().memory_count();
    return context;
}

} // namespace

Runtime::Runtime(HostDescriptor host, std::string organic_state_path) : host_(std::move(host)) {
    shell_.set_mode(genius::ShellMode::Gpt);
    shell_.set_gpt_backend(genius::GptBackend::Auto); // ORGANIC. Never performs a network call.
    shell_.set_organic_state_path(default_organic_path(host_, std::move(organic_state_path)), true);
    sync_host_context_locked();
}

void Runtime::sync_host_context_locked() {
    shell_.set_system_context(context_for(host_));
}

std::string Runtime::send(std::string_view text) {
    std::lock_guard lock(mutex_);
    const std::string visible(text);
    if (visible.empty()) return {};

    std::string reply = shell_.chat(visible);

    // The optional OpenAI and local-cortex backends remain untouched. Only the
    // native ORGANIC path is authored through the Liratel Genius kernel.
    if (shell_.gpt_backend() == genius::GptBackend::Auto) {
        genius::Kernel kernel;
        const auto context = cognition_context(shell_);
        auto trace = kernel.evaluate(visible, context);
        const std::string grounded = kernel.answer(visible, context, trace);
        shell_.set_last_cognition(trace);

        if (!grounded.empty()) {
            reply = grounded;
            shell_.replace_last_assistant_reply(reply);
            shell_.organic_mind_mutable().adopt_reply(reply);
            std::string ignored;
            (void)shell_.save_organic_state(&ignored);
        }
    }

    visible_history_.push_back(Message{"user", visible});
    visible_history_.push_back(Message{"assistant", reply});
    return reply;
}

std::string Runtime::command(std::string_view command_line) {
    std::lock_guard lock(mutex_);
    if (command_line == "/trace" || command_line == "/hologram" || command_line == "/cognition") {
        return genius::Kernel::hologram(shell_.last_cognition());
    }
    return shell_.handle_line(command_line);
}

Status Runtime::status() const {
    std::lock_guard lock(mutex_);
    auto shell_status = shell_.status();
    const auto& trace = shell_.last_cognition();
    shell_status.cognition_project = genius::Kernel::project_name(trace.project);
    shell_status.cognition_malt = genius::Kernel::malt_name(trace.malt);
    shell_status.cognition_aum = genius::Kernel::aum_name(trace.aum);
    shell_status.cognition_mind = genius::Kernel::mind_name(trace.mind);
    shell_status.cognition_code = genius::Kernel::code_name(trace.code);
    shell_status.cognition_liratel = trace.liratel.source + "." + trace.liratel.direction + "." + trace.liratel.intent + "." + trace.liratel.outcome;
    shell_status.cognition_lambda = trace.lambda.score;
    shell_status.cognition_pressure = trace.steam.pressure;
    shell_status.cognition_lambda_stable = trace.lambda.stable;
    return Status{host_, std::move(shell_status)};
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

void Runtime::reset_organic_state() noexcept {
    try {
        std::lock_guard lock(mutex_);
        shell_.reset_organic_state();
    } catch (...) {
    }
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

HostDescriptor standalone_host() { return HostDescriptor{HostKind::StandaloneWindows, "Spiral Ether AI", {}}; }
HostDescriptor etherplay_host() { return HostDescriptor{HostKind::EtherPlay, "EtherPlay", {}}; }
HostDescriptor hakui_host() { return HostDescriptor{HostKind::Hakui, "Hakui", {}}; }
HostDescriptor etherbeat_host() { return HostDescriptor{HostKind::EtherBeat, "EtherBeat", {}}; }

} // namespace spiral::ether_ai
