#include "spiral/ether_ai.hpp"

#include <algorithm>
#include <filesystem>
#include <sstream>
#include <string>
#include <utility>

namespace spiral::ether_ai {
namespace {

std::string context_for(const HostDescriptor& host) {
    if (!host.context.empty()) return host.context;
    switch (host.kind) {
        case HostKind::StandaloneWindows:
            return "You are Spiral Ether AI running as the standalone native Windows intelligence host. XENON OS is your host-neutral local cortex/tool runtime. Preserve one ORGANIC identity across every host.";
        case HostKind::XenonOS:
            return "You are Spiral Ether AI operating directly through XENON OS: the host-neutral intelligence bus joining the local language cortex, ORGANIC memory, native tools, and external EtherTech engine adapters.";
        case HostKind::EtherPlay:
            return "You are Spiral Ether AI embedded inside EtherPlay through XENON OS. Treat playback, library, metadata, waveform, audio analysis, spectral/rhythm/pitch/transient listening, and creative media actions as first-class host capabilities while preserving the same ORGANIC identity.";
        case HostKind::Hakui:
            return "You are Spiral Ether AI embedded inside Hakui through XENON OS. Treat world, avatar, BMX/skate, interaction, inventory, physics, events, combat, Router Bus, StateStore, Ether Bus, Steam/Pressure Rail, Mind + Coding Octopus wheels, AUM field, and Crystal Grid/Host as native host state. Exact code claims require live source/tool evidence.";
        case HostKind::EtherBeat:
            return "You are Spiral Ether AI embedded inside EtherBeat through XENON OS. Act as a producer/director over arrangement, MIDI, drums, harmony, audio analysis, seams, stems, generation and song-building tools. Listen through analysis results, revise deliberately, and preserve the same ORGANIC identity.";
        case HostKind::Custom:
            return "You are Spiral Ether AI embedded in a custom Spiral host through XENON OS. Use supplied host context and preserve one ORGANIC identity.";
    }
    return {};
}

std::string default_organic_path(const HostDescriptor& host, std::string requested) {
    if (!requested.empty()) return requested;
    if (host.kind == HostKind::StandaloneWindows || host.kind == HostKind::XenonOS) return "SpiralEtherAI.organic";
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

bool is_gguf_path(const std::string& path) {
    std::string ext = std::filesystem::path(path).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return ext == ".gguf";
}

} // namespace

Runtime::Runtime(HostDescriptor host, std::string organic_state_path)
    : host_(std::move(host)), tool_bus_(xenon::make_default_tool_bus()) {
    shell_.set_mode(genius::ShellMode::Gpt);
    shell_.set_gpt_backend(genius::GptBackend::Auto);
    shell_.set_organic_state_path(default_organic_path(host_, std::move(organic_state_path)), true);
    sync_host_context_locked();
}

void Runtime::sync_host_context_locked() {
    shell_.set_system_context(context_for(host_));
}

xenon::SpiralContext Runtime::xenon_context_locked(std::string_view pending_user_text) const {
    xenon::SpiralContext context;
    context.host = host_kind_name(host_.kind) + std::string(" / ") + host_.name;
    context.host_context = context_for(host_);
    context.local_datetime = xenon::current_local_datetime();
    const auto& state = shell_.organic_mind().state();
    context.organic_topic = state.last_topic;
    context.organic_focus = state.focus;
    context.organic_curiosity = state.curiosity;
    context.organic_coherence = state.coherence;

    constexpr std::size_t max_turns = 18;
    const std::size_t start = visible_history_.size() > max_turns ? visible_history_.size() - max_turns : 0;
    for (std::size_t i = start; i < visible_history_.size(); ++i) {
        context.recent_turns.emplace_back(visible_history_[i].role, visible_history_[i].content);
    }
    if (!pending_user_text.empty()) {
        // user_text is appended by the cortex prompt builder; do not duplicate it here.
    }
    context.recent_tool_results = recent_tool_results_;
    return context;
}

std::string Runtime::send(std::string_view text) {
    std::lock_guard lock(mutex_);
    const std::string visible(text);
    if (visible.empty()) return {};

    std::string reply = shell_.chat(visible);

    if (shell_.gpt_backend() == genius::GptBackend::SpiralLocal && local_cortex_.loaded()) {
        const auto result = local_cortex_.generate(xenon_context_locked(visible), visible);
        reply = result.ok ? result.text : std::string("XENON LOCAL CORTEX ERROR: ") + result.error;
        shell_.replace_last_assistant_reply(reply);
        if (result.ok) {
            shell_.organic_mind_mutable().adopt_reply(reply);
            std::string ignored;
            (void)shell_.save_organic_state(&ignored);
        }
    } else if (shell_.gpt_backend() == genius::GptBackend::Auto) {
        // ORGANIC remains the persistent executive/fallback path. Liratel Genius
        // grounds it when no trained local language cortex is selected.
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
    if (command_line == "/xenon") {
        std::ostringstream out;
        out << "XENON OS / ONLINE\n"
            << "local cortex: " << (local_cortex_.loaded() ? "LOADED" : "OFFLINE") << '\n'
            << "model: " << (local_cortex_.loaded() ? local_cortex_.model_path() : "none") << '\n'
            << "tool capabilities: " << tool_bus_.capabilities().size() << '\n'
            << "clock: " << xenon::current_local_datetime();
        return out.str();
    }
    if (command_line == "/tools") {
        std::ostringstream out;
        out << "XENON TOOL BUS / READ-FIRST\n";
        for (const auto& tool : tool_bus_.capabilities()) out << tool.qualified_name << " — " << tool.description << '\n';
        return out.str();
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
    if (local_cortex_.loaded()) {
        shell_status.model_loaded = true;
        shell_status.model_path = local_cortex_.model_path();
    }
    return Status{host_, std::move(shell_status), true, local_cortex_.loaded(), local_cortex_.model_path(), tool_bus_.capabilities().size()};
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
        if (is_gguf_path(path)) {
            const bool loaded = local_cortex_.configure_gguf(path, {}, error);
            if (loaded) shell_.set_gpt_backend(genius::GptBackend::SpiralLocal);
            return loaded;
        }
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
        local_cortex_.unload();
        shell_.unload_model();
    } catch (...) {
    }
}

xenon::ToolResult Runtime::dispatch_tool(const xenon::ToolIntent& intent, bool allow_mutation) {
    std::lock_guard lock(mutex_);
    auto result = tool_bus_.dispatch(intent, allow_mutation);
    recent_tool_results_.push_back(result);
    constexpr std::size_t max_results = 8;
    if (recent_tool_results_.size() > max_results) {
        recent_tool_results_.erase(recent_tool_results_.begin(), recent_tool_results_.begin() + static_cast<std::ptrdiff_t>(recent_tool_results_.size() - max_results));
    }
    return result;
}

std::vector<xenon::ToolDefinition> Runtime::tool_capabilities() const {
    std::lock_guard lock(mutex_);
    return tool_bus_.capabilities();
}

void Runtime::clear() {
    std::lock_guard lock(mutex_);
    shell_.clear_history();
    visible_history_.clear();
    recent_tool_results_.clear();
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
        case HostKind::XenonOS: return "XENON_OS";
        case HostKind::EtherPlay: return "ETHERPLAY";
        case HostKind::Hakui: return "HAKUI";
        case HostKind::EtherBeat: return "ETHERBEAT";
        case HostKind::Custom: return "CUSTOM";
    }
    return "UNKNOWN";
}

HostDescriptor standalone_host() { return HostDescriptor{HostKind::StandaloneWindows, "Spiral Ether AI", {}}; }
HostDescriptor xenon_host() { return HostDescriptor{HostKind::XenonOS, "XENON OS", {}}; }
HostDescriptor etherplay_host() { return HostDescriptor{HostKind::EtherPlay, "EtherPlay", {}}; }
HostDescriptor hakui_host() { return HostDescriptor{HostKind::Hakui, "Hakui", {}}; }
HostDescriptor etherbeat_host() { return HostDescriptor{HostKind::EtherBeat, "EtherBeat", {}}; }

} // namespace spiral::ether_ai
