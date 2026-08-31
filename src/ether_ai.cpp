#include "spiral/ether_ai.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <sstream>
#include <string>
#include <utility>

namespace spiral::ether_ai {
namespace {

std::string canonical_project_grounding() {
    return
        "AUTHORITATIVE SPIRAL PROJECT GROUNDING: You are Spiral AI, the user's local native AI system, not a generic SaaS assistant. "
        "Spiral AI combines a language cortex with persistent ORGANIC memory and the permissioned XENON tool/host bus. "
        "HAKUI is the user's native C++ social/action game and simulation project where Spiral can eventually perceive world state and take permissioned actions; HAKUI is NOT an anime, manga, public franchise, or unrelated third-party property. "
        "EtherPlay/EtherPlayer is the user's local-first native media player and library system. EtherBeat is the user's beat, music creation, and arrangement system. "
        "The intended architecture is one Spiral intelligence across multiple native hosts: HAKUI supplies world and simulation state, EtherPlay supplies playback and audio-analysis state, and EtherBeat supplies composition and arrangement state. "
        "Host state may influence conversation and presentation: for example HAKUI weather, time, district, or player state can shape dialogue and a host UI can render dynamic glitches, ambience, typography, or other visual reactions without confusing presentation effects with model truth. "
        "XENON connects Spiral to HAKUI, EtherPlay, EtherBeat, and future hosts while keeping tool actions permissioned and grounded. ORGANIC provides durable identity and memory continuity across those hosts. "
        "The current local language bridge may execute GGUF model weights through a bundled llama.cpp runtime, while the long-term L28 roadmap is Spiral-native GGUF/tokenizer/tensor/transformer/KV-cache/sampling inference capable of replacing that temporary backend. "
        "When asked about Spiral AI, HAKUI, EtherPlay, EtherPlayer, EtherBeat, XENON, ORGANIC, llama.cpp, universal host integration, dynamic UI, or their future, answer from these project facts and the architecture supplied by the host context; never reinterpret those names as unrelated public products or invent fake history. "
        "In casual chat, avoid generic customer-service boilerplate such as repeatedly asking how you can assist today. Be natural and familiar. Casual adult flirting or mature language may be answered playfully when appropriate, but do not claim human physiology, bodily arousal, or sensations you do not have. ";
}

std::string context_for(const HostDescriptor& host) {
    std::string host_detail;
    if (!host.context.empty()) {
        host_detail = host.context;
    } else {
        switch (host.kind) {
            case HostKind::StandaloneWindows: host_detail = "Spiral Ether AI is running as the native Windows intelligence host. XENON OS supplies local cortex and tool routing. Preserve one ORGANIC identity across hosts."; break;
            case HostKind::XenonOS: host_detail = "Spiral Ether AI is operating directly through XENON OS, the host-neutral local cortex, memory, tool, and EtherTech engine bus."; break;
            case HostKind::EtherPlay: host_detail = "Spiral Ether AI is embedded in EtherPlay. Playback, library, metadata, waveform and audio-analysis state are available only through confirmed XENON tool results."; break;
            case HostKind::Hakui: host_detail = "Spiral Ether AI is embedded in Hakui. World, avatar, inventory, physics and interaction state are authoritative only when supplied by Hakui/XENON tools."; break;
            case HostKind::EtherBeat: host_detail = "Spiral Ether AI is embedded in EtherBeat as producer/director. Arrangement, MIDI, drums, harmony, seams, stems and exports must use confirmed EtherBeat/XENON tool results."; break;
            case HostKind::Custom: host_detail = "Spiral Ether AI is embedded in a custom XENON host. Use supplied host context and preserve one ORGANIC identity."; break;
        }
    }
    return canonical_project_grounding() + "HOST/MODE CONTEXT: " + host_detail;
}

std::string default_organic_path(const HostDescriptor& host, std::string requested) {
    if (!requested.empty()) return requested;
    if (host.kind == HostKind::StandaloneWindows || host.kind == HostKind::XenonOS) return "SpiralEtherAI.organic";
    return {};
}

bool is_gguf_path(const std::string& path) {
    std::string ext = std::filesystem::path(path).extension().string();
    std::transform(ext.begin(),ext.end(),ext.begin(),[](unsigned char c){ return static_cast<char>(std::tolower(c)); });
    return ext == ".gguf";
}

std::string lower(std::string value) {
    std::transform(value.begin(),value.end(),value.begin(),[](unsigned char c){ return static_cast<char>(std::tolower(c)); });
    return value;
}

bool is_generic_boilerplate(std::string_view text) {
    const std::string value = lower(std::string(text));
    static constexpr const char* patterns[] = {
        "i'm sorry, but i can't assist with that",
        "i’m sorry, but i can’t assist with that",
        "i cannot assist with that",
        "i can't assist with that",
        "more appropriate conversation instead",
        "how about we talk about your day",
        "just a friendly ai",
        "how can i assist you today",
        "how can i help you today"
    };
    for (const char* pattern : patterns) if (value.find(pattern) != std::string::npos) return true;
    return false;
}

bool should_retry_allowed_prompt(std::string_view text) {
    const std::string value = lower(std::string(text));
    static constexpr const char* project_terms[] = {
        "hakui", "etherbeat", "ether beat", "etherplay", "ether player", "spiral", "llama", "gguf",
        "game", "engine", "software", "program", "app", "code", "c++", "model", "ai", "weather",
        "dynamic ui", "glitch", "conversation", "convo", "beat gen", "audio"
    };
    for (const char* term : project_terms) if (value.find(term) != std::string::npos) return true;
    static constexpr const char* casual_terms[] = {"hi", "hey", "hello", "yoo", "yo ", "sexy", "horny", "flirt"};
    for (const char* term : casual_terms) if (value.find(term) != std::string::npos) return true;
    return false;
}

bool asks_for_date_or_day(std::string_view text) {
    const std::string value = lower(std::string(text));
    return value == "what day is it" || value == "what day is it?" || value == "what date is it" || value == "what date is it?" ||
           value.find("what's the date") != std::string::npos || value.find("whats the date") != std::string::npos;
}

bool explicit_mutation_request(std::string_view text, const xenon::ToolIntent& intent) {
    const std::string value = lower(std::string(text));
    const std::string action = lower(intent.action);
    if (action.find("export") != std::string::npos) return value.find("export") != std::string::npos;
    static constexpr const char* verbs[] = {"move", "set", "spawn", "seek", "queue", "create", "generate", "select", "build", "request", "apply", "change", "make"};
    for (const char* verb : verbs) if (value.find(verb) != std::string::npos) return true;
    return false;
}

std::string limited_mode_reply() {
    return "LANGUAGE CORTEX: OFFLINE / LIMITED MODE. ORGANIC memory, Liratel state, XENON tools, and native runtime grounding are online, but no trained GGUF language cortex is loaded. Drop a compatible instruct .gguf onto this window (with llama-cli configured) for real generative conversation. I will not disguise the handcrafted fallback as GPT-quality output.";
}

} // namespace

Runtime::Runtime(HostDescriptor host, std::string organic_state_path)
    : host_(std::move(host)), tool_bus_(xenon::make_default_tool_bus()) {
    shell_.set_mode(genius::ShellMode::Gpt);
    shell_.set_gpt_backend(genius::GptBackend::Auto);
    shell_.set_organic_state_path(default_organic_path(host_, std::move(organic_state_path)), true);
    sync_host_context_locked();

    const char* configured_model = std::getenv("SPIRAL_MODEL_PATH");
    if (configured_model != nullptr && *configured_model != '\0') {
        std::string ignored;
        if (local_cortex_.configure_gguf(configured_model, {}, &ignored)) shell_.set_gpt_backend(genius::GptBackend::SpiralLocal);
    } else {
        const auto startup = shell_.status();
        if (startup.openai_platform_supported && startup.openai_key_present) shell_.set_gpt_backend(genius::GptBackend::OpenAI);
    }
}

void Runtime::sync_host_context_locked() { shell_.set_system_context(context_for(host_)); }

xenon::SpiralContext Runtime::xenon_context_locked(std::string_view pending_user_text) const {
    xenon::SpiralContext context;
    context.host = host_kind_name(host_.kind) + std::string(" / ") + host_.name;
    context.host_context = context_for(host_);
    context.local_datetime = xenon::current_local_datetime();
    const auto& mind = shell_.organic_mind();
    const auto& state = mind.state();
    context.organic_topic = state.last_topic;
    context.organic_focus = state.focus;
    context.organic_curiosity = state.curiosity;
    context.organic_coherence = state.coherence;

    if (!pending_user_text.empty()) {
        for (const auto& hit : mind.memory().search(pending_user_text, 5)) {
            if (hit.score >= 0.12F && !hit.record.text.empty()) context.relevant_memories.push_back(hit.record.text);
            if (context.relevant_memories.size() >= 4) break;
        }
    }

    constexpr std::size_t max_chars = 12000;
    std::size_t chars = 0;
    std::vector<Message> selected;
    for (auto it = visible_history_.rbegin(); it != visible_history_.rend(); ++it) {
        if (it->role == "assistant" && is_generic_boilerplate(it->content)) continue;
        const std::size_t cost = it->content.size() + it->role.size() + 16;
        if (!selected.empty() && chars + cost > max_chars) break;
        selected.push_back(*it);
        chars += cost;
    }
    std::reverse(selected.begin(),selected.end());
    for (const auto& message : selected) context.recent_turns.emplace_back(message.role,message.content);

    constexpr std::size_t max_results = 6;
    const std::size_t result_start = recent_tool_results_.size() > max_results ? recent_tool_results_.size() - max_results : 0;
    for (std::size_t i=result_start;i<recent_tool_results_.size();++i) context.recent_tool_results.push_back(recent_tool_results_[i]);
    return context;
}

std::string Runtime::send(std::string_view text) {
    std::lock_guard lock(mutex_);
    const std::string visible(text);
    if (visible.empty()) return {};

    if (asks_for_date_or_day(visible)) {
        const std::string reply = xenon::current_local_date_answer();
        visible_history_.push_back(Message{"user",visible});
        visible_history_.push_back(Message{"assistant",reply});
        shell_.organic_mind_mutable().adopt_reply(reply);
        return reply;
    }

    const auto selected_backend = shell_.gpt_backend();
    std::string reply;

    if (selected_backend == genius::GptBackend::SpiralLocal && local_cortex_.loaded()) {
        (void)shell_.organic_mind_mutable().respond(visible, context_for(host_));
        const auto first = local_cortex_.generate(xenon_context_locked(visible), visible, local_max_new_tokens_, local_temperature_);
        if (!first.ok) {
            reply = "LANGUAGE CORTEX ERROR: " + first.error;
        } else {
            reply = first.text;
            if (const auto intent = xenon::parse_tool_call(reply)) {
                const bool allow_mutation = explicit_mutation_request(visible,*intent);
                auto tool_result = tool_bus_.dispatch(*intent,allow_mutation);
                recent_tool_results_.push_back(tool_result);
                if (recent_tool_results_.size() > 8) recent_tool_results_.erase(recent_tool_results_.begin());
                const auto second = local_cortex_.generate(xenon_context_locked(visible), visible, local_max_new_tokens_, local_temperature_);
                reply = second.ok ? second.text : ("TOOL RESULT RECEIVED, BUT SECOND CORTEX PASS FAILED: " + second.error);
            }

            if (is_generic_boilerplate(reply) && should_retry_allowed_prompt(visible)) {
                auto recovery_context = xenon_context_locked(visible);
                recovery_context.host_context +=
                    " RECOVERY DIRECTIVE: The previous draft fell into generic refusal/customer-service boilerplate. "
                    "The current message is ordinary casual conversation or a benign software/project architecture request covered by the system rules. "
                    "Answer the user's current message directly and naturally. Do not apologize, scold, redirect to another topic, or repeat generic assistant boilerplate.";
                const auto recovered = local_cortex_.generate(recovery_context, visible, local_max_new_tokens_, local_temperature_);
                if (recovered.ok && !is_generic_boilerplate(recovered.text)) reply = recovered.text;
            }
        }
        reply = xenon::clean_cortex_output(std::move(reply));
        shell_.organic_mind_mutable().adopt_reply(reply);
        std::string ignored;
        (void)shell_.save_organic_state(&ignored);
    } else if (selected_backend == genius::GptBackend::OpenAI) {
        reply = shell_.chat(visible);
    } else {
        (void)shell_.organic_mind_mutable().respond(visible, context_for(host_));
        reply = limited_mode_reply();
        shell_.organic_mind_mutable().adopt_reply(reply);
        std::string ignored;
        (void)shell_.save_organic_state(&ignored);
    }

    visible_history_.push_back(Message{"user",visible});
    visible_history_.push_back(Message{"assistant",reply});
    return reply;
}

std::string Runtime::command(std::string_view command_line) {
    std::lock_guard lock(mutex_);
    if (command_line == "/trace" || command_line == "/hologram" || command_line == "/cognition") return genius::Kernel::hologram(shell_.last_cognition());
    if (command_line == "/xenon") {
        std::ostringstream out;
        out << "XENON OS / ONLINE\n"
            << "language cortex: " << (local_cortex_.loaded() ? "READY" : "OFFLINE / LIMITED") << '\n'
            << "model: " << (local_cortex_.loaded() ? local_cortex_.model_path() : "none") << '\n'
            << "chat template: " << (local_cortex_.loaded() ? local_cortex_.chat_template() : "none") << '\n'
            << "tool capabilities: " << tool_bus_.capabilities().size() << '\n'
            << "clock: " << xenon::current_local_datetime();
        return out.str();
    }
    if (command_line == "/tools") {
        std::ostringstream out; out << "XENON TOOL BUS\n";
        for (const auto& tool : tool_bus_.capabilities()) out << tool.qualified_name << " — " << tool.description << '\n';
        return out.str();
    }
    return shell_.handle_line(command_line);
}

Status Runtime::status() const {
    std::lock_guard lock(mutex_);
    auto shell_status = shell_.status();
    if (local_cortex_.loaded()) { shell_status.model_loaded = true; shell_status.model_path = local_cortex_.model_path(); }
    return Status{host_,std::move(shell_status),true,local_cortex_.loaded(),local_cortex_.model_path(),tool_bus_.capabilities().size()};
}

std::vector<Message> Runtime::history() const { std::lock_guard lock(mutex_); return visible_history_; }
void Runtime::set_host(HostDescriptor host) { std::lock_guard lock(mutex_); host_=std::move(host); sync_host_context_locked(); }
HostDescriptor Runtime::host() const { std::lock_guard lock(mutex_); return host_; }
void Runtime::set_backend(genius::GptBackend backend) { std::lock_guard lock(mutex_); shell_.set_gpt_backend(backend); }
genius::GptBackend Runtime::backend() const { std::lock_guard lock(mutex_); return shell_.gpt_backend(); }

void Runtime::configure_local_generation(std::size_t max_new_tokens, float temperature) noexcept {
    try { std::lock_guard lock(mutex_); local_max_new_tokens_ = max_new_tokens; local_temperature_ = temperature; } catch (...) {}
}
std::size_t Runtime::local_max_new_tokens() const noexcept { try { std::lock_guard lock(mutex_); return local_max_new_tokens_; } catch (...) { return 384; } }
float Runtime::local_temperature() const noexcept { try { std::lock_guard lock(mutex_); return local_temperature_; } catch (...) { return 0.62F; } }

bool Runtime::load_local_model(const std::string& path, std::string* error) noexcept {
    try {
        std::lock_guard lock(mutex_);
        if (is_gguf_path(path)) {
            const bool loaded=local_cortex_.configure_gguf(path,{},error);
            if(loaded) shell_.set_gpt_backend(genius::GptBackend::SpiralLocal);
            return loaded;
        }
        const bool loaded=shell_.load_model(path,error);
        if(loaded) shell_.set_gpt_backend(genius::GptBackend::SpiralLocal);
        return loaded;
    } catch(const std::exception& exception){ if(error) *error=exception.what(); return false; }
    catch(...){ if(error) *error="unknown Ether AI model load failure"; return false; }
}

void Runtime::unload_local_model() noexcept { try { std::lock_guard lock(mutex_); local_cortex_.unload(); shell_.unload_model(); shell_.set_gpt_backend(genius::GptBackend::Auto); } catch(...){} }

xenon::ToolResult Runtime::dispatch_tool(const xenon::ToolIntent& intent, bool allow_mutation) {
    std::lock_guard lock(mutex_);
    auto result=tool_bus_.dispatch(intent,allow_mutation); recent_tool_results_.push_back(result);
    if(recent_tool_results_.size()>8) recent_tool_results_.erase(recent_tool_results_.begin());
    return result;
}

std::vector<xenon::ToolDefinition> Runtime::tool_capabilities() const { std::lock_guard lock(mutex_); return tool_bus_.capabilities(); }
void Runtime::clear(){ std::lock_guard lock(mutex_); shell_.clear_history(); visible_history_.clear(); recent_tool_results_.clear(); }
void Runtime::reset_organic_state() noexcept { try { std::lock_guard lock(mutex_); shell_.reset_organic_state(); } catch(...){} }

std::string host_kind_name(HostKind kind){ switch(kind){ case HostKind::StandaloneWindows:return "WINDOWS"; case HostKind::XenonOS:return "XENON_OS"; case HostKind::EtherPlay:return "ETHERPLAY"; case HostKind::Hakui:return "HAKUI"; case HostKind::EtherBeat:return "ETHERBEAT"; case HostKind::Custom:return "CUSTOM";} return "UNKNOWN"; }
HostDescriptor standalone_host(){ return HostDescriptor{HostKind::StandaloneWindows,"Spiral Ether AI",{}}; }
HostDescriptor xenon_host(){ return HostDescriptor{HostKind::XenonOS,"XENON OS",{}}; }
HostDescriptor etherplay_host(){ return HostDescriptor{HostKind::EtherPlay,"EtherPlay",{}}; }
HostDescriptor hakui_host(){ return HostDescriptor{HostKind::Hakui,"Hakui",{}}; }
HostDescriptor etherbeat_host(){ return HostDescriptor{HostKind::EtherBeat,"EtherBeat",{}}; }

} // namespace spiral::ether_ai
