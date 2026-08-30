#include "spiral/genius_shell.hpp"

#include "spiral/tokenizer.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cctype>
#include <iomanip>
#include <sstream>
#include <string>
#include <system_error>

namespace spiral::genius {
namespace {

std::string clean_generated_text(std::string text) {
    for (char& ch : text) {
        const unsigned char value = static_cast<unsigned char>(ch);
        if (value < 0x20U && ch != '\n' && ch != '\r' && ch != '\t') ch = ' ';
    }
    while (!text.empty() && std::isspace(static_cast<unsigned char>(text.back())) != 0) text.pop_back();
    return text;
}

std::string maybe_unquote(std::string value) {
    if (value.size() >= 2 && ((value.front() == '"' && value.back() == '"') ||
                            (value.front() == '\'' && value.back() == '\''))) {
        return value.substr(1, value.size() - 2);
    }
    return value;
}

} // namespace

GeniusShell::GeniusShell() : openai_model_(openai::ResponsesBackend::default_model()) {
    generation_.max_new_tokens = 128;
    generation_.max_context_tokens = 1024;
    generation_.sampling.temperature = 0.8F;
    generation_.sampling.top_k = 40;
    generation_.sampling.top_p = 0.95F;
    generation_.sampling.repetition_penalty = 1.05F;

    gpu_device_ = gpu::D3D11GpuDevice::try_create(&gpu_error_);
    if (gpu_device_ != nullptr) {
        try {
            gpu_compute_ = std::make_unique<gpu::D3D11ComputeEngine>(*gpu_device_);
        } catch (const std::exception& exception) {
            gpu_error_ = exception.what();
            gpu_compute_.reset();
        }
    }
}

ShellStatus GeniusShell::status() const {
    ShellStatus result;
    result.mode = mode_;
    result.gpt_backend = gpt_backend_;
    result.gpu_platform_supported = gpu::D3D11GpuDevice::platform_supported();
    result.gpu_available = gpu_device_ != nullptr && gpu_compute_ != nullptr && gpu_compute_->available();
    if (gpu_device_ != nullptr) {
        const auto capabilities = gpu_device_->capabilities();
        result.gpu_hardware_accelerated = capabilities.hardware_accelerated;
        result.gpu_adapter = capabilities.adapter_name;
        result.gpu_feature_level = capabilities.feature_level;
    }
    result.openai_platform_supported = openai::ResponsesBackend::platform_supported();
    result.openai_key_present = openai::ResponsesBackend::api_key_present();
    result.openai_model = openai_model_;
    result.model_loaded = model_bundle_ != nullptr && model_bundle_->model != nullptr;
    result.model_path = model_path_;
    result.conversation_turns = history_.size();
    result.max_new_tokens = generation_.max_new_tokens;
    result.temperature = generation_.sampling.temperature;

    const auto& organic_state = organic_mind_.state();
    result.organic_revision = organic_state.revision;
    result.organic_turns = organic_state.turn_count;
    result.organic_memories = organic_mind_.memory_count();
    result.organic_energy = organic_state.energy;
    result.organic_focus = organic_state.focus;
    result.organic_curiosity = organic_state.curiosity;
    result.organic_confidence = organic_state.confidence;
    result.organic_warmth = organic_state.warmth;
    result.organic_novelty = organic_state.novelty;
    result.organic_coherence = organic_state.coherence;
    result.organic_topic = organic_state.last_topic;
    return result;
}

void GeniusShell::set_organic_state_path(std::string path, bool load_existing) noexcept {
    organic_state_path_ = std::move(path);
    if (load_existing && !organic_state_path_.empty()) {
        std::string ignored;
        (void)load_organic_state(&ignored);
    }
}

bool GeniusShell::save_organic_state(std::string* error) const noexcept {
    if (organic_state_path_.empty()) {
        if (error != nullptr) error->clear();
        return true;
    }
    try {
        organic_mind_.save(organic_state_path_);
        if (error != nullptr) error->clear();
        return true;
    } catch (const std::exception& exception) {
        if (error != nullptr) *error = exception.what();
        return false;
    } catch (...) {
        if (error != nullptr) *error = "unknown organic state save failure";
        return false;
    }
}

bool GeniusShell::load_organic_state(std::string* error) noexcept {
    if (organic_state_path_.empty()) {
        if (error != nullptr) *error = "organic state path is not configured";
        return false;
    }
    try {
        organic_mind_.load(organic_state_path_);
        if (error != nullptr) error->clear();
        return true;
    } catch (const std::exception& exception) {
        if (error != nullptr) *error = exception.what();
        return false;
    } catch (...) {
        if (error != nullptr) *error = "unknown organic state load failure";
        return false;
    }
}

void GeniusShell::reset_organic_state() noexcept {
    organic_mind_.reset();
    std::string ignored;
    (void)save_organic_state(&ignored);
}

bool GeniusShell::load_model(const std::string& path, std::string* error) noexcept {
    try {
        auto loaded = runtime::load_model_bundle(path);
        model_bundle_ = std::make_unique<runtime::LoadedModelBundle>(std::move(loaded));
        model_path_ = path;
        if (error != nullptr) error->clear();
        return true;
    } catch (const std::exception& exception) {
        if (error != nullptr) *error = exception.what();
        return false;
    } catch (...) {
        if (error != nullptr) *error = "unknown model bundle load failure";
        return false;
    }
}

void GeniusShell::unload_model() noexcept {
    model_bundle_.reset();
    model_path_.clear();
}

std::string GeniusShell::trim(std::string_view text) {
    std::size_t begin = 0;
    while (begin < text.size() && std::isspace(static_cast<unsigned char>(text[begin])) != 0) ++begin;
    std::size_t end = text.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(text[end - 1])) != 0) --end;
    return std::string(text.substr(begin, end - begin));
}

std::optional<float> GeniusShell::parse_float(std::string_view text) {
    const std::string value = trim(text);
    if (value.empty()) return std::nullopt;
    try {
        std::size_t consumed = 0;
        const float parsed = std::stof(value, &consumed);
        if (consumed != value.size() || !std::isfinite(parsed)) return std::nullopt;
        return parsed;
    } catch (...) {
        return std::nullopt;
    }
}

std::optional<std::size_t> GeniusShell::parse_size(std::string_view text) {
    const std::string value = trim(text);
    if (value.empty()) return std::nullopt;
    std::size_t parsed = 0;
    const auto result = std::from_chars(value.data(), value.data() + value.size(), parsed);
    if (result.ec != std::errc{} || result.ptr != value.data() + value.size()) return std::nullopt;
    return parsed;
}

std::string GeniusShell::handle_line(std::string_view line) {
    const std::string normalized = trim(line);
    if (normalized.empty()) return {};
    if (normalized.front() == '/') return run_command(normalized);
    return chat(normalized);
}

std::string GeniusShell::chat(std::string_view user_message) {
    const std::string message = trim(user_message);
    if (message.empty()) return {};
    history_.push_back(ChatTurn{"user", message});
    std::string reply = mode_ == ShellMode::Gpt ? gpt_reply() : native_reply(message);
    history_.push_back(ChatTurn{"assistant", reply});
    return reply;
}

std::string GeniusShell::build_chat_prompt() const {
    std::ostringstream prompt;
    prompt << "SYSTEM: You are the language cortex plugged into Spiral Ether AI.\n";
    if (!system_context_.empty()) prompt << "SYSTEM: " << system_context_ << '\n';
    const auto& state = organic_mind_.state();
    prompt << "SYSTEM: Organic state topic=" << state.last_topic
           << " focus=" << state.focus << " curiosity=" << state.curiosity
           << " coherence=" << state.coherence << ". Preserve this continuity.\n";
    constexpr std::size_t max_turns = 16;
    const std::size_t start = history_.size() > max_turns ? history_.size() - max_turns : 0;
    for (std::size_t i = start; i < history_.size(); ++i) {
        prompt << (history_[i].role == "assistant" ? "ASSISTANT: " : "USER: ") << history_[i].content << '\n';
    }
    prompt << "ASSISTANT: ";
    return prompt.str();
}

std::string GeniusShell::build_openai_input() const {
    std::ostringstream input;
    constexpr std::size_t max_turns = 24;
    const std::size_t start = history_.size() > max_turns ? history_.size() - max_turns : 0;
    for (std::size_t i = start; i < history_.size(); ++i) {
        input << (history_[i].role == "assistant" ? "ASSISTANT: " : "USER: ") << history_[i].content << '\n';
    }
    input << "ASSISTANT: ";
    return input.str();
}

std::string GeniusShell::openai_instructions() const {
    std::ostringstream out;
    out << "You are an optional language bridge inside Spiral Ether AI, not its persistent mind. Respond helpfully and directly.";
    if (!system_context_.empty()) out << " Host context: " << system_context_;
    return out.str();
}

std::string GeniusShell::organic_reply() {
    const std::string input = history_.empty() ? std::string{} : history_.back().content;
    auto response = organic_mind_.respond(input, system_context_);
    std::string ignored;
    (void)save_organic_state(&ignored);
    return response.text;
}

std::string GeniusShell::local_spiral_reply() {
    if (model_bundle_ == nullptr || model_bundle_->model == nullptr) {
        return "LOCAL SPIRAL cortex has no trained language-model bundle loaded. Organic Spiral remains online; use /backend organic to return to it.";
    }
    try {
        generate::ByteTextGenerator generator(*model_bundle_->model);
        auto result = generator.generate(build_chat_prompt(), generation_);
        std::string text = clean_generated_text(std::move(result.text));
        return text.empty() ? "[loaded Spiral cortex emitted no text]" : text;
    } catch (const std::exception& exception) {
        return std::string("LOCAL SPIRAL generation error: ") + exception.what();
    }
}

std::string GeniusShell::openai_gpt_reply() {
    if (!openai::ResponsesBackend::platform_supported()) return "OPENAI bridge is unavailable on this platform.";
    if (!openai::ResponsesBackend::api_key_present()) return "OPENAI bridge selected, but OPENAI_API_KEY is not set. Organic Spiral does not require it.";
    const auto response = openai_backend_.respond(openai_instructions(), build_openai_input(), openai_model_);
    if (!response.ok) {
        std::ostringstream out;
        out << "OPENAI bridge error";
        if (response.http_status != 0) out << " (HTTP " << response.http_status << ')';
        out << ": " << response.error;
        return out.str();
    }
    return response.text.empty() ? "[OpenAI bridge returned no text]" : response.text;
}

std::string GeniusShell::gpt_reply() {
    switch (gpt_backend_) {
        case GptBackend::Auto: return organic_reply();
        case GptBackend::OpenAI: return openai_gpt_reply();
        case GptBackend::SpiralLocal: return local_spiral_reply();
    }
    return organic_reply();
}

std::string GeniusShell::native_reply(std::string_view user_message) {
    ByteTokenizer tokenizer;
    const auto tokens = tokenizer.encode(user_message);
    const auto current = status();
    std::ostringstream out;
    out << "NATIVE MODE / substrate inspection\n";
    out << "input bytes: " << user_message.size() << " | byte tokens: " << tokens.size() << '\n';
    out << "organic mind: ONLINE | revision " << current.organic_revision << " | memories " << current.organic_memories << '\n';
    out << "gpu compute: " << (current.gpu_available ? "online" : "offline");
    if (current.gpu_available) out << " | " << (current.gpu_hardware_accelerated ? "hardware" : "software/WARP") << " | " << current.gpu_adapter;
    out << "\nlocal language cortex: " << (current.model_loaded ? current.model_path : "none")
        << "\nOpenAI bridge: " << (current.openai_key_present ? "available if explicitly selected" : "not configured")
        << "\nUse /gpt for conversational mode.";
    return out.str();
}

std::string GeniusShell::run_command(std::string_view line) {
    const std::size_t split = line.find_first_of(" \t");
    const std::string command = std::string(line.substr(0, split));
    const std::string argument = split == std::string_view::npos ? std::string{} : trim(line.substr(split + 1));

    if (command == "/help" || command == "/?") return help_text();
    if (command == "/status") return status_text();
    if (command == "/gpt") { mode_ = ShellMode::Gpt; return "CHAT MODE enabled. Brain = " + gpt_backend_name(gpt_backend_) + "."; }
    if (command == "/native") { mode_ = ShellMode::Native; return "NATIVE MODE enabled — runtime/substrate inspection."; }
    if (command == "/mode") {
        if (argument == "gpt" || argument == "chat") { mode_ = ShellMode::Gpt; return "mode = CHAT"; }
        if (argument == "native") { mode_ = ShellMode::Native; return "mode = NATIVE"; }
        return "usage: /mode chat|native";
    }
    if (command == "/backend") {
        if (argument.empty()) return "brain = " + gpt_backend_name(gpt_backend_);
        if (argument == "auto" || argument == "organic" || argument == "native") { gpt_backend_ = GptBackend::Auto; return "brain = ORGANIC"; }
        if (argument == "openai" || argument == "gpt") { gpt_backend_ = GptBackend::OpenAI; return "brain = OPENAI_BRIDGE (explicit network mode)"; }
        if (argument == "spiral" || argument == "local") { gpt_backend_ = GptBackend::SpiralLocal; return "brain = SPIRAL_LOCAL_CORTEX"; }
        return "usage: /backend organic|spiral|openai";
    }
    if (command == "/organic" || command == "/mind") {
        const auto current = status();
        std::ostringstream out;
        out << "ORGANIC MIND / ONLINE\n"
            << "revision: " << current.organic_revision << " | turns: " << current.organic_turns << " | memories: " << current.organic_memories << '\n'
            << "topic: " << (current.organic_topic.empty() ? "none" : current.organic_topic) << '\n'
            << std::fixed << std::setprecision(2)
            << "energy " << current.organic_energy << " | focus " << current.organic_focus << " | curiosity " << current.organic_curiosity << '\n'
            << "confidence " << current.organic_confidence << " | warmth " << current.organic_warmth
            << " | novelty " << current.organic_novelty << " | coherence " << current.organic_coherence << '\n'
            << "persistence: " << (organic_state_path_.empty() ? "session-only" : organic_state_path_);
        return out.str();
    }
    if (command == "/resetorganic") { reset_organic_state(); return "organic state + durable memory reset."; }
    if (command == "/openai") {
        const auto current = status();
        std::ostringstream out;
        out << "OPENAI BRIDGE: " << (current.openai_key_present && current.openai_platform_supported ? "AVAILABLE" : "NOT READY") << '\n'
            << "This bridge is optional and never used by ORGANIC mode.\n"
            << "platform: " << (current.openai_platform_supported ? "supported" : "unsupported") << '\n'
            << "OPENAI_API_KEY: " << (current.openai_key_present ? "present" : "not set") << '\n'
            << "model: " << current.openai_model;
        return out.str();
    }
    if (command == "/gptmodel") {
        if (argument.empty()) return "OpenAI bridge model = " + openai_model_;
        openai_model_ = argument;
        return "OpenAI bridge model = " + openai_model_;
    }
    if (command == "/gpu") {
        const auto current = status();
        std::ostringstream out;
        out << "GPU: " << (current.gpu_available ? "ONLINE" : "OFFLINE") << '\n';
        out << "platform: " << (current.gpu_platform_supported ? "D3D11 supported" : "D3D11 unavailable") << '\n';
        if (current.gpu_available) {
            out << "adapter: " << current.gpu_adapter << '\n';
            out << "feature level: " << current.gpu_feature_level << '\n';
            out << "execution: " << (current.gpu_hardware_accelerated ? "physical hardware" : "software/WARP compatibility");
        } else if (!gpu_error_.empty()) out << "reason: " << gpu_error_;
        return out.str();
    }
    if (command == "/model") {
        const auto current = status();
        if (!current.model_loaded) return "LOCAL CORTEX: none loaded. Organic mind remains online.";
        return "LOCAL CORTEX: loaded from " + current.model_path;
    }
    if (command == "/load") {
        if (argument.empty()) return "usage: /load <path-to-model.bundle>";
        const std::string path = maybe_unquote(argument);
        std::string error;
        if (!load_model(path, &error)) return "MODEL LOAD FAILED: " + error;
        return "LOCAL CORTEX LOADED: " + path;
    }
    if (command == "/unload") { unload_model(); return "LOCAL CORTEX unloaded. Organic mind remains online."; }
    if (command == "/clear") { clear_history(); return "visible conversation history cleared; organic durable memory preserved."; }
    if (command == "/temperature") {
        const auto value = parse_float(argument);
        if (!value.has_value() || *value < 0.05F || *value > 2.0F) return "usage: /temperature <0.05..2.0>";
        generation_.sampling.temperature = *value;
        std::ostringstream out; out << "local cortex temperature = " << std::fixed << std::setprecision(2) << *value; return out.str();
    }
    if (command == "/max") {
        const auto value = parse_size(argument);
        if (!value.has_value() || *value == 0 || *value > 2048) return "usage: /max <1..2048>";
        generation_.max_new_tokens = *value;
        return "local cortex max new tokens = " + std::to_string(*value);
    }
    if (command == "/history") {
        if (history_.empty()) return "history: empty";
        std::ostringstream out; out << "visible history: " << history_.size() << " turns\n";
        for (const auto& turn : history_) {
            std::string content = turn.content;
            if (content.size() > 120) content = content.substr(0, 117) + "...";
            out << turn.role << "> " << content << '\n';
        }
        return out.str();
    }
    if (command == "/memory") {
        return "ORGANIC MEMORY: " + std::to_string(organic_mind_.memory_count()) + " durable records | visible transcript: " + std::to_string(history_.size()) + " turns.";
    }
    if (command == "/agent") return "AGENT: L7 AgentEngine is linked; host-specific action bindings are the next execution layer.";
    if (command == "/render") return "RENDER: native Spiral Ether AI window + L21 software framebuffer + L22 D3D11 host are linked.";
    if (command == "/exit" || command == "/quit") { should_exit_ = true; return "Spiral state saved. Sleeping."; }
    return "unknown command: " + command + " — use /help";
}

std::string GeniusShell::banner_text() const {
    std::ostringstream out;
    out << "SPIRAL ETHER AI / ORGANIC STATE\n";
    out << "brain: " << gpt_backend_name(gpt_backend_) << " | organic revision: " << organic_mind_.state().revision << '\n';
    out << "The native mind exists without an API key. OpenAI is an explicit optional bridge; a trained Spiral bundle is an optional local cortex.\n";
    out << "type /help for commands | /organic for internal state";
    return out.str();
}

std::string GeniusShell::help_text() const {
    return
        "/gpt                 conversational mode\n"
        "/native              runtime/substrate inspection\n"
        "/backend <name>      organic|spiral|openai\n"
        "/organic             inspect persistent organic state\n"
        "/resetorganic        reset organic state + durable memories\n"
        "/openai              optional OpenAI bridge readiness\n"
        "/gptmodel [id]       show/set optional OpenAI model id\n"
        "/status              full runtime status\n"
        "/gpu                 D3D11 compute status\n"
        "/model               optional local Spiral cortex status\n"
        "/load <bundle>       load native Spiral model bundle\n"
        "/unload              unload local cortex\n"
        "/temperature <n>     local cortex temperature 0.05..2.0\n"
        "/max <n>             local cortex max generated tokens\n"
        "/history             visible conversation turns\n"
        "/memory              organic durable memory status\n"
        "/clear               clear visible transcript only\n"
        "/agent               agent binding status\n"
        "/render              renderer/host status\n"
        "/exit                quit";
}

std::string GeniusShell::status_text() const {
    const auto current = status();
    std::ostringstream out;
    out << "SPIRAL ETHER AI / STATUS\n";
    out << "mode: " << shell_mode_name(current.mode) << '\n';
    out << "brain: " << gpt_backend_name(current.gpt_backend) << " (ORGANIC is native/offline)\n";
    out << "organic: ONLINE | rev " << current.organic_revision << " | turns " << current.organic_turns << " | memories " << current.organic_memories << '\n';
    out << std::fixed << std::setprecision(2)
        << "drives: energy " << current.organic_energy << " focus " << current.organic_focus << " curiosity " << current.organic_curiosity
        << " coherence " << current.organic_coherence << '\n';
    out << "topic: " << (current.organic_topic.empty() ? "none" : current.organic_topic) << '\n';
    out << "optional OpenAI bridge: " << (current.openai_platform_supported && current.openai_key_present ? "available" : "not configured") << '\n';
    out << "optional local cortex: " << (current.model_loaded ? current.model_path : "not loaded") << '\n';
    out << "gpu: " << (current.gpu_available ? "online" : "offline");
    if (current.gpu_available) out << " | " << current.gpu_adapter << " | " << (current.gpu_hardware_accelerated ? "hardware" : "software/WARP");
    out << "\nvisible conversation turns: " << current.conversation_turns;
    return out.str();
}

std::string shell_mode_name(ShellMode mode) {
    switch (mode) {
        case ShellMode::Native: return "NATIVE";
        case ShellMode::Gpt: return "CHAT";
    }
    return "UNKNOWN";
}

std::string gpt_backend_name(GptBackend backend) {
    switch (backend) {
        case GptBackend::Auto: return "ORGANIC";
        case GptBackend::OpenAI: return "OPENAI_BRIDGE";
        case GptBackend::SpiralLocal: return "SPIRAL_LOCAL_CORTEX";
    }
    return "UNKNOWN";
}

} // namespace spiral::genius
