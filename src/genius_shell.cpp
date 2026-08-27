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
    return result;
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
    const auto* begin = value.data();
    const auto* end = value.data() + value.size();
    const auto result = std::from_chars(begin, end, parsed);
    if (result.ec != std::errc{} || result.ptr != end) return std::nullopt;
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
    prompt << "SYSTEM: You are Spiral AI Genius. Respond helpfully, directly, and concisely.\n";
    prompt << "SYSTEM: This is GPT MODE using a local sovereign Spiral model bundle.\n";
    constexpr std::size_t max_turns = 16;
    const std::size_t start = history_.size() > max_turns ? history_.size() - max_turns : 0;
    for (std::size_t i = start; i < history_.size(); ++i) {
        prompt << (history_[i].role == "assistant" ? "ASSISTANT: " : "USER: ")
               << history_[i].content << '\n';
    }
    prompt << "ASSISTANT: ";
    return prompt.str();
}

std::string GeniusShell::build_openai_input() const {
    std::ostringstream input;
    constexpr std::size_t max_turns = 24;
    const std::size_t start = history_.size() > max_turns ? history_.size() - max_turns : 0;
    for (std::size_t i = start; i < history_.size(); ++i) {
        input << (history_[i].role == "assistant" ? "ASSISTANT: " : "USER: ")
              << history_[i].content << '\n';
    }
    input << "ASSISTANT: ";
    return input.str();
}

std::string GeniusShell::local_spiral_reply() {
    if (model_bundle_ == nullptr || model_bundle_->model == nullptr) {
        return "LOCAL SPIRAL backend has no trained language-model bundle loaded. Use /load <path-to-model.bundle>. "
               "I will not fake intelligence with random weights.";
    }
    try {
        generate::ByteTextGenerator generator(*model_bundle_->model);
        auto result = generator.generate(build_chat_prompt(), generation_);
        std::string text = clean_generated_text(std::move(result.text));
        return text.empty() ? "[loaded Spiral model emitted no text]" : text;
    } catch (const std::exception& exception) {
        return std::string("LOCAL SPIRAL generation error: ") + exception.what();
    }
}

std::string GeniusShell::openai_gpt_reply() {
    if (!openai::ResponsesBackend::platform_supported()) {
        return "OPENAI backend is unavailable on this platform at this rung.";
    }
    if (!openai::ResponsesBackend::api_key_present()) {
        return "OPENAI backend selected, but OPENAI_API_KEY is not set. Set it in your Windows environment and restart the shell.";
    }
    const auto response = openai_backend_.respond(
        "You are GPT inside Spiral AI Genius. Respond helpfully, directly, and concisely. Preserve the conversation context provided in the input.",
        build_openai_input(),
        openai_model_);
    if (!response.ok) {
        std::ostringstream out;
        out << "OPENAI GPT error";
        if (response.http_status != 0) out << " (HTTP " << response.http_status << ')';
        out << ": " << response.error;
        return out.str();
    }
    return response.text.empty() ? "[OpenAI response contained no text]" : response.text;
}

std::string GeniusShell::gpt_reply() {
    switch (gpt_backend_) {
        case GptBackend::OpenAI:
            return openai_gpt_reply();
        case GptBackend::SpiralLocal:
            return local_spiral_reply();
        case GptBackend::Auto:
            if (openai::ResponsesBackend::platform_supported() && openai::ResponsesBackend::api_key_present()) {
                const std::string response = openai_gpt_reply();
                if (response.rfind("OPENAI GPT error", 0) != 0) return response;
                if (model_bundle_ != nullptr && model_bundle_->model != nullptr) return local_spiral_reply();
                return response;
            }
            if (model_bundle_ != nullptr && model_bundle_->model != nullptr) return local_spiral_reply();
            return "GPT MODE has no active backend. For real GPT, set OPENAI_API_KEY and use /backend openai. "
                   "For sovereign local generation, /load a trained Spiral model and use /backend spiral.";
    }
    return "GPT MODE backend routing failed.";
}

std::string GeniusShell::native_reply(std::string_view user_message) {
    ByteTokenizer tokenizer;
    const auto tokens = tokenizer.encode(user_message);
    const auto current = status();
    std::ostringstream out;
    out << "NATIVE MODE / substrate inspection\n";
    out << "input bytes: " << user_message.size() << " | byte tokens: " << tokens.size() << '\n';
    out << "gpu compute: " << (current.gpu_available ? "online" : "offline");
    if (current.gpu_available) {
        out << " | " << (current.gpu_hardware_accelerated ? "hardware" : "WARP")
            << " | " << current.gpu_adapter;
    }
    out << "\nlocal model bundle: " << (current.model_loaded ? current.model_path : "none")
        << "\nOpenAI key: " << (current.openai_key_present ? "present" : "not set")
        << "\nUse /gpt for conversational mode.";
    return out.str();
}

std::string GeniusShell::run_command(std::string_view line) {
    const std::size_t split = line.find_first_of(" \t");
    const std::string command = std::string(line.substr(0, split));
    const std::string argument = split == std::string_view::npos ? std::string{} : trim(line.substr(split + 1));

    if (command == "/help" || command == "/?") return help_text();
    if (command == "/status") return status_text();
    if (command == "/gpt") {
        mode_ = ShellMode::Gpt;
        return "GPT MODE enabled. Backend = " + gpt_backend_name(gpt_backend_) + ".";
    }
    if (command == "/native") {
        mode_ = ShellMode::Native;
        return "NATIVE MODE enabled — deterministic runtime/substrate inspection.";
    }
    if (command == "/mode") {
        if (argument == "gpt") { mode_ = ShellMode::Gpt; return "mode = GPT"; }
        if (argument == "native") { mode_ = ShellMode::Native; return "mode = NATIVE"; }
        return "usage: /mode gpt|native";
    }
    if (command == "/backend") {
        if (argument.empty()) return "GPT backend = " + gpt_backend_name(gpt_backend_);
        if (argument == "auto") { gpt_backend_ = GptBackend::Auto; return "GPT backend = AUTO"; }
        if (argument == "openai" || argument == "gpt") { gpt_backend_ = GptBackend::OpenAI; return "GPT backend = OPENAI"; }
        if (argument == "spiral" || argument == "local") { gpt_backend_ = GptBackend::SpiralLocal; return "GPT backend = SPIRAL_LOCAL"; }
        return "usage: /backend auto|openai|spiral";
    }
    if (command == "/openai") {
        const auto current = status();
        std::ostringstream out;
        out << "OPENAI GPT: " << (current.openai_key_present && current.openai_platform_supported ? "READY" : "NOT READY") << '\n';
        out << "platform: " << (current.openai_platform_supported ? "supported" : "unsupported") << '\n';
        out << "OPENAI_API_KEY: " << (current.openai_key_present ? "present" : "not set") << '\n';
        out << "model: " << current.openai_model;
        return out.str();
    }
    if (command == "/gptmodel") {
        if (argument.empty()) return "OpenAI model = " + openai_model_;
        openai_model_ = argument;
        return "OpenAI model = " + openai_model_;
    }
    if (command == "/gpu") {
        const auto current = status();
        std::ostringstream out;
        out << "GPU: " << (current.gpu_available ? "ONLINE" : "OFFLINE") << '\n';
        out << "platform: " << (current.gpu_platform_supported ? "D3D11 supported" : "D3D11 unavailable") << '\n';
        if (current.gpu_available) {
            out << "adapter: " << current.gpu_adapter << '\n';
            out << "feature level: " << current.gpu_feature_level << '\n';
            out << "execution: " << (current.gpu_hardware_accelerated ? "hardware" : "WARP compatibility");
        } else if (!gpu_error_.empty()) {
            out << "reason: " << gpu_error_;
        }
        return out.str();
    }
    if (command == "/model") {
        const auto current = status();
        if (!current.model_loaded) return "LOCAL MODEL: none loaded. Use /load <bundle-path>.";
        return "LOCAL MODEL: loaded from " + current.model_path;
    }
    if (command == "/load") {
        if (argument.empty()) return "usage: /load <path-to-model.bundle>";
        const std::string path = maybe_unquote(argument);
        std::string error;
        if (!load_model(path, &error)) return "MODEL LOAD FAILED: " + error;
        return "LOCAL MODEL LOADED: " + path;
    }
    if (command == "/unload") { unload_model(); return "LOCAL MODEL unloaded."; }
    if (command == "/clear") { clear_history(); return "conversation history cleared."; }
    if (command == "/temperature") {
        const auto value = parse_float(argument);
        if (!value.has_value() || *value < 0.05F || *value > 2.0F) return "usage: /temperature <0.05..2.0>";
        generation_.sampling.temperature = *value;
        std::ostringstream out; out << "local temperature = " << std::fixed << std::setprecision(2) << *value;
        return out.str();
    }
    if (command == "/max") {
        const auto value = parse_size(argument);
        if (!value.has_value() || *value == 0 || *value > 2048) return "usage: /max <1..2048>";
        generation_.max_new_tokens = *value;
        return "local max new tokens = " + std::to_string(*value);
    }
    if (command == "/history") {
        if (history_.empty()) return "history: empty";
        std::ostringstream out;
        out << "history: " << history_.size() << " turns\n";
        for (const auto& turn : history_) {
            std::string content = turn.content;
            if (content.size() > 120) content = content.substr(0, 117) + "...";
            out << turn.role << "> " << content << '\n';
        }
        return out.str();
    }
    if (command == "/memory") {
        return "MEMORY: session transcript has " + std::to_string(history_.size()) +
               " turns. L6 persistent MemoryStore exists in the library; shell persistence is not bound yet.";
    }
    if (command == "/agent") {
        return "AGENT: L7 AgentEngine is compiled into Spiral. Autonomous planner binding is not enabled in this console yet.";
    }
    if (command == "/render") {
        return "RENDER: L21 software framebuffer + L22 Win32/D3D11 host are compiled. Genius currently uses the console surface.";
    }
    if (command == "/exit" || command == "/quit") { should_exit_ = true; return "Spiral sleeping."; }
    return "unknown command: " + command + " — use /help";
}

std::string GeniusShell::banner_text() const {
    std::ostringstream out;
    out << "SPIRAL AI GENIUS / L23 SHELL\n";
    out << "mode: GPT | backend: " << gpt_backend_name(gpt_backend_) << "\n";
    out << "GPT can route to OpenAI Responses API when OPENAI_API_KEY is set, or to a trained local Spiral bundle.\n";
    out << "type /help for commands | /native for sovereign substrate mode";
    return out.str();
}

std::string GeniusShell::help_text() const {
    return
        "/gpt                 conversational GPT mode\n"
        "/native              deterministic native/runtime mode\n"
        "/backend <name>      auto|openai|spiral\n"
        "/openai              OpenAI API readiness (never prints the key)\n"
        "/gptmodel [id]       show/set OpenAI model id\n"
        "/status              full runtime status\n"
        "/gpu                 D3D11 compute status\n"
        "/model               local Spiral model status\n"
        "/load <bundle>       load native Spiral model bundle\n"
        "/unload              unload local model\n"
        "/temperature <n>     local sampling temperature 0.05..2.0\n"
        "/max <n>             local max generated tokens 1..2048\n"
        "/history             show conversation turns\n"
        "/clear               clear conversation history\n"
        "/memory              memory binding status\n"
        "/agent               agent binding status\n"
        "/render              renderer/host status\n"
        "/exit                quit";
}

std::string GeniusShell::status_text() const {
    const auto current = status();
    std::ostringstream out;
    out << "SPIRAL AI GENIUS / STATUS\n";
    out << "mode: " << shell_mode_name(current.mode) << '\n';
    out << "GPT backend: " << gpt_backend_name(current.gpt_backend) << '\n';
    out << "OpenAI GPT: " << (current.openai_platform_supported && current.openai_key_present ? "ready" : "not ready")
        << " | model " << current.openai_model << '\n';
    out << "local Spiral model: " << (current.model_loaded ? current.model_path : "NOT LOADED") << '\n';
    out << "gpu: " << (current.gpu_available ? "online" : "offline");
    if (current.gpu_available) {
        out << " (" << (current.gpu_hardware_accelerated ? "hardware" : "WARP") << ")\n";
        out << "adapter: " << current.gpu_adapter << '\n';
        out << "feature: " << current.gpu_feature_level << '\n';
    } else {
        out << '\n';
    }
    out << "conversation turns: " << current.conversation_turns << '\n';
    out << "local temperature: " << std::fixed << std::setprecision(2) << current.temperature << '\n';
    out << "local max new tokens: " << current.max_new_tokens << '\n';
    out << "agent substrate: linked\nmedia stack: linked\nSpiral Units/runtime: linked";
    return out.str();
}

std::string shell_mode_name(ShellMode mode) {
    switch (mode) {
        case ShellMode::Native: return "NATIVE";
        case ShellMode::Gpt: return "GPT";
    }
    return "UNKNOWN";
}

std::string gpt_backend_name(GptBackend backend) {
    switch (backend) {
        case GptBackend::Auto: return "AUTO";
        case GptBackend::OpenAI: return "OPENAI";
        case GptBackend::SpiralLocal: return "SPIRAL_LOCAL";
    }
    return "UNKNOWN";
}

} // namespace spiral::genius
