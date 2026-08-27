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

GeniusShell::GeniusShell() {
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
    result.gpu_platform_supported = gpu::D3D11GpuDevice::platform_supported();
    result.gpu_available = gpu_device_ != nullptr && gpu_compute_ != nullptr && gpu_compute_->available();
    if (gpu_device_ != nullptr) {
        const auto capabilities = gpu_device_->capabilities();
        result.gpu_hardware_accelerated = capabilities.hardware_accelerated;
        result.gpu_adapter = capabilities.adapter_name;
        result.gpu_feature_level = capabilities.feature_level;
    }
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
    prompt << "SYSTEM: This is GPT MODE, a chat protocol running on a local Spiral model bundle.\n";

    constexpr std::size_t max_turns = 16;
    const std::size_t start = history_.size() > max_turns ? history_.size() - max_turns : 0;
    for (std::size_t i = start; i < history_.size(); ++i) {
        prompt << (history_[i].role == "assistant" ? "ASSISTANT: " : "USER: ")
               << history_[i].content << '\n';
    }
    prompt << "ASSISTANT: ";
    return prompt.str();
}

std::string GeniusShell::gpt_reply() {
    if (model_bundle_ == nullptr || model_bundle_->model == nullptr) {
        return "GPT MODE is online, but no trained Spiral language-model bundle is loaded. "
               "Use /load <path-to-model.bundle>. I will not fake intelligence with random weights.";
    }

    try {
        generate::ByteTextGenerator generator(*model_bundle_->model);
        auto result = generator.generate(build_chat_prompt(), generation_);
        std::string text = clean_generated_text(std::move(result.text));
        if (text.empty()) return "[loaded model emitted no text]";
        return text;
    } catch (const std::exception& exception) {
        return std::string("GPT MODE generation error: ") + exception.what();
    }
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
    out << "\nmodel bundle: " << (current.model_loaded ? current.model_path : "none")
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
        return "GPT MODE enabled — ChatGPT-style conversation protocol over the local Spiral model backend.";
    }
    if (command == "/native") {
        mode_ = ShellMode::Native;
        return "NATIVE MODE enabled — deterministic runtime/substrate inspection.";
    }
    if (command == "/mode") {
        if (argument == "gpt") {
            mode_ = ShellMode::Gpt;
            return "mode = GPT";
        }
        if (argument == "native") {
            mode_ = ShellMode::Native;
            return "mode = NATIVE";
        }
        return "usage: /mode gpt|native";
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
        if (!current.model_loaded) return "MODEL: none loaded. Use /load <bundle-path>.";
        return "MODEL: loaded from " + current.model_path;
    }
    if (command == "/load") {
        if (argument.empty()) return "usage: /load <path-to-model.bundle>";
        const std::string path = maybe_unquote(argument);
        std::string error;
        if (!load_model(path, &error)) return "MODEL LOAD FAILED: " + error;
        return "MODEL LOADED: " + path;
    }
    if (command == "/unload") {
        unload_model();
        return "MODEL unloaded.";
    }
    if (command == "/clear") {
        clear_history();
        return "conversation history cleared.";
    }
    if (command == "/temperature") {
        const auto value = parse_float(argument);
        if (!value.has_value() || *value < 0.05F || *value > 2.0F) {
            return "usage: /temperature <0.05..2.0>";
        }
        generation_.sampling.temperature = *value;
        std::ostringstream out;
        out << "temperature = " << std::fixed << std::setprecision(2) << *value;
        return out.str();
    }
    if (command == "/max") {
        const auto value = parse_size(argument);
        if (!value.has_value() || *value == 0 || *value > 2048) return "usage: /max <1..2048>";
        generation_.max_new_tokens = *value;
        return "max new tokens = " + std::to_string(*value);
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
        return "AGENT: L7 AgentEngine is compiled into Spiral. This shell currently exposes chat/model control; autonomous planner binding is not enabled yet.";
    }
    if (command == "/render") {
        return "RENDER: L21 software framebuffer + L22 Win32/D3D11 host are compiled. This Genius shell currently uses the console surface.";
    }
    if (command == "/exit" || command == "/quit") {
        should_exit_ = true;
        return "Spiral sleeping.";
    }
    return "unknown command: " + command + " — use /help";
}

std::string GeniusShell::banner_text() const {
    std::ostringstream out;
    out << "SPIRAL AI GENIUS / L23 SHELL\n";
    out << "mode: GPT | local sovereign runtime\n";
    out << "type /help for commands | /native for substrate mode\n";
    out << "GPT MODE is a chat UX/protocol. It does not claim OpenAI-hosted GPT unless a future external backend is explicitly configured.";
    return out.str();
}

std::string GeniusShell::help_text() const {
    return
        "/gpt                 GPT-style conversation mode\n"
        "/native              deterministic native/runtime mode\n"
        "/mode gpt|native     switch mode\n"
        "/status              full runtime status\n"
        "/gpu                 D3D11 compute status\n"
        "/model               loaded model status\n"
        "/load <bundle>       load native Spiral model bundle\n"
        "/unload              unload model\n"
        "/temperature <n>     sampling temperature 0.05..2.0\n"
        "/max <n>             max generated tokens 1..2048\n"
        "/history             show current conversation turns\n"
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
    out << "gpu: " << (current.gpu_available ? "online" : "offline");
    if (current.gpu_available) {
        out << " (" << (current.gpu_hardware_accelerated ? "hardware" : "WARP") << ")\n";
        out << "adapter: " << current.gpu_adapter << '\n';
        out << "feature: " << current.gpu_feature_level << '\n';
    } else {
        out << '\n';
    }
    out << "language model: " << (current.model_loaded ? current.model_path : "NOT LOADED") << '\n';
    out << "conversation turns: " << current.conversation_turns << '\n';
    out << "temperature: " << std::fixed << std::setprecision(2) << current.temperature << '\n';
    out << "max new tokens: " << current.max_new_tokens << '\n';
    out << "agent substrate: linked\n";
    out << "media stack: linked\n";
    out << "Spiral Units/runtime: linked";
    return out.str();
}

std::string shell_mode_name(ShellMode mode) {
    switch (mode) {
        case ShellMode::Native: return "NATIVE";
        case ShellMode::Gpt: return "GPT";
    }
    return "UNKNOWN";
}

} // namespace spiral::genius
