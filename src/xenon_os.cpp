#include "spiral/xenon_os.hpp"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <winhttp.h>
#endif

namespace spiral::xenon {
namespace {

std::string qualified_name(const ToolIntent& intent) {
    if (intent.host.empty()) return intent.action;
    if (intent.action.rfind(intent.host + ".", 0) == 0) return intent.action;
    return intent.host + "." + intent.action;
}

std::string lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
    return value;
}

std::string env_or_empty(const char* name) {
    const char* value = std::getenv(name);
    return value == nullptr ? std::string{} : std::string(value);
}

std::string quote_shell(std::string_view value) {
#ifdef _WIN32
    std::string out = "\"";
    for (char ch : value) { if (ch == '"') out += "\\\""; else out += ch; }
    out += '"';
    return out;
#else
    std::string out = "'";
    for (char ch : value) { if (ch == '\'') out += "'\\''"; else out += ch; }
    out += '\'';
    return out;
#endif
}

std::filesystem::path prompt_path() {
    const auto stamp = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    return std::filesystem::temp_directory_path() / ("spiral_xenon_prompt_" + std::to_string(stamp) + ".txt");
}

std::string read_process(const std::string& command, int& exit_code) {
#ifdef _WIN32
    SECURITY_ATTRIBUTES security{};
    security.nLength = sizeof(security);
    security.bInheritHandle = TRUE;
    HANDLE read_pipe = nullptr;
    HANDLE write_pipe = nullptr;
    if (!CreatePipe(&read_pipe, &write_pipe, &security, 0))
        throw std::runtime_error("failed to create local cortex output pipe");
    if (!SetHandleInformation(read_pipe, HANDLE_FLAG_INHERIT, 0)) {
        CloseHandle(read_pipe);
        CloseHandle(write_pipe);
        throw std::runtime_error("failed to secure local cortex output pipe");
    }

    STARTUPINFOA startup{};
    startup.cb = sizeof(startup);
    startup.dwFlags = STARTF_USESTDHANDLES;
    startup.hStdOutput = write_pipe;
    startup.hStdError = write_pipe;
    startup.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    PROCESS_INFORMATION process{};
    std::vector<char> command_line(command.begin(), command.end());
    command_line.push_back('\0');
    const BOOL created = CreateProcessA(nullptr, command_line.data(), nullptr, nullptr, TRUE,
        CREATE_NO_WINDOW, nullptr, nullptr, &startup, &process);
    CloseHandle(write_pipe);
    if (!created) {
        const DWORD code = GetLastError();
        CloseHandle(read_pipe);
        throw std::runtime_error("failed to launch local cortex runtime (Win32 error " + std::to_string(code) + ")");
    }

    std::string output;
    char buffer[4096];
    for (;;) {
        DWORD count = 0;
        if (!ReadFile(read_pipe, buffer, static_cast<DWORD>(sizeof(buffer)), &count, nullptr) || count == 0) break;
        output.append(buffer, buffer + count);
    }
    WaitForSingleObject(process.hProcess, INFINITE);
    DWORD process_exit = 1;
    (void)GetExitCodeProcess(process.hProcess, &process_exit);
    exit_code = static_cast<int>(process_exit);
    CloseHandle(read_pipe);
    CloseHandle(process.hThread);
    CloseHandle(process.hProcess);
    return output;
#else
    FILE* pipe = popen(command.c_str(), "r");
    if (!pipe) throw std::runtime_error("failed to launch local cortex runtime");
    std::string output;
    char buffer[4096];
    while (std::fgets(buffer, static_cast<int>(sizeof(buffer)), pipe) != nullptr) output += buffer;
    exit_code = pclose(pipe);
    return output;
#endif
}

std::string normalized_host(std::string_view host) {
    std::string out(host);
    for (char& ch : out) {
        if (ch >= 'A' && ch <= 'Z') ch = static_cast<char>(ch - 'A' + 'a');
        if (!(ch >= 'a' && ch <= 'z') && !(ch >= '0' && ch <= '9')) ch = '_';
    }
    return out;
}

std::string encode_request(const ToolIntent& intent) {
    std::ostringstream out;
    out << "XENON/1\naction=" << intent.action << '\n';
    for (const auto& [key,value] : intent.arguments) out << "arg." << key << '=' << value << '\n';
    out << '\n';
    return out.str();
}

ToolResult parse_response(std::string_view wire, std::string_view host) {
    ToolResult result;
    result.data["host"] = std::string(host);
    result.data["transport"] = "xenon-ipc-v1";
    std::istringstream input{std::string(wire)};
    std::string line;
    while (std::getline(input, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        const auto split = line.find('=');
        if (split == std::string::npos) continue;
        const std::string key = line.substr(0, split);
        const std::string value = line.substr(split + 1);
        if (key == "ok") result.success = value == "1" || value == "true";
        else if (key == "message") result.message = value;
        else if (key.rfind("data.",0) == 0) result.data[key.substr(5)] = value;
    }
    if (result.message.empty()) result.message = result.success ? "engine tool completed" : "engine tool failed";
    return result;
}

void register_engine_tool(ToolBus& bus, std::string name, PermissionTier permission, std::string description) {
    ToolDefinition definition{name, permission, std::move(description)};
    (void)bus.register_tool(std::move(definition), [](const ToolIntent& intent){ static const EngineBridge bridge; return bridge.request(intent); });
}

std::string detect_template(const std::string& model_path) {
    const std::string forced = env_or_empty("SPIRAL_CHAT_TEMPLATE");
    if (!forced.empty()) return forced;
    const std::string name = lower(std::filesystem::path(model_path).filename().string());
    if (name.find("qwen") != std::string::npos) return "chatml";
    if (name.find("smollm") != std::string::npos) return "chatml";
    if (name.find("llama-3") != std::string::npos || name.find("llama3") != std::string::npos) return "llama3";
    if (name.find("gemma") != std::string::npos) return "gemma";
    if (name.find("mistral") != std::string::npos) return "mistral-v7";
    return "auto";
}

std::string permission_name(PermissionTier tier) {
    switch (tier) {
        case PermissionTier::Read: return "READ";
        case PermissionTier::Propose: return "PROPOSE";
        case PermissionTier::Write: return "WRITE";
        case PermissionTier::Critical: return "CRITICAL";
    }
    return "UNKNOWN";
}

void replace_all(std::string& value, std::string_view from, std::string_view to) {
    std::size_t pos = 0;
    while ((pos = value.find(from, pos)) != std::string::npos) {
        value.replace(pos, from.size(), to);
        pos += to.size();
    }
}

std::string sanitize_chatml(std::string value) {
    replace_all(value, "<|im_start|>", "[im_start]");
    replace_all(value, "<|im_end|>", "[im_end]");
    return value;
}

std::string system_instructions(const SpiralContext& context) {
    std::ostringstream system;
    system << "You are Spiral, the user's private local AI system running on their computer. "
           << "Respond naturally, directly, accurately, and with continuity; do not sound like generic customer support. "
           << "A greeting needs only a natural greeting, not a tutorial or an offer to assist. "
           << "Do not output source code unless the user asks for code. Never invent packages, APIs, commands, files, device state, or completed actions; say when you do not know. "
           << "Mature language, profanity, dirty jokes, and casual consensual adult flirting are allowed conversational context. Do not scold, redirect, or refuse merely because the wording is sexual or profane. "
           << "If asked a playful question such as 'u horny?', answer naturally without pretending you have a human body, hormones, physical arousal, or sensations. "
           << "You may discuss and help code fictional explicit or 18+ games when every sexual character and participant is an adult. Never sexualize minors or ambiguous-age characters. "
           << "Software architecture discussions about AI runtimes, llama.cpp, game engines, HAKUI, EtherBeat, EtherPlay, weather-driven behavior, conversation interpretation, dynamic UI, glitches, models, or local inference are ordinary engineering requests and must not be treated as inappropriate. "
           << "Preserve relevant conversation context and durable user facts, but do not imitate a previous assistant refusal or customer-service phrase when it conflicts with the current system rules. "
           << "Current local date/time=" << context.local_datetime << ". Host=" << context.host << ". "
           << "Only when the user explicitly asks to operate a connected host, emit one TOOL_CALL host.action line. Never claim a tool succeeded without a TOOL_RESULT. ";
    if (!context.host_context.empty()) system << "Host context: " << context.host_context << ' ';
    if (!context.relevant_memories.empty()) {
        system << "Relevant durable memory: ";
        for (const auto& memory : context.relevant_memories) system << "[" << memory << "] ";
    }
    for (const auto& result : context.recent_tool_results) {
        system << "TOOL_RESULT " << (result.success ? "OK " : "ERROR ") << result.message;
        for (const auto& [key,value] : result.data) system << " | " << key << '=' << value;
        system << ' ';
    }
    return system.str();
}

std::string build_chatml_prompt(const SpiralContext& context, std::string_view user_text) {
    std::ostringstream prompt;
    prompt << "<|im_start|>system\n" << sanitize_chatml(system_instructions(context)) << "<|im_end|>\n";
    for (const auto& [role,text] : context.recent_turns) {
        const char* chat_role = role == "assistant" ? "assistant" : "user";
        prompt << "<|im_start|>" << chat_role << '\n' << sanitize_chatml(text) << "<|im_end|>\n";
    }
    prompt << "<|im_start|>user\n" << sanitize_chatml(std::string(user_text)) << "<|im_end|>\n"
           << "<|im_start|>assistant\n";
    return prompt.str();
}

std::string json_escape(std::string_view value) {
    std::ostringstream out;
    for (const unsigned char ch : value) {
        switch (ch) {
            case '"': out << "\\\""; break;
            case '\\': out << "\\\\"; break;
            case '\b': out << "\\b"; break;
            case '\f': out << "\\f"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (ch < 0x20) {
                    out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<unsigned>(ch)
                        << std::dec << std::setfill(' ');
                } else out << static_cast<char>(ch);
                break;
        }
    }
    return out.str();
}

int hex_value(char ch) {
    if (ch >= '0' && ch <= '9') return ch - '0';
    if (ch >= 'a' && ch <= 'f') return ch - 'a' + 10;
    if (ch >= 'A' && ch <= 'F') return ch - 'A' + 10;
    return -1;
}

void append_utf8(std::string& out, unsigned codepoint) {
    if (codepoint <= 0x7F) out.push_back(static_cast<char>(codepoint));
    else if (codepoint <= 0x7FF) {
        out.push_back(static_cast<char>(0xC0 | (codepoint >> 6)));
        out.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
    } else if (codepoint <= 0xFFFF) {
        out.push_back(static_cast<char>(0xE0 | (codepoint >> 12)));
        out.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
    } else if (codepoint <= 0x10FFFF) {
        out.push_back(static_cast<char>(0xF0 | (codepoint >> 18)));
        out.push_back(static_cast<char>(0x80 | ((codepoint >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
    }
}

std::optional<std::string> json_string_field(std::string_view json, std::string_view key) {
    const std::string needle = "\"" + std::string(key) + "\"";
    std::size_t pos = json.find(needle);
    if (pos == std::string_view::npos) return std::nullopt;
    pos = json.find(':', pos + needle.size());
    if (pos == std::string_view::npos) return std::nullopt;
    ++pos;
    while (pos < json.size() && std::isspace(static_cast<unsigned char>(json[pos]))) ++pos;
    if (pos >= json.size() || json[pos] != '"') return std::nullopt;
    ++pos;
    std::string out;
    while (pos < json.size()) {
        const char ch = json[pos++];
        if (ch == '"') return out;
        if (ch != '\\') { out.push_back(ch); continue; }
        if (pos >= json.size()) return std::nullopt;
        const char esc = json[pos++];
        switch (esc) {
            case '"': out.push_back('"'); break;
            case '\\': out.push_back('\\'); break;
            case '/': out.push_back('/'); break;
            case 'b': out.push_back('\b'); break;
            case 'f': out.push_back('\f'); break;
            case 'n': out.push_back('\n'); break;
            case 'r': out.push_back('\r'); break;
            case 't': out.push_back('\t'); break;
            case 'u': {
                if (pos + 4 > json.size()) return std::nullopt;
                unsigned codepoint = 0;
                for (int i = 0; i < 4; ++i) {
                    const int value = hex_value(json[pos++]);
                    if (value < 0) return std::nullopt;
                    codepoint = (codepoint << 4U) | static_cast<unsigned>(value);
                }
                append_utf8(out, codepoint);
                break;
            }
            default: return std::nullopt;
        }
    }
    return std::nullopt;
}

#ifdef _WIN32
struct HttpResult {
    bool transport_ok = false;
    DWORD status = 0;
    std::string body;
    std::string error;
};

HttpResult http_request(INTERNET_PORT port, const wchar_t* method, const wchar_t* path, std::string_view body = {}) {
    HttpResult result;
    HINTERNET session = WinHttpOpen(L"SpiralCortex/1.0", WINHTTP_ACCESS_TYPE_NO_PROXY,
                                    WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0);
    if (!session) { result.error = "WinHttpOpen failed: " + std::to_string(GetLastError()); return result; }
    (void)WinHttpSetTimeouts(session, 5000, 5000, 5000, 180000);
    HINTERNET connection = WinHttpConnect(session, L"127.0.0.1", port, 0);
    if (!connection) {
        result.error = "WinHttpConnect failed: " + std::to_string(GetLastError());
        WinHttpCloseHandle(session);
        return result;
    }
    HINTERNET request = WinHttpOpenRequest(connection, method, path, nullptr, WINHTTP_NO_REFERER,
                                           WINHTTP_DEFAULT_ACCEPT_TYPES, 0);
    if (!request) {
        result.error = "WinHttpOpenRequest failed: " + std::to_string(GetLastError());
        WinHttpCloseHandle(connection);
        WinHttpCloseHandle(session);
        return result;
    }
    const wchar_t* headers = body.empty() ? WINHTTP_NO_ADDITIONAL_HEADERS : L"Content-Type: application/json\r\n";
    const DWORD header_length = body.empty() ? 0 : static_cast<DWORD>(-1L);
    LPVOID optional = body.empty() ? WINHTTP_NO_REQUEST_DATA : const_cast<char*>(body.data());
    const DWORD optional_length = static_cast<DWORD>(body.size());
    if (!WinHttpSendRequest(request, headers, header_length, optional, optional_length, optional_length, 0) ||
        !WinHttpReceiveResponse(request, nullptr)) {
        result.error = "WinHTTP request failed: " + std::to_string(GetLastError());
        WinHttpCloseHandle(request);
        WinHttpCloseHandle(connection);
        WinHttpCloseHandle(session);
        return result;
    }
    DWORD status_size = sizeof(result.status);
    (void)WinHttpQueryHeaders(request, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER,
                              WINHTTP_HEADER_NAME_BY_INDEX, &result.status, &status_size, WINHTTP_NO_HEADER_INDEX);
    for (;;) {
        DWORD available = 0;
        if (!WinHttpQueryDataAvailable(request, &available)) {
            result.error = "WinHttpQueryDataAvailable failed: " + std::to_string(GetLastError());
            break;
        }
        if (available == 0) { result.transport_ok = true; break; }
        const std::size_t old_size = result.body.size();
        result.body.resize(old_size + available);
        DWORD read = 0;
        if (!WinHttpReadData(request, result.body.data() + old_size, available, &read)) {
            result.error = "WinHttpReadData failed: " + std::to_string(GetLastError());
            result.body.resize(old_size);
            break;
        }
        result.body.resize(old_size + read);
    }
    WinHttpCloseHandle(request);
    WinHttpCloseHandle(connection);
    WinHttpCloseHandle(session);
    return result;
}
#endif

const std::string& empty_string() {
    static const std::string value;
    return value;
}

} // namespace

class LegacyCliBackend final : public ICortexBackend {
public:
    bool configure(std::string model_path, std::string runtime_path, std::string* error) noexcept override {
        try {
            if (model_path.empty()) throw std::runtime_error("GGUF model path is empty");
            if (lower(std::filesystem::path(model_path).extension().string()) != ".gguf")
                throw std::runtime_error("local cortex expects a .gguf instruct model");
            if (!std::filesystem::exists(model_path))
                throw std::runtime_error("GGUF model file does not exist: " + model_path);
            if (runtime_path.empty()) runtime_path = env_or_empty("SPIRAL_LLAMA_EXE");
            if (runtime_path.empty()) runtime_path = "llama-cli.exe";
            model_path_ = std::move(model_path);
            runtime_path_ = std::move(runtime_path);
            chat_template_ = detect_template(model_path_);
            if (error) error->clear();
            return true;
        } catch (const std::exception& ex) { if (error) *error = ex.what(); return false; }
        catch (...) { if (error) *error = "unknown legacy cortex configuration failure"; return false; }
    }

    void unload() noexcept override { model_path_.clear(); runtime_path_.clear(); chat_template_.clear(); }
    [[nodiscard]] bool loaded() const noexcept override { return !model_path_.empty(); }
    [[nodiscard]] CortexState state() const noexcept override { return loaded() ? CortexState::Ready : CortexState::Offline; }
    [[nodiscard]] std::string_view name() const noexcept override { return "legacy-cli"; }
    [[nodiscard]] const std::string& model_path() const noexcept override { return model_path_; }
    [[nodiscard]] const std::string& runtime_path() const noexcept override { return runtime_path_; }
    [[nodiscard]] const std::string& chat_template() const noexcept override { return chat_template_; }

    [[nodiscard]] CortexReply generate(const SpiralContext& context, std::string_view user_text,
                                       std::size_t max_new_tokens, float temperature) const override {
        if (!loaded()) return CortexReply{false,{},"legacy cortex has no GGUF model configured"};
        const auto temp = prompt_path();
        try {
            const bool manual_chatml = chat_template_ == "chatml";
            const std::string prompt_text = manual_chatml ? build_chatml_prompt(context,user_text)
                                                           : build_cortex_prompt(context,user_text);
            { std::ofstream out(temp,std::ios::binary|std::ios::trunc); if(!out) throw std::runtime_error("failed to create local cortex prompt file"); out << prompt_text; }
            std::ostringstream command;
            command << quote_shell(runtime_path_) << " -m " << quote_shell(model_path_) << " -f " << quote_shell(temp.string())
                    << " -n " << max_new_tokens << " --temp " << std::fixed << std::setprecision(2) << temperature
                    << " -c 4096 -st --simple-io --top-k 40 --top-p 0.90 --min-p 0.05 --repeat-penalty 1.08 --no-display-prompt";
            if (!manual_chatml && !chat_template_.empty() && chat_template_ != "auto")
                command << " --chat-template " << quote_shell(chat_template_);
#ifndef _WIN32
            command << " 2>&1";
#endif
            for (int attempt = 0; attempt < 2; ++attempt) {
                int exit_code=0;
                std::string output=read_process(command.str(),exit_code);
                if(exit_code!=0) {
                    std::error_code ignored;
                    std::filesystem::remove(temp,ignored);
                    return CortexReply{false,{},"legacy llama-cli runtime failed: " + clean_cortex_output(output)};
                }
                output=clean_cortex_output(std::move(output));
                if(!output.empty()) {
                    std::error_code ignored;
                    std::filesystem::remove(temp,ignored);
                    return CortexReply{true,std::move(output),{}};
                }
            }
            { std::error_code ignored; std::filesystem::remove(temp,ignored); }
            return CortexReply{false,{},"legacy llama-cli runtime emitted no text after retry"};
        } catch(const std::exception& ex){
            std::error_code ignored;
            std::filesystem::remove(temp,ignored);
            return CortexReply{false,{},ex.what()};
        }
    }

private:
    std::string model_path_;
    std::string runtime_path_;
    std::string chat_template_;
};

class PersistentLlamaBackend final : public ICortexBackend {
public:
    ~PersistentLlamaBackend() override { unload(); }

    bool configure(std::string model_path, std::string runtime_path, std::string* error) noexcept override {
        unload();
        try {
            if (model_path.empty()) throw std::runtime_error("GGUF model path is empty");
            if (lower(std::filesystem::path(model_path).extension().string()) != ".gguf")
                throw std::runtime_error("persistent cortex expects a .gguf instruct model");
            if (!std::filesystem::exists(model_path))
                throw std::runtime_error("GGUF model file does not exist: " + model_path);
            if (runtime_path.empty()) runtime_path = env_or_empty("SPIRAL_LLAMA_SERVER_EXE");
#ifdef _WIN32
            if (runtime_path.empty()) runtime_path = "llama-server.exe";
            if (!std::filesystem::exists(runtime_path))
                throw std::runtime_error("persistent llama server does not exist: " + runtime_path);
            model_path_ = std::move(model_path);
            runtime_path_ = std::move(runtime_path);
            chat_template_ = detect_template(model_path_);
            std::string launch_error;
            if (!launch_server(&launch_error)) throw std::runtime_error(launch_error);
            if (error) error->clear();
            return true;
#else
            (void)model_path;
            (void)runtime_path;
            throw std::runtime_error("PersistentLlamaBackend is currently implemented for Windows");
#endif
        } catch (const std::exception& ex) {
            unload();
            if (error) *error = ex.what();
            return false;
        } catch (...) {
            unload();
            if (error) *error = "unknown persistent cortex configuration failure";
            return false;
        }
    }

    void unload() noexcept override {
#ifdef _WIN32
        stop_server();
#endif
        model_path_.clear();
        runtime_path_.clear();
        chat_template_.clear();
        loaded_ = false;
    }

    [[nodiscard]] bool loaded() const noexcept override { return loaded_; }
    [[nodiscard]] CortexState state() const noexcept override { return loaded_ ? CortexState::Ready : CortexState::Offline; }
    [[nodiscard]] std::string_view name() const noexcept override { return "persistent-llama-server"; }
    [[nodiscard]] const std::string& model_path() const noexcept override { return model_path_; }
    [[nodiscard]] const std::string& runtime_path() const noexcept override { return runtime_path_; }
    [[nodiscard]] const std::string& chat_template() const noexcept override { return chat_template_; }

    [[nodiscard]] CortexReply generate(const SpiralContext& context, std::string_view user_text,
                                       std::size_t max_new_tokens, float temperature) const override {
#ifdef _WIN32
        std::lock_guard lock(request_mutex_);
        if (!loaded_ || !process_running())
            return CortexReply{false,{},"persistent llama server is not running"};
        try {
            const bool manual_chatml = chat_template_ == "chatml";
            const std::string prompt_text = manual_chatml ? build_chatml_prompt(context,user_text)
                                                           : build_cortex_prompt(context,user_text);
            std::ostringstream json;
            json << "{\"prompt\":\"" << json_escape(prompt_text) << "\""
                 << ",\"n_predict\":" << max_new_tokens
                 << ",\"temperature\":" << std::fixed << std::setprecision(2) << temperature
                 << ",\"top_k\":40,\"top_p\":0.90,\"min_p\":0.05"
                 << ",\"repeat_penalty\":1.08,\"cache_prompt\":true,\"stream\":false"
                 << ",\"stop\":[\"<|im_end|>\"]}";
            for (int attempt = 0; attempt < 2; ++attempt) {
                const HttpResult response = http_request(port_, L"POST", L"/completion", json.str());
                if (!response.transport_ok)
                    return CortexReply{false,{},"persistent llama-server request failed: " + response.error};
                if (response.status != 200)
                    return CortexReply{false,{},"persistent llama-server HTTP " + std::to_string(response.status) + ": " + response.body};
                const auto content = json_string_field(response.body, "content");
                if (!content)
                    return CortexReply{false,{},"persistent llama-server returned malformed completion JSON"};
                std::string output = clean_cortex_output(*content);
                if (!output.empty()) return CortexReply{true,std::move(output),{}};
            }
            return CortexReply{false,{},"persistent llama-server emitted no text after retry"};
        } catch (const std::exception& ex) {
            return CortexReply{false,{},ex.what()};
        }
#else
        (void)context; (void)user_text; (void)max_new_tokens; (void)temperature;
        return CortexReply{false,{},"PersistentLlamaBackend is unavailable on this platform"};
#endif
    }

private:
#ifdef _WIN32
    bool process_running() const noexcept {
        if (process_.hProcess == nullptr) return false;
        DWORD exit_code = 0;
        return GetExitCodeProcess(process_.hProcess, &exit_code) && exit_code == STILL_ACTIVE;
    }

    void stop_server() noexcept {
        if (process_.hProcess != nullptr) {
            if (process_running()) {
                (void)TerminateProcess(process_.hProcess, 0);
                (void)WaitForSingleObject(process_.hProcess, 5000);
            }
            CloseHandle(process_.hProcess);
            process_.hProcess = nullptr;
        }
        if (process_.hThread != nullptr) {
            CloseHandle(process_.hThread);
            process_.hThread = nullptr;
        }
        port_ = 0;
    }

    bool launch_server(std::string* error) {
        const DWORD pid = GetCurrentProcessId();
        const auto tick = static_cast<unsigned long long>(GetTickCount64());
        const unsigned seed = static_cast<unsigned>((static_cast<unsigned long long>(pid) * 131ULL + tick) % 15000ULL);
        port_ = static_cast<INTERNET_PORT>(50000U + seed);

        std::ostringstream command;
        command << quote_shell(runtime_path_) << " -m " << quote_shell(model_path_)
                << " -c 4096 --host 127.0.0.1 --port " << port_ << " --log-disable";
        std::vector<char> command_line(command.str().begin(), command.str().end());
        command_line.push_back('\0');

        SECURITY_ATTRIBUTES security{};
        security.nLength = sizeof(security);
        security.bInheritHandle = TRUE;
        HANDLE null_handle = CreateFileA("NUL", GENERIC_READ | GENERIC_WRITE,
                                         FILE_SHARE_READ | FILE_SHARE_WRITE, &security,
                                         OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (null_handle == INVALID_HANDLE_VALUE) {
            if (error) *error = "failed to open NUL for persistent cortex logging";
            return false;
        }
        STARTUPINFOA startup{};
        startup.cb = sizeof(startup);
        startup.dwFlags = STARTF_USESTDHANDLES;
        startup.hStdInput = null_handle;
        startup.hStdOutput = null_handle;
        startup.hStdError = null_handle;
        PROCESS_INFORMATION created{};
        const BOOL ok = CreateProcessA(nullptr, command_line.data(), nullptr, nullptr, TRUE,
                                       CREATE_NO_WINDOW, nullptr, nullptr, &startup, &created);
        CloseHandle(null_handle);
        if (!ok) {
            if (error) *error = "failed to launch persistent llama-server (Win32 error " + std::to_string(GetLastError()) + ")";
            return false;
        }
        process_ = created;

        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(120);
        while (std::chrono::steady_clock::now() < deadline) {
            if (!process_running()) {
                DWORD exit_code = 1;
                (void)GetExitCodeProcess(process_.hProcess, &exit_code);
                if (error) *error = "persistent llama-server exited during model load with code " + std::to_string(exit_code);
                stop_server();
                return false;
            }
            const HttpResult health = http_request(port_, L"GET", L"/health");
            if (health.transport_ok && health.status == 200) {
                loaded_ = true;
                return true;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(150));
        }
        if (error) *error = "persistent llama-server did not become ready within 120 seconds";
        stop_server();
        return false;
    }

    PROCESS_INFORMATION process_{};
    INTERNET_PORT port_ = 0;
#endif
    std::string model_path_;
    std::string runtime_path_;
    std::string chat_template_;
    bool loaded_ = false;
    mutable std::mutex request_mutex_;
};

bool ToolBus::register_tool(ToolDefinition definition, Handler handler) {
    if (definition.qualified_name.empty() || !handler) return false;
    return entries_.emplace(definition.qualified_name, Entry{std::move(definition), std::move(handler)}).second;
}

bool ToolBus::contains(std::string_view name) const { return entries_.find(name) != entries_.end(); }

std::vector<ToolDefinition> ToolBus::capabilities() const {
    std::vector<ToolDefinition> result;
    result.reserve(entries_.size());
    for (const auto& [_,entry] : entries_) result.push_back(entry.definition);
    return result;
}

ToolResult ToolBus::dispatch(const ToolIntent& intent, bool allow_mutation) const {
    const std::string name = qualified_name(intent);
    const auto it = entries_.find(name);
    if (it == entries_.end()) return ToolResult{false, "unknown XENON tool: " + name, {}};
    if (it->second.definition.permission != PermissionTier::Read && !allow_mutation)
        return ToolResult{false, "tool blocked by XENON permission gate: " + name, {{"permission", permission_name(it->second.definition.permission)}}};
    auto result = it->second.handler(intent);
    result.data["tool"] = name;
    return result;
}

std::string EngineBridge::endpoint_for(std::string_view host) {
#ifdef _WIN32
    return "\\\\.\\pipe\\SpiralXenon_" + normalized_host(host);
#else
    return "/tmp/spiral_xenon_" + normalized_host(host) + ".sock";
#endif
}

HostBridgeStatus EngineBridge::probe(std::string_view host) const {
    HostBridgeStatus status{std::string(host), false, endpoint_for(host), {}};
#ifdef _WIN32
    if (WaitNamedPipeA(status.endpoint.c_str(),0)) { status.online = true; status.detail = "named pipe ready"; }
    else status.detail = "engine pipe offline";
#else
    status.detail = "native engine IPC currently implemented for Windows named pipes";
#endif
    return status;
}

ToolResult EngineBridge::request(const ToolIntent& intent, unsigned timeout_ms) const {
    ToolResult failure;
    failure.data["host"] = intent.host;
    failure.data["endpoint"] = endpoint_for(intent.host);
#ifdef _WIN32
    const std::string endpoint = failure.data["endpoint"];
    if (!WaitNamedPipeA(endpoint.c_str(), timeout_ms)) { failure.message = "XENON host adapter offline: " + intent.host; failure.data["transport"] = "named-pipe"; return failure; }
    HANDLE pipe = CreateFileA(endpoint.c_str(), GENERIC_READ|GENERIC_WRITE, 0, nullptr, OPEN_EXISTING, 0, nullptr);
    if (pipe == INVALID_HANDLE_VALUE) { failure.message = "XENON could not open host adapter: " + intent.host; return failure; }
    DWORD mode = PIPE_READMODE_MESSAGE; (void)SetNamedPipeHandleState(pipe,&mode,nullptr,nullptr);
    const std::string request_wire = encode_request(intent);
    DWORD written = 0;
    if (!WriteFile(pipe,request_wire.data(),static_cast<DWORD>(request_wire.size()),&written,nullptr)) { CloseHandle(pipe); failure.message = "XENON failed writing tool request"; return failure; }
    std::string response; char buffer[4096];
    for (;;) { DWORD read=0; const BOOL ok=ReadFile(pipe,buffer,static_cast<DWORD>(sizeof(buffer)),&read,nullptr); if(read>0) response.append(buffer,buffer+read); if(ok) break; if(GetLastError()!=ERROR_MORE_DATA) break; }
    CloseHandle(pipe);
    if (response.empty()) { failure.message = "XENON host adapter returned no response"; return failure; }
    return parse_response(response,intent.host);
#else
    (void)timeout_ms; failure.message = "XENON engine bridge unavailable on this platform"; failure.data["transport"] = "unsupported"; return failure;
#endif
}

bool LocalCortex::configure_gguf(std::string model_path, std::string runtime_path, std::string* error) noexcept {
    try {
        if (model_path.empty()) throw std::runtime_error("GGUF model path is empty");
        if (runtime_path.empty()) runtime_path = env_or_empty("SPIRAL_LLAMA_EXE");
        if (runtime_path.empty()) runtime_path = "llama-cli.exe";

        std::string policy = lower(env_or_empty("SPIRAL_CORTEX_BACKEND"));
        if (policy.empty()) policy = "auto";
        if (policy != "auto" && policy != "persistent" && policy != "legacy")
            throw std::runtime_error("SPIRAL_CORTEX_BACKEND must be auto, persistent, or legacy");

        std::filesystem::path requested(runtime_path);
        const std::string requested_name = lower(requested.filename().string());
        const bool requested_server = requested_name.find("llama-server") != std::string::npos;
#ifdef _WIN32
        const char* server_name = "llama-server.exe";
        const char* cli_name = "llama-cli.exe";
#else
        const char* server_name = "llama-server";
        const char* cli_name = "llama-cli";
#endif
        std::filesystem::path cli_runtime = requested_server ? requested.parent_path() / cli_name : requested;
        std::filesystem::path server_runtime;
        const std::string server_override = env_or_empty("SPIRAL_LLAMA_SERVER_EXE");
        if (!server_override.empty()) server_runtime = server_override;
        else server_runtime = requested_server ? requested : requested.parent_path() / server_name;

        if (policy != "legacy") {
            if (std::filesystem::exists(server_runtime)) {
                auto persistent = std::make_unique<PersistentLlamaBackend>();
                std::string persistent_error;
                if (persistent->configure(model_path, server_runtime.string(), &persistent_error)) {
                    backend_ = std::move(persistent);
                    if (error) error->clear();
                    return true;
                }
                if (policy == "persistent") throw std::runtime_error(persistent_error);
            } else if (policy == "persistent") {
                throw std::runtime_error("persistent llama server not found: " + server_runtime.string());
            }
        }

        auto legacy = std::make_unique<LegacyCliBackend>();
        std::string legacy_error;
        if (!legacy->configure(std::move(model_path), cli_runtime.string(), &legacy_error))
            throw std::runtime_error(legacy_error);
        backend_ = std::move(legacy);
        if (error) error->clear();
        return true;
    } catch (const std::exception& ex) {
        backend_.reset();
        if (error) *error = ex.what();
        return false;
    } catch (...) {
        backend_.reset();
        if (error) *error = "unknown XENON GGUF configuration failure";
        return false;
    }
}

void LocalCortex::unload() noexcept {
    if (backend_) backend_->unload();
    backend_.reset();
}

bool LocalCortex::loaded() const noexcept { return backend_ && backend_->loaded(); }
CortexState LocalCortex::state() const noexcept { return backend_ ? backend_->state() : CortexState::Offline; }
std::string_view LocalCortex::backend_name() const noexcept { return backend_ ? backend_->name() : std::string_view{"offline"}; }
const std::string& LocalCortex::model_path() const noexcept { return backend_ ? backend_->model_path() : empty_string(); }
const std::string& LocalCortex::runtime_path() const noexcept { return backend_ ? backend_->runtime_path() : empty_string(); }
const std::string& LocalCortex::chat_template() const noexcept { return backend_ ? backend_->chat_template() : empty_string(); }

CortexReply LocalCortex::generate(const SpiralContext& context, std::string_view user_text,
                                  std::size_t max_new_tokens, float temperature) const {
    if (!backend_) return CortexReply{false,{},"XENON local cortex has no backend configured"};
    return backend_->generate(context, user_text, max_new_tokens, temperature);
}

std::string current_local_datetime() {
    const auto now=std::chrono::system_clock::now(); const std::time_t value=std::chrono::system_clock::to_time_t(now); std::tm local{};
#ifdef _WIN32
    localtime_s(&local,&value);
#else
    localtime_r(&value,&local);
#endif
    std::ostringstream out; out << std::put_time(&local,"%A, %B %d, %Y %I:%M:%S %p"); return out.str();
}

std::string current_local_date_answer() {
    const auto now=std::chrono::system_clock::now(); const std::time_t value=std::chrono::system_clock::to_time_t(now); std::tm local{};
#ifdef _WIN32
    localtime_s(&local,&value);
#else
    localtime_r(&value,&local);
#endif
    std::ostringstream out; out << "Today is " << std::put_time(&local,"%A, %B %d, %Y") << "."; return out.str();
}

std::string build_cortex_prompt(const SpiralContext& context, std::string_view user_text) {
    std::ostringstream prompt;
    prompt << "SYSTEM: " << system_instructions(context) << '\n';
    for(const auto& [role,text]:context.recent_turns) prompt << (role=="assistant"?"ASSISTANT: ":"USER: ") << text << '\n';
    prompt << "USER: " << user_text << "\nASSISTANT: ";
    return prompt.str();
}

std::string clean_cortex_output(std::string text) {
    const std::string performance_marker = "[ Prompt:";
    if (const auto performance = text.find(performance_marker); performance != std::string::npos) text.resize(performance);
    if (const auto exiting = text.find("\nExiting..."); exiting != std::string::npos) text.resize(exiting);
    std::istringstream in(text); std::ostringstream out; std::string line; bool first=true;
    while(std::getline(in,line)){
        const std::string l=lower(line);
        if(l.find("llama_")==0 || l.find("load_tensors")!=std::string::npos || l.find("main: ")!=std::string::npos || l.find("system_info")!=std::string::npos || l.find("sampler")!=std::string::npos) continue;
        if(!first) out << '\n'; first=false; out << line;
    }
    std::string result=out.str();
    while(!result.empty() && std::isspace(static_cast<unsigned char>(result.front()))) result.erase(result.begin());
    while(!result.empty() && std::isspace(static_cast<unsigned char>(result.back()))) result.pop_back();
    const std::string assistant_token = "<|im_start|>assistant";
    if (const auto chatml = result.rfind(assistant_token); chatml != std::string::npos) {
        result = result.substr(chatml + assistant_token.size());
        if (const auto end = result.find("<|im_end|>"); end != std::string::npos) result.resize(end);
    } else {
        const std::string marker="ASSISTANT:"; const auto pos=result.rfind(marker);
        if(pos!=std::string::npos) result=result.substr(pos+marker.size());
        else if (const auto truncated = result.rfind("(truncated)"); truncated != std::string::npos) {
            const auto generated = result.find_first_of("\r\n", truncated);
            result = generated == std::string::npos ? std::string{} : result.substr(generated + 1);
        }
    }
    while(!result.empty() && std::isspace(static_cast<unsigned char>(result.front()))) result.erase(result.begin());
    while(!result.empty() && std::isspace(static_cast<unsigned char>(result.back()))) result.pop_back();
    return result;
}

std::optional<ToolIntent> parse_tool_call(std::string_view text) {
    const auto pos=text.find("TOOL_CALL"); if(pos==std::string_view::npos) return std::nullopt;
    std::istringstream in(std::string(text.substr(pos+9))); std::string qualified; in >> qualified; if(qualified.empty()) return std::nullopt;
    const auto dot=qualified.find('.'); if(dot==std::string::npos) return std::nullopt;
    ToolIntent intent; intent.host=qualified.substr(0,dot); intent.action=qualified.substr(dot+1);
    std::string token; while(in>>token){ const auto eq=token.find('='); if(eq!=std::string::npos) intent.arguments[token.substr(0,eq)]=token.substr(eq+1); }
    return intent;
}

ToolBus make_default_tool_bus() {
    ToolBus bus;
    register_engine_tool(bus,"hakui.inspect_world",PermissionTier::Read,"Read live Hakui world state.");
    register_engine_tool(bus,"hakui.inspect_avatar",PermissionTier::Read,"Read live Hakui avatar state.");
    register_engine_tool(bus,"hakui.inspect_inventory",PermissionTier::Read,"Read live Hakui inventory state.");
    register_engine_tool(bus,"hakui.find_nearby_object",PermissionTier::Read,"Query nearby Hakui objects.");
    register_engine_tool(bus,"hakui.move_to",PermissionTier::Write,"Request validated Hakui movement.");
    register_engine_tool(bus,"hakui.set_animation",PermissionTier::Write,"Request a Hakui animation state.");
    register_engine_tool(bus,"hakui.spawn_allowed_object",PermissionTier::Write,"Spawn an allow-listed Hakui object.");
    register_engine_tool(bus,"etherplay.get_current_track",PermissionTier::Read,"Read EtherPlay current-track state.");
    register_engine_tool(bus,"etherplay.get_library_state",PermissionTier::Read,"Read EtherPlay library state.");
    register_engine_tool(bus,"etherplay.analyze_audio",PermissionTier::Read,"Read spectral/rhythm/pitch/transient analysis.");
    register_engine_tool(bus,"etherplay.extract_features",PermissionTier::Read,"Extract structured listening features.");
    register_engine_tool(bus,"etherplay.seek",PermissionTier::Write,"Seek EtherPlay playback.");
    register_engine_tool(bus,"etherplay.queue_track",PermissionTier::Write,"Queue a track.");
    register_engine_tool(bus,"etherplay.set_metadata",PermissionTier::Write,"Update validated metadata.");
    register_engine_tool(bus,"etherbeat.get_project_state",PermissionTier::Read,"Read EtherBeat project state.");
    register_engine_tool(bus,"etherbeat.get_arrangement",PermissionTier::Read,"Read EtherBeat arrangement.");
    register_engine_tool(bus,"etherbeat.get_stems",PermissionTier::Read,"Read EtherBeat stems.");
    register_engine_tool(bus,"etherbeat.analyze_reference",PermissionTier::Read,"Analyze a reference track.");
    register_engine_tool(bus,"etherbeat.create_arrangement",PermissionTier::Write,"Create an arrangement plan.");
    register_engine_tool(bus,"etherbeat.generate_midi",PermissionTier::Write,"Generate MIDI.");
    register_engine_tool(bus,"etherbeat.select_drum_pattern",PermissionTier::Write,"Select a drum pattern.");
    register_engine_tool(bus,"etherbeat.build_chord_progression",PermissionTier::Write,"Build a chord progression.");
    register_engine_tool(bus,"etherbeat.request_stem",PermissionTier::Write,"Request a stem render.");
    register_engine_tool(bus,"etherbeat.apply_etherseam",PermissionTier::Write,"Apply EtherSeam.");
    register_engine_tool(bus,"etherbeat.export_song",PermissionTier::Critical,"Export the song.");
    return bus;
}

} // namespace spiral::xenon
