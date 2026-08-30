#include "spiral/xenon_os.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <system_error>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

namespace spiral::xenon {
namespace {

std::string qualified_name(const ToolIntent& intent) {
    if (intent.host.empty()) return intent.action;
    if (intent.action.rfind(intent.host + ".", 0) == 0) return intent.action;
    return intent.host + "." + intent.action;
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
    FILE* pipe = _popen(command.c_str(), "r");
#else
    FILE* pipe = popen(command.c_str(), "r");
#endif
    if (pipe == nullptr) throw std::runtime_error("failed to launch local cortex runtime");
    std::string output;
    char buffer[4096];
    while (std::fgets(buffer, static_cast<int>(sizeof(buffer)), pipe) != nullptr) output += buffer;
#ifdef _WIN32
    exit_code = _pclose(pipe);
#else
    exit_code = pclose(pipe);
#endif
    return output;
}

std::string env_or_empty(const char* name) {
    const char* value = std::getenv(name);
    return value == nullptr ? std::string{} : std::string(value);
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
    out << "XENON/1\n" << "action=" << intent.action << '\n';
    for (const auto& [key, value] : intent.arguments) out << "arg." << key << '=' << value << '\n';
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
        else if (key.rfind("data.", 0) == 0) result.data[key.substr(5)] = value;
    }
    if (result.message.empty()) result.message = result.success ? "engine tool completed" : "engine tool failed";
    return result;
}

void register_engine_tool(ToolBus& bus, std::string name, PermissionTier permission, std::string description) {
    ToolDefinition definition{std::move(name), permission, std::move(description)};
    (void)bus.register_tool(std::move(definition), [](const ToolIntent& intent) {
        static const EngineBridge bridge;
        return bridge.request(intent);
    });
}

} // namespace

bool ToolBus::register_tool(ToolDefinition definition, Handler handler) {
    if (definition.qualified_name.empty() || !handler) return false;
    return entries_.emplace(definition.qualified_name, Entry{std::move(definition), std::move(handler)}).second;
}

bool ToolBus::contains(std::string_view name) const { return entries_.find(name) != entries_.end(); }

std::vector<ToolDefinition> ToolBus::capabilities() const {
    std::vector<ToolDefinition> result;
    result.reserve(entries_.size());
    for (const auto& [_, entry] : entries_) result.push_back(entry.definition);
    return result;
}

ToolResult ToolBus::dispatch(const ToolIntent& intent, bool allow_mutation) const {
    const std::string name = qualified_name(intent);
    const auto it = entries_.find(name);
    if (it == entries_.end()) return ToolResult{false, "unknown XENON tool: " + name, {}};
    if (it->second.definition.permission != PermissionTier::Read && !allow_mutation)
        return ToolResult{false, "tool blocked by XENON permission gate: " + name, {{"permission", "mutation-required"}}};
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
    HostBridgeStatus status;
    status.host = std::string(host);
    status.endpoint = endpoint_for(host);
#ifdef _WIN32
    if (WaitNamedPipeA(status.endpoint.c_str(), 0)) { status.online = true; status.detail = "named pipe ready"; }
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
    if (!WaitNamedPipeA(endpoint.c_str(), timeout_ms)) {
        failure.message = "XENON host adapter offline: " + intent.host;
        failure.data["transport"] = "named-pipe";
        return failure;
    }
    HANDLE pipe = CreateFileA(endpoint.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr, OPEN_EXISTING, 0, nullptr);
    if (pipe == INVALID_HANDLE_VALUE) { failure.message = "XENON could not open host adapter: " + intent.host; return failure; }
    DWORD mode = PIPE_READMODE_MESSAGE;
    (void)SetNamedPipeHandleState(pipe, &mode, nullptr, nullptr);
    const std::string request_wire = encode_request(intent);
    DWORD written = 0;
    if (!WriteFile(pipe, request_wire.data(), static_cast<DWORD>(request_wire.size()), &written, nullptr)) {
        CloseHandle(pipe); failure.message = "XENON failed writing tool request"; return failure;
    }
    std::string response;
    char buffer[4096];
    for (;;) {
        DWORD read = 0;
        const BOOL ok = ReadFile(pipe, buffer, static_cast<DWORD>(sizeof(buffer)), &read, nullptr);
        if (read > 0) response.append(buffer, buffer + read);
        if (ok) break;
        if (GetLastError() != ERROR_MORE_DATA) break;
    }
    CloseHandle(pipe);
    if (response.empty()) { failure.message = "XENON host adapter returned no response"; return failure; }
    return parse_response(response, intent.host);
#else
    (void)timeout_ms;
    failure.message = "XENON engine bridge unavailable on this platform";
    failure.data["transport"] = "unsupported";
    return failure;
#endif
}

bool LocalCortex::configure_gguf(std::string model_path, std::string runtime_path, std::string* error) noexcept {
    try {
        if (model_path.empty()) throw std::runtime_error("GGUF model path is empty");
        if (std::filesystem::path(model_path).extension() != ".gguf") throw std::runtime_error("L27 XENON local cortex expects a .gguf model");
        if (!std::filesystem::exists(model_path)) throw std::runtime_error("GGUF model file does not exist: " + model_path);
        if (runtime_path.empty()) runtime_path = env_or_empty("SPIRAL_LLAMA_EXE");
        if (runtime_path.empty()) runtime_path = "llama-cli.exe";
        model_path_ = std::move(model_path);
        runtime_path_ = std::move(runtime_path);
        if (error != nullptr) error->clear();
        return true;
    } catch (const std::exception& ex) { if (error != nullptr) *error = ex.what(); return false; }
    catch (...) { if (error != nullptr) *error = "unknown XENON GGUF configuration failure"; return false; }
}

void LocalCortex::unload() noexcept { model_path_.clear(); runtime_path_.clear(); }
bool LocalCortex::loaded() const noexcept { return !model_path_.empty(); }
const std::string& LocalCortex::model_path() const noexcept { return model_path_; }
const std::string& LocalCortex::runtime_path() const noexcept { return runtime_path_; }

CortexReply LocalCortex::generate(const SpiralContext& context, std::string_view user_text,
                                  std::size_t max_new_tokens, float temperature) const {
    if (!loaded()) return CortexReply{false, {}, "XENON local cortex has no GGUF model configured"};
    const auto temp = prompt_path();
    try {
        { std::ofstream out(temp, std::ios::binary | std::ios::trunc); if (!out) throw std::runtime_error("failed to create local cortex prompt file"); out << build_cortex_prompt(context, user_text); }
        std::ostringstream command;
        command << quote_shell(runtime_path_) << " -m " << quote_shell(model_path_)
                << " -f " << quote_shell(temp.string()) << " -n " << max_new_tokens
                << " --temp " << std::fixed << std::setprecision(2) << temperature << " --no-display-prompt 2>&1";
        int exit_code = 0;
        std::string output = read_process(command.str(), exit_code);
        std::error_code ignored; std::filesystem::remove(temp, ignored);
        if (exit_code != 0) return CortexReply{false, {}, "local llama.cpp runtime failed: " + output};
        if (output.empty()) return CortexReply{false, {}, "local llama.cpp runtime emitted no text"};
        return CortexReply{true, std::move(output), {}};
    } catch (const std::exception& ex) { std::error_code ignored; std::filesystem::remove(temp, ignored); return CortexReply{false, {}, ex.what()}; }
}

std::string current_local_datetime() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t value = std::chrono::system_clock::to_time_t(now);
    std::tm local{};
#ifdef _WIN32
    localtime_s(&local, &value);
#else
    localtime_r(&value, &local);
#endif
    std::ostringstream out; out << std::put_time(&local, "%A, %Y-%m-%d %H:%M:%S %Z"); return out.str();
}

std::string build_cortex_prompt(const SpiralContext& context, std::string_view user_text) {
    std::ostringstream prompt;
    prompt << "SYSTEM: You are the local language cortex of Spiral Ether AI running through XENON OS.\n"
           << "SYSTEM: ORGANIC is persistent identity/executive memory. Native C++ hosts are authoritative.\n"
           << "SYSTEM: Tool requests must use XENON capabilities; never claim execution without a successful ToolResult.\n"
           << "SYSTEM: Current host=" << context.host << ". Current local date/time=" << context.local_datetime << ".\n";
    if (!context.host_context.empty()) prompt << "SYSTEM: Host context: " << context.host_context << '\n';
    prompt << "SYSTEM: Organic topic=" << context.organic_topic << " focus=" << context.organic_focus
           << " curiosity=" << context.organic_curiosity << " coherence=" << context.organic_coherence << ".\n";
    for (const auto& result : context.recent_tool_results) {
        prompt << "TOOL: " << (result.success ? "OK " : "ERROR ") << result.message;
        for (const auto& [key, value] : result.data) prompt << " | " << key << '=' << value;
        prompt << '\n';
    }
    for (const auto& [role, text] : context.recent_turns) prompt << (role == "assistant" ? "ASSISTANT: " : "USER: ") << text << '\n';
    prompt << "USER: " << user_text << "\nASSISTANT: ";
    return prompt.str();
}

ToolBus make_default_tool_bus() {
    ToolBus bus;
    register_engine_tool(bus, "hakui.inspect_world", PermissionTier::Read, "Read live Hakui world state.");
    register_engine_tool(bus, "hakui.inspect_avatar", PermissionTier::Read, "Read live Hakui avatar state.");
    register_engine_tool(bus, "hakui.inspect_inventory", PermissionTier::Read, "Read live Hakui inventory state.");
    register_engine_tool(bus, "hakui.find_nearby_object", PermissionTier::Read, "Query nearby Hakui objects.");
    register_engine_tool(bus, "hakui.move_to", PermissionTier::Write, "Request validated Hakui movement.");
    register_engine_tool(bus, "hakui.set_animation", PermissionTier::Write, "Request a Hakui animation state.");
    register_engine_tool(bus, "hakui.spawn_allowed_object", PermissionTier::Write, "Spawn an allow-listed Hakui object.");
    register_engine_tool(bus, "etherplay.get_current_track", PermissionTier::Read, "Read EtherPlay current-track state.");
    register_engine_tool(bus, "etherplay.get_library_state", PermissionTier::Read, "Read EtherPlay library state.");
    register_engine_tool(bus, "etherplay.analyze_audio", PermissionTier::Read, "Read spectral/rhythm/pitch/transient analysis from EtherPlay.");
    register_engine_tool(bus, "etherplay.extract_features", PermissionTier::Read, "Extract structured listening features.");
    register_engine_tool(bus, "etherplay.seek", PermissionTier::Write, "Seek playback through EtherPlay.");
    register_engine_tool(bus, "etherplay.queue_track", PermissionTier::Write, "Queue a track through EtherPlay.");
    register_engine_tool(bus, "etherplay.set_metadata", PermissionTier::Write, "Update validated media metadata.");
    register_engine_tool(bus, "etherbeat.get_project_state", PermissionTier::Read, "Read EtherBeat project state.");
    register_engine_tool(bus, "etherbeat.get_arrangement", PermissionTier::Read, "Read EtherBeat arrangement state.");
    register_engine_tool(bus, "etherbeat.get_stems", PermissionTier::Read, "Read EtherBeat stem state.");
    register_engine_tool(bus, "etherbeat.analyze_reference", PermissionTier::Read, "Analyze a reference through listening state.");
    register_engine_tool(bus, "etherbeat.create_arrangement", PermissionTier::Write, "Create a validated arrangement plan.");
    register_engine_tool(bus, "etherbeat.generate_midi", PermissionTier::Write, "Generate MIDI from a producer plan.");
    register_engine_tool(bus, "etherbeat.select_drum_pattern", PermissionTier::Write, "Select or generate a drum pattern.");
    register_engine_tool(bus, "etherbeat.build_chord_progression", PermissionTier::Write, "Build a chord progression.");
    register_engine_tool(bus, "etherbeat.request_stem", PermissionTier::Write, "Request a generated or rendered stem.");
    register_engine_tool(bus, "etherbeat.apply_etherseam", PermissionTier::Write, "Apply EtherSeam at an arrangement boundary.");
    register_engine_tool(bus, "etherbeat.export_song", PermissionTier::Critical, "Export final song output after explicit authorization.");
    return bus;
}

} // namespace spiral::xenon
