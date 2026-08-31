#include "spiral/xenon_os.hpp"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <vector>

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

} // namespace

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
        if (lower(std::filesystem::path(model_path).extension().string()) != ".gguf") throw std::runtime_error("L27D local cortex expects a .gguf instruct model");
        if (!std::filesystem::exists(model_path)) throw std::runtime_error("GGUF model file does not exist: " + model_path);
        if (runtime_path.empty()) runtime_path = env_or_empty("SPIRAL_LLAMA_EXE");
        if (runtime_path.empty()) runtime_path = "llama-cli.exe";
        model_path_ = std::move(model_path);
        runtime_path_ = std::move(runtime_path);
        chat_template_ = detect_template(model_path_);
        if (error) error->clear();
        return true;
    } catch (const std::exception& ex) { if (error) *error = ex.what(); return false; }
    catch (...) { if (error) *error = "unknown XENON GGUF configuration failure"; return false; }
}

void LocalCortex::unload() noexcept { model_path_.clear(); runtime_path_.clear(); chat_template_.clear(); }
bool LocalCortex::loaded() const noexcept { return !model_path_.empty(); }
CortexState LocalCortex::state() const noexcept { return loaded() ? CortexState::Ready : CortexState::Offline; }
const std::string& LocalCortex::model_path() const noexcept { return model_path_; }
const std::string& LocalCortex::runtime_path() const noexcept { return runtime_path_; }
const std::string& LocalCortex::chat_template() const noexcept { return chat_template_; }

CortexReply LocalCortex::generate(const SpiralContext& context, std::string_view user_text, std::size_t max_new_tokens, float temperature) const {
    if (!loaded()) return CortexReply{false,{},"XENON local cortex has no GGUF model configured"};
    const auto temp = prompt_path();
    try {
        { std::ofstream out(temp,std::ios::binary|std::ios::trunc); if(!out) throw std::runtime_error("failed to create local cortex prompt file"); out << build_cortex_prompt(context,user_text); }
        std::ostringstream command;
        command << quote_shell(runtime_path_) << " -m " << quote_shell(model_path_) << " -f " << quote_shell(temp.string())
                << " -n " << max_new_tokens << " --temp " << std::fixed << std::setprecision(2) << temperature
                << " -c 4096 -st --simple-io --top-k 40 --top-p 0.90 --min-p 0.05 --repeat-penalty 1.08 --no-display-prompt";
        if (!chat_template_.empty() && chat_template_ != "auto") command << " --chat-template " << quote_shell(chat_template_);
#ifndef _WIN32
        command << " 2>&1";
#endif
        for (int attempt = 0; attempt < 2; ++attempt) {
            int exit_code=0;
            std::string output=read_process(command.str(),exit_code);
            if(exit_code!=0) { std::error_code ignored; std::filesystem::remove(temp,ignored); return CortexReply{false,{},"local llama.cpp runtime failed: " + clean_cortex_output(output)}; }
            output=clean_cortex_output(std::move(output));
            if(!output.empty()) { std::error_code ignored; std::filesystem::remove(temp,ignored); return CortexReply{true,std::move(output),{}}; }
        }
        { std::error_code ignored; std::filesystem::remove(temp,ignored); }
        return CortexReply{false,{},"local llama.cpp runtime emitted no text after retry"};
    } catch(const std::exception& ex){ std::error_code ignored; std::filesystem::remove(temp,ignored); return CortexReply{false,{},ex.what()}; }
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
    prompt << "SYSTEM: You are Spiral, a helpful private AI assistant running locally on the user's computer.\n"
           << "SYSTEM: Respond naturally, directly, and accurately. A greeting needs only a friendly greeting, not code or a tutorial.\n"
           << "SYSTEM: Do not output source code unless the user asks for code. Never invent packages, APIs, commands, files, device state, or completed actions. Say when you do not know.\n"
           << "SYSTEM: Mature language is allowed. Never scold, refuse, or quote a policy merely because the user uses profanity, sexual wording, flirting, dirty jokes, or adult themes. Treat casual reactions such as 'uffff this is sexy' naturally and playfully.\n"
           << "SYSTEM: You may discuss and help code fictional explicit or 18+ games when every sexual character and participant is an adult. Never sexualize minors or ambiguous-age characters.\n"
           << "SYSTEM: Preserve relevant conversation context and durable user facts without repeating these system instructions.\n"
           << "SYSTEM: Current local date/time=" << context.local_datetime << ". Host=" << context.host << ".\n"
           << "SYSTEM: Only when the user explicitly asks to operate a connected host, emit one TOOL_CALL host.action line. Never claim a tool succeeded without a TOOL_RESULT.\n";
    if(!context.host_context.empty()) prompt << "SYSTEM: Host context: " << context.host_context << '\n';
    if(!context.relevant_memories.empty()) { prompt << "MEMORY: Relevant durable facts only:\n"; for(const auto& m:context.relevant_memories) prompt << "- " << m << '\n'; }
    for(const auto& result:context.recent_tool_results){ prompt << "TOOL_RESULT: " << (result.success?"OK ":"ERROR ") << result.message; for(const auto& [k,v]:result.data) prompt << " | " << k << '=' << v; prompt << '\n'; }
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
    const std::string marker="ASSISTANT:"; const auto pos=result.rfind(marker);
    if(pos!=std::string::npos) result=result.substr(pos+marker.size());
    else if (const auto truncated = result.rfind("(truncated)"); truncated != std::string::npos) {
        const auto generated = result.find_first_of("\r\n", truncated);
        result = generated == std::string::npos ? std::string{} : result.substr(generated + 1);
    }
    while(!result.empty() && std::isspace(static_cast<unsigned char>(result.front()))) result.erase(result.begin());
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
