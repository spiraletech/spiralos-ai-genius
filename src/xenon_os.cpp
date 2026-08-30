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
    for (char ch : value) {
        if (ch == '"') out += "\\\"";
        else out += ch;
    }
    out += '"';
    return out;
#else
    std::string out = "'";
    for (char ch : value) {
        if (ch == '\'') out += "'\\''";
        else out += ch;
    }
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

ToolResult read_snapshot(std::string host, std::string capability, const ToolIntent& intent) {
    ToolResult result;
    result.success = true;
    result.message = host + " read-only capability ready";
    result.data["host"] = std::move(host);
    result.data["capability"] = std::move(capability);
    for (const auto& [key, value] : intent.arguments) result.data["arg." + key] = value;
    return result;
}

} // namespace

bool ToolBus::register_tool(ToolDefinition definition, Handler handler) {
    if (definition.qualified_name.empty() || !handler) return false;
    return entries_.emplace(definition.qualified_name, Entry{std::move(definition), std::move(handler)}).second;
}

bool ToolBus::contains(std::string_view name) const {
    return entries_.find(name) != entries_.end();
}

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
    if (it->second.definition.permission != PermissionTier::Read && !allow_mutation) {
        return ToolResult{false, "tool blocked by XENON permission gate: " + name, {}};
    }
    return it->second.handler(intent);
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
    } catch (const std::exception& ex) {
        if (error != nullptr) *error = ex.what();
        return false;
    } catch (...) {
        if (error != nullptr) *error = "unknown XENON GGUF configuration failure";
        return false;
    }
}

void LocalCortex::unload() noexcept {
    model_path_.clear();
    runtime_path_.clear();
}

bool LocalCortex::loaded() const noexcept { return !model_path_.empty(); }
const std::string& LocalCortex::model_path() const noexcept { return model_path_; }
const std::string& LocalCortex::runtime_path() const noexcept { return runtime_path_; }

CortexReply LocalCortex::generate(const SpiralContext& context, std::string_view user_text,
                                  std::size_t max_new_tokens, float temperature) const {
    if (!loaded()) return CortexReply{false, {}, "XENON local cortex has no GGUF model configured"};

    const auto temp = prompt_path();
    try {
        {
            std::ofstream out(temp, std::ios::binary | std::ios::trunc);
            if (!out) throw std::runtime_error("failed to create local cortex prompt file");
            out << build_cortex_prompt(context, user_text);
        }

        std::ostringstream command;
        command << quote_shell(runtime_path_)
                << " -m " << quote_shell(model_path_)
                << " -f " << quote_shell(temp.string())
                << " -n " << max_new_tokens
                << " --temp " << std::fixed << std::setprecision(2) << temperature
                << " --no-display-prompt 2>&1";
        int exit_code = 0;
        std::string output = read_process(command.str(), exit_code);
        std::error_code ignored;
        std::filesystem::remove(temp, ignored);
        if (exit_code != 0) return CortexReply{false, {}, "local llama.cpp runtime failed: " + output};
        if (output.empty()) return CortexReply{false, {}, "local llama.cpp runtime emitted no text"};
        return CortexReply{true, std::move(output), {}};
    } catch (const std::exception& ex) {
        std::error_code ignored;
        std::filesystem::remove(temp, ignored);
        return CortexReply{false, {}, ex.what()};
    }
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
    std::ostringstream out;
    out << std::put_time(&local, "%A, %Y-%m-%d %H:%M:%S %Z");
    return out.str();
}

std::string build_cortex_prompt(const SpiralContext& context, std::string_view user_text) {
    std::ostringstream prompt;
    prompt << "SYSTEM: You are the local language cortex of Spiral Ether AI running through XENON OS.\n"
           << "SYSTEM: XENON OS is the host-neutral intelligence/tool runtime. ORGANIC remains persistent identity and long-term executive memory.\n"
           << "SYSTEM: Never claim a tool action happened unless a ToolResult confirms it. Native C++ engines are authoritative.\n"
           << "SYSTEM: Current host=" << context.host << ". Current local date/time=" << context.local_datetime << ".\n";
    if (!context.host_context.empty()) prompt << "SYSTEM: Host context: " << context.host_context << '\n';
    prompt << "SYSTEM: Organic topic=" << context.organic_topic
           << " focus=" << context.organic_focus
           << " curiosity=" << context.organic_curiosity
           << " coherence=" << context.organic_coherence << ".\n";
    for (const auto& result : context.recent_tool_results) {
        prompt << "TOOL: " << (result.success ? "OK " : "ERROR ") << result.message;
        for (const auto& [key, value] : result.data) prompt << " | " << key << '=' << value;
        prompt << '\n';
    }
    for (const auto& [role, text] : context.recent_turns) {
        prompt << (role == "assistant" ? "ASSISTANT: " : "USER: ") << text << '\n';
    }
    prompt << "USER: " << user_text << "\nASSISTANT: ";
    return prompt.str();
}

ToolBus make_default_tool_bus() {
    ToolBus bus;
    const auto add_read = [&bus](std::string name, std::string host, std::string capability, std::string description) {
        ToolDefinition definition{name, PermissionTier::Read, std::move(description)};
        bus.register_tool(std::move(definition), [host = std::move(host), capability = std::move(capability)](const ToolIntent& intent) {
            return read_snapshot(host, capability, intent);
        });
    };

    add_read("hakui.inspect_world", "hakui", "inspect_world", "Read Hakui world state through its adapter.");
    add_read("hakui.inspect_avatar", "hakui", "inspect_avatar", "Read Hakui avatar state through its adapter.");
    add_read("hakui.inspect_inventory", "hakui", "inspect_inventory", "Read Hakui inventory state through its adapter.");

    add_read("etherplay.get_current_track", "etherplay", "get_current_track", "Read EtherPlay current-track state.");
    add_read("etherplay.get_library_state", "etherplay", "get_library_state", "Read EtherPlay library state.");
    add_read("etherplay.analyze_audio", "etherplay", "analyze_audio", "Read spectral/rhythm/pitch/transient analysis from EtherPlay.");

    add_read("etherbeat.get_project_state", "etherbeat", "get_project_state", "Read EtherBeat project state.");
    add_read("etherbeat.get_arrangement", "etherbeat", "get_arrangement", "Read EtherBeat arrangement state.");
    add_read("etherbeat.get_stems", "etherbeat", "get_stems", "Read EtherBeat stem state.");

    return bus;
}

} // namespace spiral::xenon
