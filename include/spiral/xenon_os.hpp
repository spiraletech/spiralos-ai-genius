#pragma once

#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace spiral::xenon {

enum class PermissionTier { Read, Propose, Write, Critical };
enum class CortexState { Offline, Ready, Error };

struct ToolIntent {
    std::string host;
    std::string action;
    std::map<std::string, std::string> arguments;
};

struct ToolResult {
    bool success = false;
    std::string message;
    std::map<std::string, std::string> data;
};

struct ToolDefinition {
    std::string qualified_name;
    PermissionTier permission = PermissionTier::Read;
    std::string description;
};

class ToolBus final {
public:
    using Handler = std::function<ToolResult(const ToolIntent&)>;
    bool register_tool(ToolDefinition definition, Handler handler);
    [[nodiscard]] bool contains(std::string_view qualified_name) const;
    [[nodiscard]] std::vector<ToolDefinition> capabilities() const;
    [[nodiscard]] ToolResult dispatch(const ToolIntent& intent, bool allow_mutation = false) const;
private:
    struct Entry { ToolDefinition definition; Handler handler; };
    std::map<std::string, Entry, std::less<>> entries_;
};

struct HostBridgeStatus {
    std::string host;
    bool online = false;
    std::string endpoint;
    std::string detail;
};

class EngineBridge final {
public:
    [[nodiscard]] static std::string endpoint_for(std::string_view host);
    [[nodiscard]] HostBridgeStatus probe(std::string_view host) const;
    [[nodiscard]] ToolResult request(const ToolIntent& intent, unsigned timeout_ms = 1200) const;
};

struct SpiralContext {
    std::string host;
    std::string host_context;
    std::string local_datetime;
    std::string organic_topic;
    float organic_focus = 0.0F;
    float organic_curiosity = 0.0F;
    float organic_coherence = 0.0F;
    std::vector<std::string> relevant_memories;
    std::vector<std::pair<std::string, std::string>> recent_turns;
    std::vector<ToolResult> recent_tool_results;
};

struct CortexReply { bool ok = false; std::string text; std::string error; };

class ICortexBackend {
public:
    virtual ~ICortexBackend() = default;
    virtual bool configure(std::string model_path, std::string runtime_path, std::string* error = nullptr) noexcept = 0;
    virtual void unload() noexcept = 0;
    [[nodiscard]] virtual bool loaded() const noexcept = 0;
    [[nodiscard]] virtual CortexState state() const noexcept = 0;
    [[nodiscard]] virtual std::string_view name() const noexcept = 0;
    [[nodiscard]] virtual const std::string& model_path() const noexcept = 0;
    [[nodiscard]] virtual const std::string& runtime_path() const noexcept = 0;
    [[nodiscard]] virtual const std::string& chat_template() const noexcept = 0;
    [[nodiscard]] virtual CortexReply generate(const SpiralContext& context, std::string_view user_text,
                                               std::size_t max_new_tokens, float temperature) const = 0;
};

class LocalCortex final {
public:
    LocalCortex() = default;
    ~LocalCortex() = default;
    LocalCortex(const LocalCortex&) = delete;
    LocalCortex& operator=(const LocalCortex&) = delete;
    LocalCortex(LocalCortex&&) noexcept = default;
    LocalCortex& operator=(LocalCortex&&) noexcept = default;

    bool configure_gguf(std::string model_path, std::string runtime_path = {}, std::string* error = nullptr) noexcept;
    void unload() noexcept;
    [[nodiscard]] bool loaded() const noexcept;
    [[nodiscard]] CortexState state() const noexcept;
    [[nodiscard]] std::string_view backend_name() const noexcept;
    [[nodiscard]] const std::string& model_path() const noexcept;
    [[nodiscard]] const std::string& runtime_path() const noexcept;
    [[nodiscard]] const std::string& chat_template() const noexcept;
    [[nodiscard]] CortexReply generate(const SpiralContext& context, std::string_view user_text,
                                       std::size_t max_new_tokens = 384, float temperature = 0.62F) const;
private:
    std::unique_ptr<ICortexBackend> backend_;
};

[[nodiscard]] std::string current_local_datetime();
[[nodiscard]] std::string current_local_date_answer();
[[nodiscard]] std::string build_cortex_prompt(const SpiralContext& context, std::string_view user_text);
[[nodiscard]] std::string clean_cortex_output(std::string text);
[[nodiscard]] std::optional<ToolIntent> parse_tool_call(std::string_view text);
[[nodiscard]] ToolBus make_default_tool_bus();

} // namespace spiral::xenon