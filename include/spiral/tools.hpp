#pragma once

#include <functional>
#include <string>
#include <string_view>
#include <vector>

namespace spiral::tools {

struct ToolSpec {
    std::string name;
    std::string description;
};

struct ToolResult {
    bool ok = false;
    std::string output;
    std::string error;

    static ToolResult success(std::string output = {});
    static ToolResult failure(std::string error);
};

using ToolHandler = std::function<ToolResult(std::string_view input)>;

class ToolRegistry {
public:
    void register_tool(ToolSpec spec, ToolHandler handler);
    [[nodiscard]] bool contains(std::string_view name) const;
    [[nodiscard]] std::vector<ToolSpec> list() const;
    [[nodiscard]] ToolResult invoke(std::string_view name, std::string_view input = {}) const;

private:
    struct Entry { ToolSpec spec; ToolHandler handler; };
    std::vector<Entry> entries_;
};

} // namespace spiral::tools
