#include "spiral/tools.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace spiral::tools {

ToolResult ToolResult::success(std::string output) { return ToolResult{true, std::move(output), {}}; }
ToolResult ToolResult::failure(std::string error) { return ToolResult{false, {}, std::move(error)}; }

void ToolRegistry::register_tool(ToolSpec spec, ToolHandler handler) {
    if (spec.name.empty()) throw std::invalid_argument("tool name must not be empty");
    if (!handler) throw std::invalid_argument("tool handler must be callable");
    if (contains(spec.name)) throw std::invalid_argument("tool name already registered");
    entries_.push_back(Entry{std::move(spec), std::move(handler)});
}

bool ToolRegistry::contains(std::string_view name) const {
    return std::any_of(entries_.begin(), entries_.end(), [&](const Entry& e) { return e.spec.name == name; });
}

std::vector<ToolSpec> ToolRegistry::list() const {
    std::vector<ToolSpec> out;
    out.reserve(entries_.size());
    for (const auto& entry : entries_) out.push_back(entry.spec);
    std::sort(out.begin(), out.end(), [](const ToolSpec& a, const ToolSpec& b) { return a.name < b.name; });
    return out;
}

ToolResult ToolRegistry::invoke(std::string_view name, std::string_view input) const {
    const auto it = std::find_if(entries_.begin(), entries_.end(), [&](const Entry& e) { return e.spec.name == name; });
    if (it == entries_.end()) return ToolResult::failure("unknown tool: " + std::string(name));
    try { return it->handler(input); }
    catch (const std::exception& e) { return ToolResult::failure(e.what()); }
    catch (...) { return ToolResult::failure("tool threw an unknown exception"); }
}

} // namespace spiral::tools
