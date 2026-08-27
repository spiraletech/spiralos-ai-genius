#pragma once

#include "spiral/memory.hpp"
#include "spiral/tools.hpp"

#include <cstddef>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

namespace spiral::agent {

enum class IntentKind {
    General,
    ToolUse,
    MultiStep,
};

struct Intent {
    IntentKind kind = IntentKind::General;
    std::string goal;
    std::vector<std::string> candidate_tools;
};

enum class TaskStatus {
    Pending,
    Running,
    Succeeded,
    Failed,
    Skipped,
};

struct TaskNode {
    std::size_t id = 0;
    std::string description;
    std::string tool_name;
    std::string input;
    std::vector<std::size_t> depends_on;
    TaskStatus status = TaskStatus::Pending;
    std::size_t attempts = 0;
    tools::ToolResult result;
};

struct TaskGraph {
    std::vector<TaskNode> tasks;

    void validate() const;
    [[nodiscard]] bool all_succeeded() const noexcept;
};

struct AgentContext {
    std::string goal;
    Intent intent;
    std::vector<memory::MemoryHit> memories;
    std::vector<tools::ToolSpec> tools;
};

struct Critique {
    bool acceptable = false;
    std::string reason;
};

struct RepairDecision {
    bool retry = false;
    std::string tool_name;
    std::string input;
};

using PlannerPolicy = std::function<TaskGraph(const AgentContext&)>;
using CriticPolicy = std::function<Critique(const AgentContext&, const TaskNode&)>;
using RepairPolicy = std::function<RepairDecision(
    const AgentContext&,
    const TaskNode&,
    std::string_view reason)>;

struct AgentPolicies {
    PlannerPolicy planner;
    CriticPolicy critic;
    RepairPolicy repair;
};

struct AgentConfig {
    std::size_t memory_top_k = 5;
    std::size_t max_attempts_per_task = 3;
    bool remember_outcomes = true;
};

struct TraceEvent {
    std::size_t sequence = 0;
    std::size_t task_id = 0;
    std::size_t attempt = 0;
    std::string phase;
    std::string detail;
};

struct AgentResult {
    bool ok = false;
    std::string output;
    std::string verification;
    Intent intent;
    TaskGraph graph;
    std::vector<memory::MemoryHit> memories;
    std::vector<TraceEvent> trace;
};

class IntentParser {
public:
    [[nodiscard]] Intent parse(
        std::string_view goal,
        const tools::ToolRegistry& registry) const;
};

class AgentEngine {
public:
    AgentEngine(
        memory::MemoryStore& memory_store,
        tools::ToolRegistry& tool_registry,
        AgentConfig config = {},
        AgentPolicies policies = {});

    [[nodiscard]] AgentResult run(std::string_view goal);

private:
    [[nodiscard]] TaskGraph plan(const AgentContext& context) const;
    [[nodiscard]] Critique critique(const AgentContext& context, const TaskNode& task) const;
    [[nodiscard]] RepairDecision repair(
        const AgentContext& context,
        const TaskNode& task,
        std::string_view reason) const;
    [[nodiscard]] bool verify(const TaskGraph& graph, std::string& reason) const;

    memory::MemoryStore& memory_store_;
    tools::ToolRegistry& tool_registry_;
    AgentConfig config_;
    AgentPolicies policies_;
    IntentParser intent_parser_;
};

} // namespace spiral::agent
