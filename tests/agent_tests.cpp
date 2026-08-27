#include "spiral/agent.hpp"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

bool has_phase(const spiral::agent::AgentResult& result, const std::string& phase) {
    return std::any_of(result.trace.begin(), result.trace.end(), [&](const auto& event) {
        return event.phase == phase;
    });
}

bool has_tag(const spiral::memory::MemoryRecord& record, const std::string& tag) {
    return std::find(record.tags.begin(), record.tags.end(), tag) != record.tags.end();
}

} // namespace

int main() {
    using spiral::agent::AgentConfig;
    using spiral::agent::AgentContext;
    using spiral::agent::AgentEngine;
    using spiral::agent::AgentPolicies;
    using spiral::agent::Critique;
    using spiral::agent::IntentKind;
    using spiral::agent::RepairDecision;
    using spiral::agent::TaskGraph;
    using spiral::agent::TaskNode;
    using spiral::agent::TaskStatus;
    using spiral::memory::MemoryStore;
    using spiral::tools::ToolRegistry;
    using spiral::tools::ToolResult;

    {
        MemoryStore memory;
        memory.remember("echo tool handles hello greetings", {"tool", "echo"});
        ToolRegistry tools;
        tools.register_tool(
            {"echo", "return input unchanged"},
            [](std::string_view input) { return ToolResult::success(std::string(input)); });

        AgentConfig config;
        config.remember_outcomes = false;
        AgentEngine engine(memory, tools, config);
        const auto result = engine.run("echo hello");
        assert(result.ok);
        assert(result.intent.kind == IntentKind::ToolUse);
        assert(result.graph.tasks.size() == 1);
        assert(result.graph.tasks[0].status == TaskStatus::Succeeded);
        assert(result.graph.tasks[0].attempts == 1);
        assert(result.output == "echo hello");
        assert(!result.memories.empty());
        assert(has_phase(result, "intent"));
        assert(has_phase(result, "plan"));
        assert(has_phase(result, "verify"));
    }

    {
        MemoryStore memory;
        ToolRegistry tools;
        tools.register_tool(
            {"first", "produce first stage"},
            [](std::string_view) { return ToolResult::success("ONE"); });
        tools.register_tool(
            {"second", "wrap previous stage"},
            [](std::string_view input) { return ToolResult::success("[" + std::string(input) + "]"); });

        AgentConfig config;
        config.remember_outcomes = false;
        AgentEngine engine(memory, tools, config);
        const auto result = engine.run("first then second");
        assert(result.ok);
        assert(result.intent.kind == IntentKind::MultiStep);
        assert(result.graph.tasks.size() == 2);
        assert(result.graph.tasks[1].depends_on == std::vector<std::size_t>({1}));
        assert(result.output == "[ONE]");
    }

    {
        MemoryStore memory;
        ToolRegistry tools;
        tools.register_tool(
            {"judge", "returns good only for repaired input"},
            [](std::string_view input) {
                return ToolResult::success(input == "fixed" ? "good" : "bad");
            });

        AgentPolicies policies;
        policies.planner = [](const AgentContext&) {
            TaskGraph graph;
            TaskNode task;
            task.id = 7;
            task.description = "judge candidate";
            task.tool_name = "judge";
            task.input = "broken";
            graph.tasks.push_back(std::move(task));
            return graph;
        };
        policies.critic = [](const AgentContext&, const TaskNode& task) {
            const bool accepted = task.result.ok && task.result.output == "good";
            return Critique{accepted, accepted ? "accepted" : "output failed verification"};
        };
        policies.repair = [](const AgentContext&, const TaskNode&, std::string_view) {
            return RepairDecision{true, "judge", "fixed"};
        };

        AgentConfig config;
        config.max_attempts_per_task = 3;
        config.remember_outcomes = true;
        AgentEngine engine(memory, tools, config, policies);
        const auto result = engine.run("judge this candidate");
        assert(result.ok);
        assert(result.output == "good");
        assert(result.graph.tasks[0].attempts == 2);
        assert(has_phase(result, "critic"));
        assert(has_phase(result, "repair"));
        assert(has_phase(result, "remember"));
        assert(!memory.records().empty());
        assert(has_tag(memory.records().back(), "agent"));
        assert(has_tag(memory.records().back(), "success"));
    }

    {
        TaskGraph cyclic;
        TaskNode a;
        a.id = 1;
        a.depends_on = {2};
        TaskNode b;
        b.id = 2;
        b.depends_on = {1};
        cyclic.tasks.push_back(std::move(a));
        cyclic.tasks.push_back(std::move(b));

        bool threw = false;
        try {
            cyclic.validate();
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        assert(threw);
    }

    {
        MemoryStore memory;
        ToolRegistry tools;
        AgentConfig config;
        config.remember_outcomes = true;
        AgentEngine engine(memory, tools, config);
        const auto result = engine.run("nothing actionable here");
        assert(!result.ok);
        assert(result.verification == "no executable task was planned");
        assert(has_tag(memory.records().back(), "failure"));
    }

    std::cout << "spiral_agent_tests: PASS\n";
    return 0;
}
