#include "spiral/agent.hpp"

#include <algorithm>
#include <cctype>
#include <functional>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace spiral::agent {
namespace {

std::string lower_ascii(std::string_view text) {
    std::string out(text);
    std::transform(out.begin(), out.end(), out.begin(), [](unsigned char value) {
        return static_cast<char>(std::tolower(value));
    });
    return out;
}

TaskNode* find_task(TaskGraph& graph, std::size_t id) {
    const auto it = std::find_if(graph.tasks.begin(), graph.tasks.end(), [id](const TaskNode& task) {
        return task.id == id;
    });
    return it == graph.tasks.end() ? nullptr : &*it;
}

const TaskNode* find_task(const TaskGraph& graph, std::size_t id) {
    const auto it = std::find_if(graph.tasks.begin(), graph.tasks.end(), [id](const TaskNode& task) {
        return task.id == id;
    });
    return it == graph.tasks.end() ? nullptr : &*it;
}

std::string failure_reason(const TaskNode& task, const Critique& critique) {
    if (!task.result.error.empty()) {
        return task.result.error;
    }
    if (!critique.reason.empty()) {
        return critique.reason;
    }
    return "task result rejected";
}

} // namespace

void TaskGraph::validate() const {
    std::unordered_map<std::size_t, std::size_t> index_by_id;
    index_by_id.reserve(tasks.size());
    for (std::size_t index = 0; index < tasks.size(); ++index) {
        const auto [_, inserted] = index_by_id.emplace(tasks[index].id, index);
        if (!inserted) {
            throw std::invalid_argument("task graph contains duplicate task IDs");
        }
    }

    for (const auto& task : tasks) {
        for (const auto dependency : task.depends_on) {
            if (dependency == task.id) {
                throw std::invalid_argument("task cannot depend on itself");
            }
            if (!index_by_id.contains(dependency)) {
                throw std::invalid_argument("task graph contains unknown dependency");
            }
        }
    }

    enum class Visit { Fresh, Active, Done };
    std::vector<Visit> visits(tasks.size(), Visit::Fresh);
    std::function<void(std::size_t)> visit = [&](std::size_t index) {
        if (visits[index] == Visit::Done) return;
        if (visits[index] == Visit::Active) {
            throw std::invalid_argument("task graph contains a dependency cycle");
        }
        visits[index] = Visit::Active;
        for (const auto dependency : tasks[index].depends_on) {
            visit(index_by_id.at(dependency));
        }
        visits[index] = Visit::Done;
    };
    for (std::size_t index = 0; index < tasks.size(); ++index) {
        visit(index);
    }
}

bool TaskGraph::all_succeeded() const noexcept {
    return !tasks.empty()
        && std::all_of(tasks.begin(), tasks.end(), [](const TaskNode& task) {
            return task.status == TaskStatus::Succeeded;
        });
}

Intent IntentParser::parse(
    std::string_view goal,
    const tools::ToolRegistry& registry) const {
    Intent intent;
    intent.goal = std::string(goal);
    const std::string lowered_goal = lower_ascii(goal);
    for (const auto& tool : registry.list()) {
        const std::string lowered_name = lower_ascii(tool.name);
        if (!lowered_name.empty() && lowered_goal.find(lowered_name) != std::string::npos) {
            intent.candidate_tools.push_back(tool.name);
        }
    }

    if (intent.candidate_tools.size() == 1) {
        intent.kind = IntentKind::ToolUse;
    } else if (intent.candidate_tools.size() > 1) {
        intent.kind = IntentKind::MultiStep;
    }
    return intent;
}

AgentEngine::AgentEngine(
    memory::MemoryStore& memory_store,
    tools::ToolRegistry& tool_registry,
    AgentConfig config,
    AgentPolicies policies)
    : memory_store_(memory_store),
      tool_registry_(tool_registry),
      config_(config),
      policies_(std::move(policies)) {
    if (config_.max_attempts_per_task == 0) {
        throw std::invalid_argument("agent max_attempts_per_task must be > 0");
    }
}

TaskGraph AgentEngine::plan(const AgentContext& context) const {
    if (policies_.planner) {
        return policies_.planner(context);
    }

    TaskGraph graph;
    graph.tasks.reserve(context.intent.candidate_tools.size());
    for (std::size_t index = 0; index < context.intent.candidate_tools.size(); ++index) {
        TaskNode task;
        task.id = index + 1;
        task.description = "invoke " + context.intent.candidate_tools[index];
        task.tool_name = context.intent.candidate_tools[index];
        task.input = index == 0 ? context.goal : "$previous";
        if (index > 0) {
            task.depends_on.push_back(index);
        }
        graph.tasks.push_back(std::move(task));
    }
    return graph;
}

Critique AgentEngine::critique(const AgentContext& context, const TaskNode& task) const {
    if (policies_.critic) {
        return policies_.critic(context, task);
    }
    if (!task.result.ok) {
        return Critique{false, task.result.error.empty() ? "tool execution failed" : task.result.error};
    }
    return Critique{true, "tool result accepted"};
}

RepairDecision AgentEngine::repair(
    const AgentContext& context,
    const TaskNode& task,
    std::string_view reason) const {
    if (policies_.repair) {
        return policies_.repair(context, task, reason);
    }
    return RepairDecision{true, task.tool_name, task.input};
}

bool AgentEngine::verify(const TaskGraph& graph, std::string& reason) const {
    if (graph.tasks.empty()) {
        reason = "no executable task was planned";
        return false;
    }
    for (const auto& task : graph.tasks) {
        if (task.status != TaskStatus::Succeeded) {
            reason = "task " + std::to_string(task.id) + " did not succeed";
            return false;
        }
    }
    reason = "all planned tasks succeeded";
    return true;
}

AgentResult AgentEngine::run(std::string_view goal) {
    if (goal.empty()) {
        throw std::invalid_argument("agent goal must not be empty");
    }

    AgentResult result;
    std::size_t trace_sequence = 0;
    auto trace = [&](std::size_t task_id, std::size_t attempt, std::string phase, std::string detail) {
        result.trace.push_back(TraceEvent{
            ++trace_sequence,
            task_id,
            attempt,
            std::move(phase),
            std::move(detail)});
    };

    result.intent = intent_parser_.parse(goal, tool_registry_);
    trace(0, 0, "intent", "candidate tools: " + std::to_string(result.intent.candidate_tools.size()));

    if (config_.memory_top_k > 0) {
        result.memories = memory_store_.search(goal, config_.memory_top_k);
    }
    trace(0, 0, "memory", "retrieved memories: " + std::to_string(result.memories.size()));

    AgentContext context{
        std::string(goal),
        result.intent,
        result.memories,
        tool_registry_.list()};

    result.graph = plan(context);
    result.graph.validate();
    trace(0, 0, "plan", "planned tasks: " + std::to_string(result.graph.tasks.size()));

    std::size_t completed_or_blocked = 0;
    while (completed_or_blocked < result.graph.tasks.size()) {
        bool progressed = false;

        for (auto& task : result.graph.tasks) {
            if (task.status != TaskStatus::Pending) {
                continue;
            }

            bool dependency_failed = false;
            bool dependencies_ready = true;
            for (const auto dependency_id : task.depends_on) {
                const TaskNode* dependency = find_task(result.graph, dependency_id);
                if (dependency == nullptr) {
                    throw std::logic_error("validated task dependency disappeared");
                }
                if (dependency->status == TaskStatus::Failed || dependency->status == TaskStatus::Skipped) {
                    dependency_failed = true;
                    break;
                }
                if (dependency->status != TaskStatus::Succeeded) {
                    dependencies_ready = false;
                }
            }

            if (dependency_failed) {
                task.status = TaskStatus::Skipped;
                task.result = tools::ToolResult::failure("dependency failed");
                ++completed_or_blocked;
                progressed = true;
                trace(task.id, 0, "skip", "dependency failed");
                continue;
            }
            if (!dependencies_ready) {
                continue;
            }

            task.status = TaskStatus::Running;
            std::string current_tool = task.tool_name;
            std::string current_input = task.input;
            if (current_input == "$previous" && !task.depends_on.empty()) {
                const TaskNode* dependency = find_task(result.graph, task.depends_on.back());
                current_input = dependency == nullptr ? std::string{} : dependency->result.output;
            }

            for (std::size_t attempt = 1; attempt <= config_.max_attempts_per_task; ++attempt) {
                task.attempts = attempt;
                task.tool_name = current_tool;
                task.input = current_input;
                trace(task.id, attempt, "execute", current_tool);
                task.result = tool_registry_.invoke(current_tool, current_input);

                const Critique review = critique(context, task);
                trace(task.id, attempt, "critic", review.reason);
                if (task.result.ok && review.acceptable) {
                    task.status = TaskStatus::Succeeded;
                    break;
                }

                const std::string reason = failure_reason(task, review);
                if (attempt == config_.max_attempts_per_task) {
                    task.status = TaskStatus::Failed;
                    if (task.result.error.empty()) {
                        task.result.error = reason;
                    }
                    break;
                }

                const RepairDecision decision = repair(context, task, reason);
                trace(task.id, attempt, "repair", decision.retry ? "retry" : "stop");
                if (!decision.retry) {
                    task.status = TaskStatus::Failed;
                    if (task.result.error.empty()) {
                        task.result.error = reason;
                    }
                    break;
                }
                if (!decision.tool_name.empty()) {
                    current_tool = decision.tool_name;
                }
                current_input = decision.input;
            }

            if (task.status == TaskStatus::Running) {
                task.status = TaskStatus::Failed;
                task.result = tools::ToolResult::failure("task exhausted without terminal state");
            }
            ++completed_or_blocked;
            progressed = true;
        }

        if (!progressed) {
            for (auto& task : result.graph.tasks) {
                if (task.status == TaskStatus::Pending) {
                    task.status = TaskStatus::Skipped;
                    task.result = tools::ToolResult::failure("unresolved dependencies");
                    ++completed_or_blocked;
                    trace(task.id, 0, "skip", "unresolved dependencies");
                }
            }
        }
    }

    result.ok = verify(result.graph, result.verification);
    trace(0, 0, "verify", result.verification);

    if (result.ok) {
        for (auto it = result.graph.tasks.rbegin(); it != result.graph.tasks.rend(); ++it) {
            if (it->status == TaskStatus::Succeeded) {
                result.output = it->result.output;
                break;
            }
        }
    } else {
        result.output = result.verification;
    }

    if (config_.remember_outcomes) {
        const std::string state = result.ok ? "success" : "failure";
        memory_store_.remember(
            "agent " + state + " goal: " + std::string(goal) + "\nresult: " + result.output,
            {"agent", "outcome", state});
        trace(0, 0, "remember", state);
    }

    return result;
}

} // namespace spiral::agent
