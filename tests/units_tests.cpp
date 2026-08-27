#include "spiral/agent.hpp"
#include "spiral/compute.hpp"
#include "spiral/memory.hpp"
#include "spiral/precision.hpp"
#include "spiral/tensor.hpp"
#include "spiral/tools.hpp"
#include "spiral/units.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

class CountingBackend final : public spiral::compute::ComputeBackend {
public:
    [[nodiscard]] std::string_view name() const noexcept override { return "counting-test"; }
    [[nodiscard]] std::size_t worker_count() const noexcept override { return 1; }

    spiral::Tensor matmul(const spiral::Tensor& lhs, const spiral::Tensor& rhs) override {
        ++fp32_calls;
        return lhs.matmul(rhs);
    }

    spiral::Tensor matmul_int8(
        const spiral::precision::QuantizedTensor& lhs,
        const spiral::precision::QuantizedTensor& rhs) override {
        ++int8_calls;
        return lhs.dequantize().matmul(rhs.dequantize());
    }

    std::size_t fp32_calls = 0;
    std::size_t int8_calls = 0;
};

spiral::units::SpiralUnit make_player_unit() {
    using namespace spiral::units;

    Component root;
    root.id = "root";
    root.kind = ComponentKind::Container;
    root.layout.width = 640.0F;
    root.layout.height = 360.0F;
    root.children = {"status", "play", "audio", "video", "wave"};

    Component status;
    status.id = "status";
    status.kind = ComponentKind::Text;
    status.properties["text"] = "idle";

    Component play;
    play.id = "play";
    play.kind = ComponentKind::Button;
    play.properties["label"] = "Play";
    play.events["tap"] = {
        Action{ActionKind::SetState, "track", "", "$payload"},
        Action{ActionKind::SetProperty, "status", "text", "$state.track"},
        Action{ActionKind::InvokeTool, "echo", "", "$state.track"},
        Action{ActionKind::Emit, "track.changed", "", ""},
    };
    play.events["agent"] = {
        Action{ActionKind::InvokeAgent, "", "", "$payload"},
    };

    Component audio;
    audio.id = "audio";
    audio.kind = ComponentKind::AudioSurface;
    audio.properties["source"] = "ether://track/current";

    Component video;
    video.id = "video";
    video.kind = ComponentKind::VideoSurface;
    video.properties["source"] = "spiral://visual/current";

    Component wave;
    wave.id = "wave";
    wave.kind = ComponentKind::WaveformSurface;
    wave.properties["bind"] = "audio";

    SpiralUnit unit;
    unit.id = "ether-player-live";
    unit.roots = {"root"};
    unit.components = {root, status, play, audio, video, wave};
    unit.state["track"] = "none";
    return unit;
}

bool tensors_close(const spiral::Tensor& lhs, const spiral::Tensor& rhs, float tolerance = 1.0e-6F) {
    if (lhs.shape() != rhs.shape()) return false;
    for (std::size_t i = 0; i < lhs.numel(); ++i) {
        if (std::abs(lhs.data()[i] - rhs.data()[i]) > tolerance) return false;
    }
    return true;
}

} // namespace

int main() {
    using namespace spiral;
    using namespace spiral::units;

    tools::ToolRegistry tools;
    tools.register_tool({"echo", "echo input"}, [](std::string_view input) {
        return tools::ToolResult::success("echo:" + std::string(input));
    });

    memory::MemoryStore memory;
    agent::AgentPolicies policies;
    policies.planner = [](const agent::AgentContext& context) {
        agent::TaskGraph graph;
        agent::TaskNode task;
        task.id = 1;
        task.description = "echo the unit goal";
        task.tool_name = "echo";
        task.input = context.goal;
        graph.tasks.push_back(std::move(task));
        return graph;
    };
    policies.critic = [](const agent::AgentContext&, const agent::TaskNode& task) {
        return agent::Critique{task.result.ok, task.result.ok ? "accepted" : "tool failed"};
    };
    agent::AgentEngine agent_engine(memory, tools, {0, 1, false}, policies);

    CountingBackend compute;
    SpiralUnit unit = make_player_unit();
    unit.validate();
    assert(unit.revision == 0);
    assert(is_media_surface(unit.find_component("audio")->kind));
    assert(is_media_surface(unit.find_component("video")->kind));
    assert(component_kind_name(unit.find_component("wave")->kind) == "waveform");

    UnitRuntime runtime(unit, RuntimeBindings{&tools, &agent_engine, &compute});
    const auto tap = runtime.dispatch("play", "tap", "night drive");
    assert(tap.ok);
    assert(tap.actions.size() == 4);
    assert(tap.actions[2].output == "echo:night drive");
    assert(tap.emitted_events.size() == 1 && tap.emitted_events[0] == "track.changed");
    assert(runtime.unit().state.at("track") == "night drive");
    assert(runtime.unit().find_component("status")->properties.at("text") == "night drive");
    assert(runtime.unit().revision == 0);

    const auto agent_result = runtime.dispatch("play", "agent", "inspect current mix");
    assert(agent_result.ok);
    assert(agent_result.actions.size() == 1);
    assert(agent_result.actions[0].output == "echo:inspect current mix");

    Tensor lhs({2, 2}, {1.0F, 2.0F, 3.0F, 4.0F});
    Tensor rhs({2, 2}, {5.0F, 6.0F, 7.0F, 8.0F});
    const Tensor expected = lhs.matmul(rhs);
    const Tensor routed = runtime.matmul(lhs, rhs);
    assert(compute.fp32_calls == 1);
    assert(tensors_close(expected, routed));

    UnitPatch patch;
    patch.base_revision = 0;
    patch.operations.push_back(PatchOperation{PatchKind::SetProperty, {}, "status", "text", "hot patched", {}});
    patch.operations.push_back(PatchOperation{PatchKind::SetState, {}, "", "mode", "visualizer", {}});
    runtime.apply_patch(patch);
    assert(runtime.unit().revision == 1);
    assert(runtime.unit().find_component("status")->properties.at("text") == "hot patched");
    assert(runtime.unit().state.at("mode") == "visualizer");

    bool stale_rejected = false;
    try {
        runtime.apply_patch(patch);
    } catch (const std::invalid_argument&) {
        stale_rejected = true;
    }
    assert(stale_rejected);
    assert(runtime.unit().revision == 1);

    UnitPatch corrupt;
    corrupt.base_revision = 1;
    corrupt.operations.push_back(PatchOperation{PatchKind::RemoveComponent, {}, "status", "", "", {}});
    bool corrupt_rejected = false;
    try {
        runtime.apply_patch(corrupt);
    } catch (const std::invalid_argument&) {
        corrupt_rejected = true;
    }
    assert(corrupt_rejected);
    assert(runtime.unit().revision == 1);
    assert(runtime.unit().find_component("status") != nullptr);

    std::string captured_prompt;
    runtime.generate_and_load("make a compact media console", [&](std::string_view prompt) {
        captured_prompt = std::string(prompt);
        Component root;
        root.id = "generated-root";
        root.kind = ComponentKind::Grid;
        root.children = {"generated-text"};
        Component text;
        text.id = "generated-text";
        text.kind = ComponentKind::Text;
        text.properties["text"] = std::string(prompt);
        SpiralUnit generated;
        generated.id = "generated-unit";
        generated.roots = {root.id};
        generated.components = {root, text};
        return generated;
    });
    assert(captured_prompt == "make a compact media console");
    assert(runtime.unit().id == "generated-unit");
    assert(runtime.unit().find_component("generated-text")->properties.at("text") == captured_prompt);

    SpiralUnit cyclic;
    cyclic.id = "cycle";
    Component loop;
    loop.id = "loop";
    loop.children = {"loop"};
    cyclic.roots = {"loop"};
    cyclic.components = {loop};
    bool cycle_rejected = false;
    try {
        cyclic.validate();
    } catch (const std::invalid_argument&) {
        cycle_rejected = true;
    }
    assert(cycle_rejected);

    std::cout << "L19 live state: " << runtime.unit().id << '\n';
    std::cout << "L19 compute backend calls: " << compute.fp32_calls << '\n';
    std::cout << "L19 Spiral Units runtime tests passed\n";
    return 0;
}
