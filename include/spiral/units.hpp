#pragma once

#include "spiral/agent.hpp"
#include "spiral/compute.hpp"
#include "spiral/tensor.hpp"
#include "spiral/tools.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace spiral::units {

enum class ComponentKind {
    Container,
    Text,
    Button,
    ImageSurface,
    AudioSurface,
    VideoSurface,
    WaveformSurface,
    Canvas,
    Grid,
    Avatar,
    Custom,
};

struct LayoutConstraints {
    float x = 0.0F;
    float y = 0.0F;
    float width = 0.0F;
    float height = 0.0F;
    float min_width = 0.0F;
    float min_height = 0.0F;
    float max_width = 0.0F;
    float max_height = 0.0F;
    float flex_grow = 0.0F;
    float gap = 0.0F;
    bool absolute = false;
};

enum class ActionKind {
    SetState,
    SetProperty,
    InvokeTool,
    InvokeAgent,
    Emit,
};

struct Action {
    ActionKind kind = ActionKind::SetState;
    std::string target;
    std::string key;
    std::string value;
};

struct Component {
    std::string id;
    ComponentKind kind = ComponentKind::Container;
    LayoutConstraints layout;
    std::map<std::string, std::string> properties;
    std::vector<std::string> children;
    std::map<std::string, std::vector<Action>> events;
};

struct SpiralUnit {
    std::string id;
    std::uint64_t revision = 0;
    std::vector<std::string> roots;
    std::vector<Component> components;
    std::map<std::string, std::string> state;

    void validate() const;
    [[nodiscard]] Component* find_component(std::string_view id) noexcept;
    [[nodiscard]] const Component* find_component(std::string_view id) const noexcept;
};

enum class PatchKind {
    UpsertComponent,
    RemoveComponent,
    SetState,
    EraseState,
    SetProperty,
    EraseProperty,
    SetRoots,
};

struct PatchOperation {
    PatchKind kind = PatchKind::SetState;
    Component component;
    std::string target;
    std::string key;
    std::string value;
    std::vector<std::string> roots;
};

struct UnitPatch {
    std::uint64_t base_revision = 0;
    std::vector<PatchOperation> operations;
};

struct ActionOutput {
    ActionKind kind = ActionKind::SetState;
    bool ok = false;
    std::string target;
    std::string output;
    std::string error;
};

struct DispatchResult {
    bool ok = false;
    std::vector<ActionOutput> actions;
    std::vector<std::string> emitted_events;
    std::string error;
};

struct RuntimeBindings {
    tools::ToolRegistry* tools = nullptr;
    agent::AgentEngine* agent = nullptr;
    compute::ComputeBackend* compute = nullptr;
};

using UnitGenerator = std::function<SpiralUnit(std::string_view prompt)>;

class UnitRuntime final {
public:
    explicit UnitRuntime(SpiralUnit unit = {}, RuntimeBindings bindings = {});

    void load(SpiralUnit unit);
    void apply_patch(const UnitPatch& patch);
    [[nodiscard]] DispatchResult dispatch(
        std::string_view component_id,
        std::string_view event_name,
        std::string_view payload = {});
    void generate_and_load(std::string_view prompt, const UnitGenerator& generator);

    [[nodiscard]] Tensor matmul(const Tensor& lhs, const Tensor& rhs) const;
    [[nodiscard]] const SpiralUnit& unit() const noexcept { return unit_; }
    [[nodiscard]] SpiralUnit& unit() noexcept { return unit_; }

private:
    [[nodiscard]] std::string resolve_value(std::string_view value, std::string_view payload) const;
    [[nodiscard]] ActionOutput execute_action(const Action& action, std::string_view payload);

    SpiralUnit unit_;
    RuntimeBindings bindings_;
};

[[nodiscard]] std::string component_kind_name(ComponentKind kind);
[[nodiscard]] bool is_media_surface(ComponentKind kind) noexcept;

} // namespace spiral::units
