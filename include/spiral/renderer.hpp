#pragma once

#include "spiral/device.hpp"
#include "spiral/units.hpp"

#include <cstddef>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace spiral::render {

struct Rect {
    float x = 0.0F;
    float y = 0.0F;
    float width = 0.0F;
    float height = 0.0F;

    [[nodiscard]] bool contains(float px, float py) const noexcept;
    [[nodiscard]] bool intersects(const Rect& other) const noexcept;
};

struct FrameNode {
    std::string component_id;
    units::ComponentKind kind = units::ComponentKind::Container;
    Rect bounds;
    Rect clip;
    std::size_t depth = 0;
    bool interactive = false;
    bool media_surface = false;
    std::vector<FrameNode> children;
};

struct FrameTree {
    std::uint64_t revision = 0;
    Rect viewport;
    std::vector<FrameNode> roots;

    [[nodiscard]] const FrameNode* find(std::string_view component_id) const noexcept;
    [[nodiscard]] std::optional<std::string> hit_test(float x, float y) const;
};

class LayoutEngine final {
public:
    [[nodiscard]] FrameTree resolve(const units::SpiralUnit& unit, Rect viewport) const;
};

class UnitRenderer {
public:
    virtual ~UnitRenderer() = default;

    virtual void begin_frame(
        const FrameTree& tree,
        std::span<const Rect> dirty_regions,
        device::Device* device) = 0;
    virtual void draw_component(const units::Component& component, const FrameNode& frame) = 0;
    virtual void end_frame() = 0;
};

struct HostSurfaceAdapter {
    std::string host_name;
    std::function<bool(units::ComponentKind)> accepts;
    std::function<void(const units::Component&, const FrameNode&)> present;
};

struct RenderBindings {
    device::Device* device = nullptr;
    std::vector<HostSurfaceAdapter> surface_adapters;
};

struct PointerDispatchResult {
    bool hit = false;
    std::string component_id;
    units::DispatchResult dispatch;
};

class RendererBridge final {
public:
    RendererBridge(
        units::UnitRuntime& runtime,
        UnitRenderer& renderer,
        RenderBindings bindings = {});

    void mount(Rect viewport);
    void resize(Rect viewport);
    void redraw();
    void apply_patch(const units::UnitPatch& patch);
    [[nodiscard]] PointerDispatchResult dispatch_pointer(
        float x,
        float y,
        std::string_view event_name,
        std::string_view payload = {});

    [[nodiscard]] const FrameTree& frame_tree() const noexcept { return frame_tree_; }
    [[nodiscard]] Rect viewport() const noexcept { return viewport_; }
    [[nodiscard]] device::Device* device() const noexcept { return bindings_.device; }

private:
    void render_regions(std::vector<Rect> dirty_regions);
    [[nodiscard]] std::vector<Rect> diff_frames(
        const FrameTree& before,
        const FrameTree& after,
        const std::vector<std::string>& explicit_components) const;

    units::UnitRuntime& runtime_;
    UnitRenderer& renderer_;
    RenderBindings bindings_;
    LayoutEngine layout_;
    Rect viewport_;
    FrameTree frame_tree_;
    bool mounted_ = false;
};

[[nodiscard]] Rect intersect_rect(const Rect& lhs, const Rect& rhs) noexcept;
[[nodiscard]] Rect union_rect(const Rect& lhs, const Rect& rhs) noexcept;

} // namespace spiral::render
