#include "spiral/renderer.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <map>
#include <set>
#include <stdexcept>
#include <utility>

namespace spiral::render {
namespace {

constexpr float kRectEpsilon = 1.0e-5F;

bool finite_rect(const Rect& rect) noexcept {
    return std::isfinite(rect.x) && std::isfinite(rect.y) &&
        std::isfinite(rect.width) && std::isfinite(rect.height);
}

bool valid_viewport(const Rect& rect) noexcept {
    return finite_rect(rect) && rect.width > 0.0F && rect.height > 0.0F;
}

float resolve_dimension(float requested, float minimum, float maximum, float fallback) {
    float value = requested > 0.0F ? requested : fallback;
    value = std::max(value, minimum);
    if (maximum > 0.0F) value = std::min(value, maximum);
    return std::max(0.0F, value);
}

bool rect_close(const Rect& lhs, const Rect& rhs) noexcept {
    return std::abs(lhs.x - rhs.x) <= kRectEpsilon &&
        std::abs(lhs.y - rhs.y) <= kRectEpsilon &&
        std::abs(lhs.width - rhs.width) <= kRectEpsilon &&
        std::abs(lhs.height - rhs.height) <= kRectEpsilon;
}

std::size_t grid_columns(const units::Component& component) {
    const auto it = component.properties.find("columns");
    if (it == component.properties.end()) return 1;
    std::size_t value = 0;
    const auto* begin = it->second.data();
    const auto* end = begin + it->second.size();
    const auto result = std::from_chars(begin, end, value);
    if (result.ec != std::errc{} || result.ptr != end || value == 0) return 1;
    return value;
}

const FrameNode* find_node(const std::vector<FrameNode>& nodes, std::string_view id) noexcept {
    for (const auto& node : nodes) {
        if (node.component_id == id) return &node;
        if (const auto* nested = find_node(node.children, id); nested != nullptr) return nested;
    }
    return nullptr;
}

std::optional<std::string> hit_nodes(const std::vector<FrameNode>& nodes, float x, float y) {
    for (auto it = nodes.rbegin(); it != nodes.rend(); ++it) {
        const auto& node = *it;
        if (!node.clip.contains(x, y) || !node.bounds.contains(x, y)) continue;
        if (auto nested = hit_nodes(node.children, x, y); nested.has_value()) return nested;
        if (node.interactive) return node.component_id;
    }
    return std::nullopt;
}

void flatten_frames(const std::vector<FrameNode>& nodes, std::map<std::string, Rect>& out) {
    for (const auto& node : nodes) {
        out[node.component_id] = node.bounds;
        flatten_frames(node.children, out);
    }
}

void append_unique_rect(std::vector<Rect>& out, Rect rect) {
    if (rect.width <= 0.0F || rect.height <= 0.0F) return;
    for (const auto& existing : out) {
        if (rect_close(existing, rect)) return;
    }
    out.push_back(rect);
}

bool node_is_dirty(const FrameNode& node, std::span<const Rect> dirty) {
    for (const auto& region : dirty) {
        if (node.clip.intersects(region) && node.bounds.intersects(region)) return true;
    }
    return false;
}

void render_nodes(
    const std::vector<FrameNode>& nodes,
    const units::SpiralUnit& unit,
    UnitRenderer& renderer,
    const RenderBindings& bindings,
    std::span<const Rect> dirty) {
    for (const auto& node : nodes) {
        if (!node_is_dirty(node, dirty)) continue;
        const auto* component = unit.find_component(node.component_id);
        if (component == nullptr) throw std::logic_error("frame tree references missing unit component");
        renderer.draw_component(*component, node);
        if (node.media_surface) {
            for (const auto& adapter : bindings.surface_adapters) {
                if (adapter.accepts && adapter.present && adapter.accepts(node.kind)) {
                    adapter.present(*component, node);
                }
            }
        }
        render_nodes(node.children, unit, renderer, bindings, dirty);
    }
}

FrameNode build_node(
    const units::SpiralUnit& unit,
    const units::Component& component,
    Rect bounds,
    Rect parent_clip,
    std::size_t depth) {
    FrameNode node;
    node.component_id = component.id;
    node.kind = component.kind;
    node.bounds = bounds;
    node.clip = intersect_rect(parent_clip, bounds);
    node.depth = depth;
    node.interactive = !component.events.empty() || component.kind == units::ComponentKind::Button;
    node.media_surface = units::is_media_surface(component.kind);

    if (component.children.empty()) return node;

    const float gap = component.layout.gap;
    std::vector<const units::Component*> flow_children;
    flow_children.reserve(component.children.size());
    for (const auto& child_id : component.children) {
        const auto* child = unit.find_component(child_id);
        if (child == nullptr) throw std::logic_error("validated unit child disappeared");
        if (!child->layout.absolute) flow_children.push_back(child);
    }

    std::map<std::string, Rect> child_bounds;
    if (component.kind == units::ComponentKind::Grid && !flow_children.empty()) {
        const std::size_t columns = std::min(grid_columns(component), flow_children.size());
        const std::size_t rows = (flow_children.size() + columns - 1) / columns;
        const float total_x_gap = gap * static_cast<float>(columns > 0 ? columns - 1 : 0);
        const float total_y_gap = gap * static_cast<float>(rows > 0 ? rows - 1 : 0);
        const float cell_width = std::max(0.0F, (bounds.width - total_x_gap) / static_cast<float>(columns));
        const float cell_height = std::max(0.0F, (bounds.height - total_y_gap) / static_cast<float>(rows));
        for (std::size_t i = 0; i < flow_children.size(); ++i) {
            const auto& child = *flow_children[i];
            const std::size_t column = i % columns;
            const std::size_t row = i / columns;
            const float width = resolve_dimension(
                child.layout.width, child.layout.min_width, child.layout.max_width, cell_width);
            const float height = resolve_dimension(
                child.layout.height, child.layout.min_height, child.layout.max_height, cell_height);
            child_bounds[child.id] = Rect{
                bounds.x + static_cast<float>(column) * (cell_width + gap) + child.layout.x,
                bounds.y + static_cast<float>(row) * (cell_height + gap) + child.layout.y,
                width,
                height};
        }
    } else if (!flow_children.empty()) {
        float fixed_height = 0.0F;
        float flexible_weight = 0.0F;
        for (const auto* child : flow_children) {
            if (child->layout.height > 0.0F) {
                fixed_height += resolve_dimension(
                    child->layout.height,
                    child->layout.min_height,
                    child->layout.max_height,
                    child->layout.height);
            } else {
                flexible_weight += child->layout.flex_grow > 0.0F ? child->layout.flex_grow : 1.0F;
            }
        }
        const float total_gap = gap * static_cast<float>(flow_children.size() > 0 ? flow_children.size() - 1 : 0);
        const float flexible_space = std::max(0.0F, bounds.height - total_gap - fixed_height);
        float cursor_y = bounds.y;
        for (const auto* child : flow_children) {
            float fallback_height = child->layout.height;
            if (child->layout.height <= 0.0F) {
                const float weight = child->layout.flex_grow > 0.0F ? child->layout.flex_grow : 1.0F;
                fallback_height = flexible_weight > 0.0F ? flexible_space * weight / flexible_weight : 0.0F;
            }
            const float height = resolve_dimension(
                child->layout.height,
                child->layout.min_height,
                child->layout.max_height,
                fallback_height);
            const float width = resolve_dimension(
                child->layout.width,
                child->layout.min_width,
                child->layout.max_width,
                bounds.width);
            child_bounds[child->id] = Rect{
                bounds.x + child->layout.x,
                cursor_y + child->layout.y,
                width,
                height};
            cursor_y += height + gap;
        }
    }

    for (const auto& child_id : component.children) {
        const auto* child = unit.find_component(child_id);
        if (child == nullptr) throw std::logic_error("validated unit child disappeared");
        Rect resolved;
        if (child->layout.absolute) {
            resolved = Rect{
                bounds.x + child->layout.x,
                bounds.y + child->layout.y,
                resolve_dimension(child->layout.width, child->layout.min_width, child->layout.max_width, bounds.width),
                resolve_dimension(child->layout.height, child->layout.min_height, child->layout.max_height, bounds.height)};
        } else {
            resolved = child_bounds.at(child->id);
        }
        node.children.push_back(build_node(unit, *child, resolved, node.clip, depth + 1));
    }
    return node;
}

std::vector<std::string> patch_components(const units::UnitPatch& patch, bool& full_redraw) {
    std::vector<std::string> ids;
    for (const auto& operation : patch.operations) {
        switch (operation.kind) {
            case units::PatchKind::UpsertComponent:
                if (!operation.component.id.empty()) ids.push_back(operation.component.id);
                break;
            case units::PatchKind::RemoveComponent:
            case units::PatchKind::SetProperty:
            case units::PatchKind::EraseProperty:
                if (!operation.target.empty()) ids.push_back(operation.target);
                break;
            case units::PatchKind::SetState:
            case units::PatchKind::EraseState:
                full_redraw = true;
                break;
            case units::PatchKind::SetRoots:
                full_redraw = true;
                break;
        }
    }
    return ids;
}

} // namespace

bool Rect::contains(float px, float py) const noexcept {
    return width > 0.0F && height > 0.0F &&
        px >= x && py >= y && px < x + width && py < y + height;
}

bool Rect::intersects(const Rect& other) const noexcept {
    if (width <= 0.0F || height <= 0.0F || other.width <= 0.0F || other.height <= 0.0F) return false;
    return x < other.x + other.width && x + width > other.x &&
        y < other.y + other.height && y + height > other.y;
}

Rect intersect_rect(const Rect& lhs, const Rect& rhs) noexcept {
    const float left = std::max(lhs.x, rhs.x);
    const float top = std::max(lhs.y, rhs.y);
    const float right = std::min(lhs.x + lhs.width, rhs.x + rhs.width);
    const float bottom = std::min(lhs.y + lhs.height, rhs.y + rhs.height);
    if (right <= left || bottom <= top) return Rect{left, top, 0.0F, 0.0F};
    return Rect{left, top, right - left, bottom - top};
}

Rect union_rect(const Rect& lhs, const Rect& rhs) noexcept {
    if (lhs.width <= 0.0F || lhs.height <= 0.0F) return rhs;
    if (rhs.width <= 0.0F || rhs.height <= 0.0F) return lhs;
    const float left = std::min(lhs.x, rhs.x);
    const float top = std::min(lhs.y, rhs.y);
    const float right = std::max(lhs.x + lhs.width, rhs.x + rhs.width);
    const float bottom = std::max(lhs.y + lhs.height, rhs.y + rhs.height);
    return Rect{left, top, right - left, bottom - top};
}

const FrameNode* FrameTree::find(std::string_view component_id) const noexcept {
    return find_node(roots, component_id);
}

std::optional<std::string> FrameTree::hit_test(float x, float y) const {
    if (!viewport.contains(x, y)) return std::nullopt;
    return hit_nodes(roots, x, y);
}

FrameTree LayoutEngine::resolve(const units::SpiralUnit& unit, Rect viewport) const {
    unit.validate();
    if (!valid_viewport(viewport)) throw std::invalid_argument("renderer viewport must be finite and positive");

    FrameTree tree;
    tree.revision = unit.revision;
    tree.viewport = viewport;
    const float slot_height = viewport.height / static_cast<float>(unit.roots.size());
    for (std::size_t i = 0; i < unit.roots.size(); ++i) {
        const auto* root = unit.find_component(unit.roots[i]);
        if (root == nullptr) throw std::logic_error("validated unit root disappeared");
        const float width = resolve_dimension(root->layout.width, root->layout.min_width, root->layout.max_width, viewport.width);
        const float height = resolve_dimension(root->layout.height, root->layout.min_height, root->layout.max_height, slot_height);
        Rect bounds{
            viewport.x + root->layout.x,
            viewport.y + static_cast<float>(i) * slot_height + root->layout.y,
            width,
            height};
        tree.roots.push_back(build_node(unit, *root, bounds, viewport, 0));
    }
    return tree;
}

RendererBridge::RendererBridge(
    units::UnitRuntime& runtime,
    UnitRenderer& renderer,
    RenderBindings bindings)
    : runtime_(runtime), renderer_(renderer), bindings_(std::move(bindings)) {
    for (const auto& adapter : bindings_.surface_adapters) {
        if (adapter.host_name.empty()) throw std::invalid_argument("surface adapter host name must not be empty");
        if (!adapter.accepts || !adapter.present) throw std::invalid_argument("surface adapter callbacks must be callable");
    }
}

void RendererBridge::mount(Rect viewport) {
    if (!valid_viewport(viewport)) throw std::invalid_argument("renderer viewport must be finite and positive");
    viewport_ = viewport;
    frame_tree_ = layout_.resolve(runtime_.unit(), viewport_);
    mounted_ = true;
    render_regions({viewport_});
}

void RendererBridge::resize(Rect viewport) {
    if (!mounted_) throw std::logic_error("renderer bridge must be mounted before resize");
    if (!valid_viewport(viewport)) throw std::invalid_argument("renderer viewport must be finite and positive");
    const FrameTree before = frame_tree_;
    const Rect old_viewport = viewport_;
    viewport_ = viewport;
    frame_tree_ = layout_.resolve(runtime_.unit(), viewport_);
    auto dirty = diff_frames(before, frame_tree_, {});
    append_unique_rect(dirty, old_viewport);
    append_unique_rect(dirty, viewport_);
    render_regions(std::move(dirty));
}

void RendererBridge::redraw() {
    if (!mounted_) throw std::logic_error("renderer bridge must be mounted before redraw");
    frame_tree_ = layout_.resolve(runtime_.unit(), viewport_);
    render_regions({viewport_});
}

void RendererBridge::apply_patch(const units::UnitPatch& patch) {
    if (!mounted_) throw std::logic_error("renderer bridge must be mounted before patching");
    const FrameTree before = frame_tree_;
    bool full_redraw = false;
    const auto explicit_components = patch_components(patch, full_redraw);
    runtime_.apply_patch(patch);
    frame_tree_ = layout_.resolve(runtime_.unit(), viewport_);
    auto dirty = diff_frames(before, frame_tree_, explicit_components);
    if (full_redraw) append_unique_rect(dirty, viewport_);
    render_regions(std::move(dirty));
}

PointerDispatchResult RendererBridge::dispatch_pointer(
    float x,
    float y,
    std::string_view event_name,
    std::string_view payload) {
    if (!mounted_) throw std::logic_error("renderer bridge must be mounted before pointer dispatch");
    PointerDispatchResult result;
    const auto hit = frame_tree_.hit_test(x, y);
    if (!hit.has_value()) return result;
    result.hit = true;
    result.component_id = *hit;

    const FrameTree before = frame_tree_;
    result.dispatch = runtime_.dispatch(*hit, event_name, payload);
    frame_tree_ = layout_.resolve(runtime_.unit(), viewport_);

    bool full_redraw = false;
    std::vector<std::string> explicit_components;
    for (const auto& action : result.dispatch.actions) {
        if (!action.ok) continue;
        if (action.kind == units::ActionKind::SetProperty && !action.target.empty()) {
            explicit_components.push_back(action.target);
        } else if (action.kind == units::ActionKind::SetState) {
            full_redraw = true;
        }
    }
    auto dirty = diff_frames(before, frame_tree_, explicit_components);
    if (full_redraw) append_unique_rect(dirty, viewport_);
    render_regions(std::move(dirty));
    return result;
}

std::vector<Rect> RendererBridge::diff_frames(
    const FrameTree& before,
    const FrameTree& after,
    const std::vector<std::string>& explicit_components) const {
    std::map<std::string, Rect> old_frames;
    std::map<std::string, Rect> new_frames;
    flatten_frames(before.roots, old_frames);
    flatten_frames(after.roots, new_frames);
    std::set<std::string> ids;
    for (const auto& [id, rect] : old_frames) {
        (void)rect;
        ids.insert(id);
    }
    for (const auto& [id, rect] : new_frames) {
        (void)rect;
        ids.insert(id);
    }
    for (const auto& id : explicit_components) ids.insert(id);

    std::vector<Rect> dirty;
    for (const auto& id : ids) {
        const auto old_it = old_frames.find(id);
        const auto new_it = new_frames.find(id);
        const bool explicit_change = std::find(explicit_components.begin(), explicit_components.end(), id) != explicit_components.end();
        if (old_it == old_frames.end()) {
            append_unique_rect(dirty, new_it->second);
        } else if (new_it == new_frames.end()) {
            append_unique_rect(dirty, old_it->second);
        } else if (explicit_change || !rect_close(old_it->second, new_it->second)) {
            append_unique_rect(dirty, union_rect(old_it->second, new_it->second));
        }
    }
    return dirty;
}

void RendererBridge::render_regions(std::vector<Rect> dirty_regions) {
    if (dirty_regions.empty()) return;
    renderer_.begin_frame(frame_tree_, dirty_regions, bindings_.device);
    render_nodes(frame_tree_.roots, runtime_.unit(), renderer_, bindings_, dirty_regions);
    renderer_.end_frame();
}

} // namespace spiral::render
