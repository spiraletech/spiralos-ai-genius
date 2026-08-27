#include "spiral/units.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <set>
#include <stdexcept>
#include <utility>

namespace spiral::units {
namespace {

bool finite_layout_value(float value) noexcept {
    return std::isfinite(value);
}

void validate_layout(const LayoutConstraints& layout) {
    const float values[] = {
        layout.x, layout.y, layout.width, layout.height,
        layout.min_width, layout.min_height, layout.max_width, layout.max_height,
        layout.flex_grow, layout.gap,
    };
    for (float value : values) {
        if (!finite_layout_value(value)) throw std::invalid_argument("unit layout contains a non-finite value");
    }
    if (layout.width < 0.0F || layout.height < 0.0F || layout.min_width < 0.0F ||
        layout.min_height < 0.0F || layout.max_width < 0.0F || layout.max_height < 0.0F ||
        layout.flex_grow < 0.0F || layout.gap < 0.0F) {
        throw std::invalid_argument("unit layout dimensions must be non-negative");
    }
    if (layout.max_width > 0.0F && layout.min_width > layout.max_width) {
        throw std::invalid_argument("unit layout min_width exceeds max_width");
    }
    if (layout.max_height > 0.0F && layout.min_height > layout.max_height) {
        throw std::invalid_argument("unit layout min_height exceeds max_height");
    }
}

void erase_component(std::vector<Component>& components, std::string_view id) {
    components.erase(
        std::remove_if(components.begin(), components.end(), [id](const Component& component) {
            return component.id == id;
        }),
        components.end());
}

} // namespace

Component* SpiralUnit::find_component(std::string_view id) noexcept {
    auto it = std::find_if(components.begin(), components.end(), [id](const Component& component) {
        return component.id == id;
    });
    return it == components.end() ? nullptr : &*it;
}

const Component* SpiralUnit::find_component(std::string_view id) const noexcept {
    auto it = std::find_if(components.begin(), components.end(), [id](const Component& component) {
        return component.id == id;
    });
    return it == components.end() ? nullptr : &*it;
}

void SpiralUnit::validate() const {
    if (id.empty()) throw std::invalid_argument("SpiralUnit id must not be empty");
    if (components.empty()) throw std::invalid_argument("SpiralUnit must contain at least one component");
    if (roots.empty()) throw std::invalid_argument("SpiralUnit must contain at least one root component");

    std::set<std::string> ids;
    for (const auto& component : components) {
        if (component.id.empty()) throw std::invalid_argument("unit component id must not be empty");
        if (!ids.insert(component.id).second) throw std::invalid_argument("duplicate unit component id: " + component.id);
        validate_layout(component.layout);
        for (const auto& [event_name, actions] : component.events) {
            if (event_name.empty()) throw std::invalid_argument("unit event name must not be empty");
            for (const auto& action : actions) {
                if (action.kind == ActionKind::SetState && action.target.empty()) {
                    throw std::invalid_argument("SetState action requires a state key target");
                }
                if (action.kind == ActionKind::SetProperty && (action.target.empty() || action.key.empty())) {
                    throw std::invalid_argument("SetProperty action requires component target and property key");
                }
                if ((action.kind == ActionKind::InvokeTool || action.kind == ActionKind::Emit) && action.target.empty()) {
                    throw std::invalid_argument("tool/emit action requires a target");
                }
            }
        }
    }

    std::set<std::string> root_ids;
    for (const auto& root : roots) {
        if (!ids.contains(root)) throw std::invalid_argument("unit root references missing component: " + root);
        if (!root_ids.insert(root).second) throw std::invalid_argument("duplicate unit root: " + root);
    }

    std::map<std::string, int> colors;
    for (const auto& component : components) colors[component.id] = 0;
    std::function<void(const Component&)> visit = [&](const Component& component) {
        int& color = colors[component.id];
        if (color == 1) throw std::invalid_argument("unit component graph contains a cycle at: " + component.id);
        if (color == 2) return;
        color = 1;
        std::set<std::string> local_children;
        for (const auto& child_id : component.children) {
            if (!local_children.insert(child_id).second) {
                throw std::invalid_argument("duplicate child reference from component: " + component.id);
            }
            const auto* child = find_component(child_id);
            if (child == nullptr) throw std::invalid_argument("unit child references missing component: " + child_id);
            visit(*child);
        }
        color = 2;
    };
    for (const auto& root : roots) visit(*find_component(root));

    for (const auto& component : components) {
        if (colors[component.id] == 0) {
            throw std::invalid_argument("unit component is unreachable from roots: " + component.id);
        }
    }
}

UnitRuntime::UnitRuntime(SpiralUnit unit, RuntimeBindings bindings)
    : unit_(std::move(unit)), bindings_(bindings) {
    if (!unit_.id.empty()) unit_.validate();
}

void UnitRuntime::load(SpiralUnit unit) {
    unit.validate();
    unit_ = std::move(unit);
}

void UnitRuntime::apply_patch(const UnitPatch& patch) {
    if (unit_.id.empty()) throw std::logic_error("cannot patch an empty UnitRuntime");
    if (patch.base_revision != unit_.revision) throw std::invalid_argument("unit patch base revision mismatch");
    if (unit_.revision == std::numeric_limits<std::uint64_t>::max()) throw std::overflow_error("unit revision overflow");

    SpiralUnit candidate = unit_;
    for (const auto& operation : patch.operations) {
        switch (operation.kind) {
            case PatchKind::UpsertComponent: {
                if (operation.component.id.empty()) throw std::invalid_argument("upsert requires component id");
                if (auto* existing = candidate.find_component(operation.component.id); existing != nullptr) {
                    *existing = operation.component;
                } else {
                    candidate.components.push_back(operation.component);
                }
                break;
            }
            case PatchKind::RemoveComponent:
                if (operation.target.empty()) throw std::invalid_argument("remove component requires target id");
                erase_component(candidate.components, operation.target);
                break;
            case PatchKind::SetState:
                if (operation.key.empty()) throw std::invalid_argument("set state requires key");
                candidate.state[operation.key] = operation.value;
                break;
            case PatchKind::EraseState:
                candidate.state.erase(operation.key);
                break;
            case PatchKind::SetProperty: {
                auto* component = candidate.find_component(operation.target);
                if (component == nullptr) throw std::invalid_argument("set property target component not found");
                if (operation.key.empty()) throw std::invalid_argument("set property requires key");
                component->properties[operation.key] = operation.value;
                break;
            }
            case PatchKind::EraseProperty: {
                auto* component = candidate.find_component(operation.target);
                if (component == nullptr) throw std::invalid_argument("erase property target component not found");
                component->properties.erase(operation.key);
                break;
            }
            case PatchKind::SetRoots:
                candidate.roots = operation.roots;
                break;
        }
    }
    candidate.revision = unit_.revision + 1;
    candidate.validate();
    unit_ = std::move(candidate);
}

std::string UnitRuntime::resolve_value(std::string_view value, std::string_view payload) const {
    if (value == "$payload") return std::string(payload);
    constexpr std::string_view state_prefix = "$state.";
    if (value.starts_with(state_prefix)) {
        const auto key = std::string(value.substr(state_prefix.size()));
        const auto it = unit_.state.find(key);
        return it == unit_.state.end() ? std::string{} : it->second;
    }
    return std::string(value);
}

ActionOutput UnitRuntime::execute_action(const Action& action, std::string_view payload) {
    ActionOutput output;
    output.kind = action.kind;
    output.target = action.target;
    try {
        switch (action.kind) {
            case ActionKind::SetState:
                unit_.state[action.target] = resolve_value(action.value, payload);
                output.output = unit_.state[action.target];
                output.ok = true;
                break;
            case ActionKind::SetProperty: {
                auto* component = unit_.find_component(action.target);
                if (component == nullptr) throw std::invalid_argument("SetProperty target component not found");
                component->properties[action.key] = resolve_value(action.value, payload);
                output.output = component->properties[action.key];
                output.ok = true;
                break;
            }
            case ActionKind::InvokeTool: {
                if (bindings_.tools == nullptr) throw std::logic_error("UnitRuntime has no ToolRegistry binding");
                const auto result = bindings_.tools->invoke(action.target, resolve_value(action.value, payload));
                output.ok = result.ok;
                output.output = result.output;
                output.error = result.error;
                break;
            }
            case ActionKind::InvokeAgent: {
                if (bindings_.agent == nullptr) throw std::logic_error("UnitRuntime has no AgentEngine binding");
                const auto result = bindings_.agent->run(resolve_value(action.value, payload));
                output.ok = result.ok;
                output.output = result.output;
                if (!result.ok) output.error = result.verification.empty() ? "agent execution failed" : result.verification;
                break;
            }
            case ActionKind::Emit:
                output.ok = true;
                output.output = action.target;
                break;
        }
    } catch (const std::exception& error) {
        output.ok = false;
        output.error = error.what();
    } catch (...) {
        output.ok = false;
        output.error = "unknown unit action failure";
    }
    return output;
}

DispatchResult UnitRuntime::dispatch(
    std::string_view component_id,
    std::string_view event_name,
    std::string_view payload) {
    DispatchResult result;
    if (unit_.id.empty()) {
        result.error = "UnitRuntime has no loaded unit";
        return result;
    }
    auto* component = unit_.find_component(component_id);
    if (component == nullptr) {
        result.error = "event target component not found";
        return result;
    }
    const auto event_it = component->events.find(std::string(event_name));
    if (event_it == component->events.end()) {
        result.error = "event binding not found";
        return result;
    }

    const auto actions = event_it->second;
    for (const auto& action : actions) {
        auto action_output = execute_action(action, payload);
        if (action.kind == ActionKind::Emit && action_output.ok) result.emitted_events.push_back(action_output.output);
        result.actions.push_back(action_output);
        if (!action_output.ok) {
            result.error = action_output.error;
            result.ok = false;
            return result;
        }
    }
    result.ok = true;
    return result;
}

void UnitRuntime::generate_and_load(std::string_view prompt, const UnitGenerator& generator) {
    if (!generator) throw std::invalid_argument("unit generator must be callable");
    SpiralUnit generated = generator(prompt);
    generated.validate();
    unit_ = std::move(generated);
}

Tensor UnitRuntime::matmul(const Tensor& lhs, const Tensor& rhs) const {
    if (bindings_.compute != nullptr) return bindings_.compute->matmul(lhs, rhs);
    return lhs.matmul(rhs);
}

std::string component_kind_name(ComponentKind kind) {
    switch (kind) {
        case ComponentKind::Container: return "container";
        case ComponentKind::Text: return "text";
        case ComponentKind::Button: return "button";
        case ComponentKind::ImageSurface: return "image";
        case ComponentKind::AudioSurface: return "audio";
        case ComponentKind::VideoSurface: return "video";
        case ComponentKind::WaveformSurface: return "waveform";
        case ComponentKind::Canvas: return "canvas";
        case ComponentKind::Grid: return "grid";
        case ComponentKind::Avatar: return "avatar";
        case ComponentKind::Custom: return "custom";
    }
    return "unknown";
}

bool is_media_surface(ComponentKind kind) noexcept {
    return kind == ComponentKind::ImageSurface || kind == ComponentKind::AudioSurface ||
        kind == ComponentKind::VideoSurface || kind == ComponentKind::WaveformSurface;
}

} // namespace spiral::units
