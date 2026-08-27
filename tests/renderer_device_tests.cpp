#include "spiral/device.hpp"
#include "spiral/renderer.hpp"
#include "spiral/units.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {

class RecordingRenderer final : public spiral::render::UnitRenderer {
public:
    void begin_frame(
        const spiral::render::FrameTree& tree,
        std::span<const spiral::render::Rect> dirty_regions,
        spiral::device::Device* device) override {
        ++begin_count;
        last_revision = tree.revision;
        last_device = device;
        dirty.assign(dirty_regions.begin(), dirty_regions.end());
        drawn.clear();
    }

    void draw_component(
        const spiral::units::Component& component,
        const spiral::render::FrameNode&) override {
        drawn.push_back(component.id);
    }

    void end_frame() override { ++end_count; }

    std::size_t begin_count = 0;
    std::size_t end_count = 0;
    std::uint64_t last_revision = 0;
    spiral::device::Device* last_device = nullptr;
    std::vector<spiral::render::Rect> dirty;
    std::vector<std::string> drawn;
};

spiral::units::SpiralUnit make_live_unit() {
    using namespace spiral::units;

    Component root;
    root.id = "root";
    root.kind = ComponentKind::Container;
    root.layout.width = 400.0F;
    root.layout.height = 240.0F;
    root.layout.gap = 4.0F;
    root.children = {"play", "status", "audio", "video"};

    Component play;
    play.id = "play";
    play.kind = ComponentKind::Button;
    play.layout.height = 50.0F;
    play.events["tap"] = {
        Action{ActionKind::SetProperty, "status", "text", "$payload"},
        Action{ActionKind::Emit, "ui.play", "", ""},
    };

    Component status;
    status.id = "status";
    status.kind = ComponentKind::Text;
    status.layout.height = 30.0F;
    status.properties["text"] = "idle";

    Component audio;
    audio.id = "audio";
    audio.kind = ComponentKind::AudioSurface;
    audio.layout.flex_grow = 2.0F;
    audio.properties["source"] = "ether://current/audio";

    Component video;
    video.id = "video";
    video.kind = ComponentKind::VideoSurface;
    video.layout.flex_grow = 1.0F;
    video.properties["source"] = "hakui://scene/main";

    SpiralUnit unit;
    unit.id = "l20-live-unit";
    unit.roots = {"root"};
    unit.components = {root, play, status, audio, video};
    return unit;
}

spiral::units::SpiralUnit make_grid_unit() {
    using namespace spiral::units;

    Component grid;
    grid.id = "grid";
    grid.kind = ComponentKind::Grid;
    grid.layout.width = 200.0F;
    grid.layout.height = 100.0F;
    grid.layout.gap = 10.0F;
    grid.properties["columns"] = "2";
    grid.children = {"a", "b", "c", "d"};

    std::vector<Component> components;
    components.push_back(grid);
    for (const char* id : {"a", "b", "c", "d"}) {
        Component child;
        child.id = id;
        child.kind = ComponentKind::Button;
        components.push_back(child);
    }

    SpiralUnit unit;
    unit.id = "grid-unit";
    unit.roots = {"grid"};
    unit.components = std::move(components);
    return unit;
}

bool contains_id(const std::vector<std::string>& values, const std::string& id) {
    return std::find(values.begin(), values.end(), id) != values.end();
}

} // namespace

int main() {
    using namespace spiral;

    // Native device/buffer/command-queue substrate.
    device::CpuReferenceDevice device("l20-reference-device");
    const auto source = device.create_buffer({16, device::BufferUsage::Transfer, true});
    const auto destination = device.create_buffer({16, device::BufferUsage::Storage, true});
    std::vector<std::byte> source_bytes(16);
    for (std::size_t i = 0; i < source_bytes.size(); ++i) {
        source_bytes[i] = static_cast<std::byte>(static_cast<unsigned char>(i));
    }
    device.upload(source, source_bytes);

    device::CommandList commands;
    commands.fill_buffer(destination, std::byte{0xEE});
    commands.copy_buffer(source, destination, 8, 4, 2);
    device::CommandQueue queue(device);
    queue.submit(commands);
    assert(queue.submission_count() == 1);

    const auto copied = device.download(destination);
    assert(copied.size() == 16);
    assert(copied[0] == std::byte{0xEE});
    assert(copied[1] == std::byte{0xEE});
    assert(copied[2] == std::byte{4});
    assert(copied[9] == std::byte{11});
    assert(copied[10] == std::byte{0xEE});
    auto info = device.info();
    assert(info.kind == device::DeviceKind::CpuReference);
    assert(info.buffer_count == 2);
    assert(info.allocated_bytes == 32);
    assert(device::device_kind_name(info.kind) == "cpu-reference");

    bool bounds_rejected = false;
    try {
        device::CommandList invalid;
        invalid.copy_buffer(source, destination, 32);
        queue.submit(invalid);
    } catch (const std::out_of_range&) {
        bounds_rejected = true;
    }
    assert(bounds_rejected);
    assert(queue.submission_count() == 1);

    // Deterministic layout and hit testing.
    render::LayoutEngine layout;
    const auto grid_tree = layout.resolve(make_grid_unit(), {0.0F, 0.0F, 200.0F, 100.0F});
    const auto* a = grid_tree.find("a");
    const auto* b = grid_tree.find("b");
    const auto* d = grid_tree.find("d");
    assert(a != nullptr && b != nullptr && d != nullptr);
    assert(a->bounds.width == 95.0F);
    assert(a->bounds.height == 45.0F);
    assert(b->bounds.x == 105.0F);
    assert(d->bounds.y == 55.0F);
    const auto hit_b = grid_tree.hit_test(150.0F, 20.0F);
    assert(hit_b.has_value() && *hit_b == "b");
    assert(!grid_tree.hit_test(250.0F, 20.0F).has_value());

    // Live Unit -> resolved frame tree -> renderer + host adapters.
    units::UnitRuntime runtime(make_live_unit());
    RecordingRenderer renderer;
    std::size_t etherplay_presentations = 0;
    std::size_t hakui_presentations = 0;
    std::string etherplay_source;
    std::string hakui_source;

    render::RenderBindings render_bindings;
    render_bindings.device = &device;
    render_bindings.surface_adapters.push_back(render::HostSurfaceAdapter{
        "etherplay",
        [](units::ComponentKind kind) { return kind == units::ComponentKind::AudioSurface; },
        [&](const units::Component& component, const render::FrameNode&) {
            ++etherplay_presentations;
            etherplay_source = component.properties.at("source");
        }});
    render_bindings.surface_adapters.push_back(render::HostSurfaceAdapter{
        "hakui",
        [](units::ComponentKind kind) { return kind == units::ComponentKind::VideoSurface; },
        [&](const units::Component& component, const render::FrameNode&) {
            ++hakui_presentations;
            hakui_source = component.properties.at("source");
        }});

    render::RendererBridge bridge(runtime, renderer, std::move(render_bindings));
    bridge.mount({0.0F, 0.0F, 400.0F, 240.0F});
    assert(renderer.begin_count == 1 && renderer.end_count == 1);
    assert(renderer.last_device == &device);
    assert(renderer.last_revision == 0);
    assert(contains_id(renderer.drawn, "root"));
    assert(contains_id(renderer.drawn, "play"));
    assert(contains_id(renderer.drawn, "audio"));
    assert(contains_id(renderer.drawn, "video"));
    assert(etherplay_presentations == 1);
    assert(hakui_presentations == 1);
    assert(etherplay_source == "ether://current/audio");
    assert(hakui_source == "hakui://scene/main");

    const auto* play_frame = bridge.frame_tree().find("play");
    const auto* status_frame = bridge.frame_tree().find("status");
    assert(play_frame != nullptr && status_frame != nullptr);
    assert(play_frame->bounds.height == 50.0F);
    assert(status_frame->bounds.y == 54.0F);

    const auto tap = bridge.dispatch_pointer(10.0F, 10.0F, "tap", "playing");
    assert(tap.hit && tap.component_id == "play");
    assert(tap.dispatch.ok);
    assert(tap.dispatch.emitted_events.size() == 1 && tap.dispatch.emitted_events[0] == "ui.play");
    assert(runtime.unit().find_component("status")->properties.at("text") == "playing");
    assert(renderer.begin_count == 2);
    assert(contains_id(renderer.drawn, "status"));
    assert(!contains_id(renderer.drawn, "play"));

    // Hot layout patch must invalidate changed geometry and downstream flow, while preserving revision semantics.
    units::Component patched_status = *runtime.unit().find_component("status");
    patched_status.layout.height = 60.0F;
    units::UnitPatch patch;
    patch.base_revision = 0;
    units::PatchOperation upsert;
    upsert.kind = units::PatchKind::UpsertComponent;
    upsert.component = patched_status;
    patch.operations.push_back(upsert);
    bridge.apply_patch(patch);
    assert(runtime.unit().revision == 1);
    assert(bridge.frame_tree().revision == 1);
    assert(bridge.frame_tree().find("status")->bounds.height == 60.0F);
    assert(bridge.frame_tree().find("audio")->bounds.y > 100.0F);
    assert(contains_id(renderer.drawn, "status"));
    assert(contains_id(renderer.drawn, "audio"));

    bool stale_rejected = false;
    try {
        bridge.apply_patch(patch);
    } catch (const std::invalid_argument&) {
        stale_rejected = true;
    }
    assert(stale_rejected);
    assert(runtime.unit().revision == 1);

    // Corrupt patch must roll back without changing the live frame tree.
    const auto status_before = bridge.frame_tree().find("status")->bounds;
    units::UnitPatch corrupt;
    corrupt.base_revision = 1;
    units::PatchOperation remove;
    remove.kind = units::PatchKind::RemoveComponent;
    remove.target = "status";
    corrupt.operations.push_back(remove);
    bool corrupt_rejected = false;
    try {
        bridge.apply_patch(corrupt);
    } catch (const std::invalid_argument&) {
        corrupt_rejected = true;
    }
    assert(corrupt_rejected);
    assert(runtime.unit().revision == 1);
    assert(bridge.frame_tree().find("status") != nullptr);
    assert(bridge.frame_tree().find("status")->bounds.x == status_before.x);
    assert(bridge.frame_tree().find("status")->bounds.y == status_before.y);

    bridge.resize({0.0F, 0.0F, 800.0F, 480.0F});
    assert(bridge.viewport().width == 800.0F);
    assert(renderer.dirty.size() >= 1);
    bridge.redraw();
    assert(renderer.dirty.size() == 1);
    assert(renderer.dirty[0].width == 800.0F);

    device.destroy_buffer(source);
    info = device.info();
    assert(info.buffer_count == 1);

    std::cout << "L20 device submissions: " << queue.submission_count() << '\n';
    std::cout << "L20 EtherPlay presentations: " << etherplay_presentations << '\n';
    std::cout << "L20 Hakui presentations: " << hakui_presentations << '\n';
    std::cout << "L20 renderer/device bridge tests passed\n";
    return 0;
}
