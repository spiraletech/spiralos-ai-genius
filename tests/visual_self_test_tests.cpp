#include "spiral/host_bridge.hpp"
#include "spiral/raster.hpp"
#include "spiral/renderer.hpp"
#include "spiral/units.hpp"
#include "spiral/visual_self_test.hpp"

#include <cassert>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace {

spiral::units::SpiralUnit make_visual_unit() {
    using namespace spiral::units;

    Component root;
    root.id = "root";
    root.kind = ComponentKind::Container;
    root.layout.width = 180.0F;
    root.layout.height = 180.0F;
    root.layout.gap = 4.0F;
    root.properties["background"] = "#181C24";
    root.children = {"title", "tiny", "wave", "video"};

    Component title;
    title.id = "title";
    title.kind = ComponentKind::Text;
    title.layout.height = 20.0F;
    title.properties["text"] = "SPIRAL VISUAL SELF TEST";

    Component tiny;
    tiny.id = "tiny";
    tiny.kind = ComponentKind::Button;
    tiny.layout.width = 20.0F;
    tiny.layout.height = 14.0F;
    tiny.properties["label"] = "GO";
    tiny.events["tap"] = {
        Action{ActionKind::SetProperty, "tiny", "label", "$payload"},
    };

    Component wave;
    wave.id = "wave";
    wave.kind = ComponentKind::WaveformSurface;
    wave.layout.height = 32.0F;
    wave.properties["source"] = "ether://current-track";
    wave.properties["samples"] = "0,0.8,-0.5,1,-1,0.4,-0.2,0.7,0";

    Component video;
    video.id = "video";
    video.kind = ComponentKind::VideoSurface;
    video.layout.height = 52.0F;
    video.properties["source"] = "hakui://scene/live";

    SpiralUnit unit;
    unit.id = "l21-visual-unit";
    unit.roots = {"root"};
    unit.components = {root, title, tiny, wave, video};
    return unit;
}

bool has_issue(const spiral::visual::Report& report, spiral::visual::IssueKind kind, const std::string& id) {
    for (const auto& issue : report.issues) {
        if (issue.kind == kind && issue.component_id == id) return true;
    }
    return false;
}

} // namespace

int main() {
    using namespace spiral;

    auto unit = make_visual_unit();
    unit.validate();
    units::UnitRuntime runtime(unit);
    raster::SoftwareRenderer renderer;
    host::HostBridgeLog host_log;

    render::RenderBindings bindings;
    bindings.surface_adapters = {
        host::make_etherplay_adapter(host_log),
        host::make_hakui_adapter(host_log),
    };

    render::RendererBridge bridge(runtime, renderer, bindings);
    bridge.mount(render::Rect{0.0F, 0.0F, 180.0F, 180.0F});

    assert(renderer.framebuffer().width() == 180);
    assert(renderer.framebuffer().height() == 180);
    assert(renderer.frame_count() == 1);
    const std::uint64_t initial_hash = renderer.framebuffer().hash64();
    assert(initial_hash != 0);
    assert(renderer.framebuffer().count_non_background(raster::RasterTheme{}.background) > 1000);

    bool saw_etherplay_wave = false;
    bool saw_hakui_video = false;
    for (const auto& record : host_log.records()) {
        if (record.host == host::HostKind::EtherPlay && record.component_id == "wave") saw_etherplay_wave = true;
        if (record.host == host::HostKind::Hakui && record.component_id == "video") saw_hakui_video = true;
    }
    assert(saw_etherplay_wave);
    assert(saw_hakui_video);

    const std::filesystem::path snapshot_path = "spiral_l21_snapshot.ppm";
    renderer.framebuffer().save_ppm(snapshot_path.string());
    {
        std::ifstream snapshot(snapshot_path, std::ios::binary);
        assert(snapshot.good());
        std::string magic;
        snapshot >> magic;
        assert(magic == "P6");
    }
    std::filesystem::remove(snapshot_path);

    visual::VisualCritic critic;
    const auto before = critic.inspect(
        runtime.unit(), bridge.frame_tree(), renderer.framebuffer(), raster::RasterTheme{}.background);
    assert(!before.healthy);
    assert(has_issue(before, visual::IssueKind::UndersizedInteractive, "tiny"));
    assert(visual::issue_kind_name(visual::IssueKind::UndersizedInteractive) == "undersized-interactive");

    visual::VisualSelfTest self_test(runtime, bridge, renderer);
    const auto result = self_test.run(2);
    assert(result.repaired);
    assert(result.repair_passes == 1);
    assert(result.healthy);
    assert(result.reports.size() == 2);
    assert(!result.reports.front().healthy);
    assert(result.reports.back().healthy);
    assert(runtime.unit().revision == 1);

    const auto* repaired_component = runtime.unit().find_component("tiny");
    const auto* repaired_frame = bridge.frame_tree().find("tiny");
    assert(repaired_component != nullptr && repaired_frame != nullptr);
    assert(repaired_component->layout.min_width >= 44.0F);
    assert(repaired_component->layout.min_height >= 44.0F);
    assert(repaired_frame->bounds.width >= 44.0F);
    assert(repaired_frame->bounds.height >= 44.0F);

    const std::uint64_t repaired_hash = renderer.framebuffer().hash64();
    assert(repaired_hash != initial_hash);

    const float tap_x = repaired_frame->bounds.x + repaired_frame->bounds.width * 0.5F;
    const float tap_y = repaired_frame->bounds.y + repaired_frame->bounds.height * 0.5F;
    const auto tap = bridge.dispatch_pointer(tap_x, tap_y, "tap", "PLAY");
    assert(tap.hit);
    assert(tap.component_id == "tiny");
    assert(tap.dispatch.ok);
    assert(runtime.unit().find_component("tiny")->properties.at("label") == "PLAY");
    assert(renderer.framebuffer().hash64() != repaired_hash);

    // Same Unit + same software renderer must produce the same initial snapshot hash.
    units::UnitRuntime runtime_clone(make_visual_unit());
    raster::SoftwareRenderer renderer_clone;
    render::RendererBridge bridge_clone(runtime_clone, renderer_clone);
    bridge_clone.mount(render::Rect{0.0F, 0.0F, 180.0F, 180.0F});
    assert(renderer_clone.framebuffer().hash64() == initial_hash);

    std::cout << "L21 initial snapshot hash: " << initial_hash << '\n';
    std::cout << "L21 repaired snapshot hash: " << repaired_hash << '\n';
    std::cout << "L21 host presentations: " << host_log.records().size() << '\n';
    std::cout << "L21 visual self-test passed\n";
    return 0;
}
