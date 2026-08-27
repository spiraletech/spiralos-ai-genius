#include "spiral/host_bridge.hpp"

#include <utility>

namespace spiral::host {

void HostBridgeLog::present(
    HostKind host,
    const units::Component& component,
    const render::FrameNode& frame) {
    const auto source = component.properties.find("source");
    records_.push_back(PresentationRecord{
        host,
        component.id,
        component.kind,
        source == component.properties.end() ? std::string{} : source->second,
        frame.bounds});
}

render::HostSurfaceAdapter make_etherplay_adapter(HostBridgeLog& log) {
    render::HostSurfaceAdapter adapter;
    adapter.host_name = "EtherPlay";
    adapter.accepts = [](units::ComponentKind kind) {
        return kind == units::ComponentKind::ImageSurface ||
            kind == units::ComponentKind::AudioSurface ||
            kind == units::ComponentKind::VideoSurface ||
            kind == units::ComponentKind::WaveformSurface;
    };
    adapter.present = [&log](const units::Component& component, const render::FrameNode& frame) {
        log.present(HostKind::EtherPlay, component, frame);
    };
    return adapter;
}

render::HostSurfaceAdapter make_hakui_adapter(HostBridgeLog& log) {
    render::HostSurfaceAdapter adapter;
    adapter.host_name = "Hakui";
    adapter.accepts = [](units::ComponentKind kind) {
        return kind == units::ComponentKind::ImageSurface ||
            kind == units::ComponentKind::VideoSurface ||
            kind == units::ComponentKind::Canvas ||
            kind == units::ComponentKind::Avatar ||
            kind == units::ComponentKind::Custom;
    };
    adapter.present = [&log](const units::Component& component, const render::FrameNode& frame) {
        log.present(HostKind::Hakui, component, frame);
    };
    return adapter;
}

std::string host_kind_name(HostKind kind) {
    switch (kind) {
        case HostKind::EtherPlay: return "EtherPlay";
        case HostKind::Hakui: return "Hakui";
    }
    return "unknown";
}

} // namespace spiral::host
