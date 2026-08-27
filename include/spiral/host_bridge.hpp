#pragma once

#include "spiral/renderer.hpp"

#include <string>
#include <vector>

namespace spiral::host {

enum class HostKind {
    EtherPlay,
    Hakui,
};

struct PresentationRecord {
    HostKind host = HostKind::EtherPlay;
    std::string component_id;
    units::ComponentKind kind = units::ComponentKind::Custom;
    std::string source;
    render::Rect bounds;
};

class HostBridgeLog final {
public:
    void present(HostKind host, const units::Component& component, const render::FrameNode& frame);
    void clear() noexcept { records_.clear(); }
    [[nodiscard]] const std::vector<PresentationRecord>& records() const noexcept { return records_; }

private:
    std::vector<PresentationRecord> records_;
};

[[nodiscard]] render::HostSurfaceAdapter make_etherplay_adapter(HostBridgeLog& log);
[[nodiscard]] render::HostSurfaceAdapter make_hakui_adapter(HostBridgeLog& log);
[[nodiscard]] std::string host_kind_name(HostKind kind);

} // namespace spiral::host
