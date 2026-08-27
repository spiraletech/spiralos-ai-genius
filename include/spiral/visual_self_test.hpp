#pragma once

#include "spiral/raster.hpp"
#include "spiral/renderer.hpp"
#include "spiral/units.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace spiral::visual {

enum class IssueKind {
    UndersizedInteractive,
    HeavilyClipped,
    SparseFrame,
};

struct Issue {
    IssueKind kind = IssueKind::SparseFrame;
    std::string component_id;
    std::string message;
    render::Rect bounds;
};

struct Report {
    bool healthy = false;
    std::uint64_t snapshot_hash = 0;
    std::size_t non_background_pixels = 0;
    std::vector<Issue> issues;
};

class VisualCritic final {
public:
    struct Config {
        float minimum_interactive_width = 44.0F;
        float minimum_interactive_height = 44.0F;
        float minimum_visible_fraction = 0.50F;
        float minimum_ink_fraction = 0.002F;
    };

    VisualCritic();
    explicit VisualCritic(Config config);

    [[nodiscard]] Report inspect(
        const units::SpiralUnit& unit,
        const render::FrameTree& tree,
        const raster::Framebuffer& framebuffer,
        raster::Rgba8 background) const;

private:
    Config config_;
};

class RepairPolicy final {
public:
    [[nodiscard]] units::UnitPatch propose(
        const units::SpiralUnit& unit,
        const Report& report) const;
};

struct SelfTestResult {
    bool healthy = false;
    bool repaired = false;
    std::size_t repair_passes = 0;
    std::vector<Report> reports;
};

class VisualSelfTest final {
public:
    VisualSelfTest(
        units::UnitRuntime& runtime,
        render::RendererBridge& bridge,
        raster::SoftwareRenderer& renderer,
        VisualCritic critic = VisualCritic{},
        RepairPolicy repair = RepairPolicy{});

    [[nodiscard]] SelfTestResult run(std::size_t max_repairs = 3);

private:
    units::UnitRuntime& runtime_;
    render::RendererBridge& bridge_;
    raster::SoftwareRenderer& renderer_;
    VisualCritic critic_;
    RepairPolicy repair_;
};

[[nodiscard]] std::string issue_kind_name(IssueKind kind);

} // namespace spiral::visual
