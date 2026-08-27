#include "spiral/visual_self_test.hpp"

#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>
#include <utility>

namespace spiral::visual {
namespace {

float area(render::Rect rect) noexcept {
    return std::max(0.0F, rect.width) * std::max(0.0F, rect.height);
}

void inspect_nodes(
    const std::vector<render::FrameNode>& nodes,
    const units::SpiralUnit& unit,
    const VisualCritic::Config& config,
    std::vector<Issue>& issues) {
    for (const auto& node : nodes) {
        const auto* component = unit.find_component(node.component_id);
        if (component == nullptr) {
            throw std::logic_error("visual critic frame references missing component");
        }

        if (node.interactive &&
            (node.bounds.width + 1.0e-5F < config.minimum_interactive_width ||
             node.bounds.height + 1.0e-5F < config.minimum_interactive_height)) {
            issues.push_back(Issue{
                IssueKind::UndersizedInteractive,
                node.component_id,
                "interactive target is smaller than the configured visual minimum",
                node.bounds});
        }

        const float total_area = area(node.bounds);
        if (total_area > 0.0F) {
            const auto visible = render::intersect_rect(node.bounds, node.clip);
            const float visible_fraction = area(visible) / total_area;
            if (visible_fraction + 1.0e-5F < config.minimum_visible_fraction) {
                issues.push_back(Issue{
                    IssueKind::HeavilyClipped,
                    node.component_id,
                    "component is mostly outside its inherited clip rectangle",
                    node.bounds});
            }
        }

        inspect_nodes(node.children, unit, config, issues);
    }
}

} // namespace

VisualCritic::VisualCritic() : VisualCritic(Config{}) {}

VisualCritic::VisualCritic(Config config) : config_(config) {
    if (!(config_.minimum_interactive_width > 0.0F) ||
        !(config_.minimum_interactive_height > 0.0F)) {
        throw std::invalid_argument("visual critic interactive minimum must be positive");
    }
    if (config_.minimum_visible_fraction < 0.0F || config_.minimum_visible_fraction > 1.0F) {
        throw std::invalid_argument("visual critic visible fraction must be in [0, 1]");
    }
    if (config_.minimum_ink_fraction < 0.0F || config_.minimum_ink_fraction > 1.0F) {
        throw std::invalid_argument("visual critic ink fraction must be in [0, 1]");
    }
}

Report VisualCritic::inspect(
    const units::SpiralUnit& unit,
    const render::FrameTree& tree,
    const raster::Framebuffer& framebuffer,
    raster::Rgba8 background) const {
    unit.validate();
    if (framebuffer.width() == 0 || framebuffer.height() == 0) {
        throw std::invalid_argument("visual critic requires a rendered framebuffer");
    }

    Report report;
    report.snapshot_hash = framebuffer.hash64();
    report.non_background_pixels = framebuffer.count_non_background(background);
    inspect_nodes(tree.roots, unit, config_, report.issues);

    const std::size_t total_pixels = framebuffer.width() * framebuffer.height();
    const float ink_fraction = total_pixels == 0
        ? 0.0F
        : static_cast<float>(report.non_background_pixels) / static_cast<float>(total_pixels);
    if (ink_fraction + 1.0e-7F < config_.minimum_ink_fraction) {
        report.issues.push_back(Issue{
            IssueKind::SparseFrame,
            {},
            "rendered framebuffer contains too little visible content",
            tree.viewport});
    }

    report.healthy = report.issues.empty();
    return report;
}

units::UnitPatch RepairPolicy::propose(
    const units::SpiralUnit& unit,
    const Report& report) const {
    units::UnitPatch patch;
    patch.base_revision = unit.revision;

    std::set<std::string> repaired_components;
    for (const auto& issue : report.issues) {
        if (issue.kind != IssueKind::UndersizedInteractive || issue.component_id.empty()) continue;
        if (!repaired_components.insert(issue.component_id).second) continue;

        const auto* original = unit.find_component(issue.component_id);
        if (original == nullptr) continue;
        units::Component repaired = *original;
        repaired.layout.min_width = std::max(repaired.layout.min_width, 44.0F);
        repaired.layout.min_height = std::max(repaired.layout.min_height, 44.0F);
        if (repaired.layout.width > 0.0F && repaired.layout.width < 44.0F) repaired.layout.width = 44.0F;
        if (repaired.layout.height > 0.0F && repaired.layout.height < 44.0F) repaired.layout.height = 44.0F;

        units::PatchOperation operation;
        operation.kind = units::PatchKind::UpsertComponent;
        operation.component = std::move(repaired);
        patch.operations.push_back(std::move(operation));
    }
    return patch;
}

VisualSelfTest::VisualSelfTest(
    units::UnitRuntime& runtime,
    render::RendererBridge& bridge,
    raster::SoftwareRenderer& renderer,
    VisualCritic critic,
    RepairPolicy repair)
    : runtime_(runtime),
      bridge_(bridge),
      renderer_(renderer),
      critic_(std::move(critic)),
      repair_(std::move(repair)) {}

SelfTestResult VisualSelfTest::run(std::size_t max_repairs) {
    SelfTestResult result;
    for (std::size_t pass = 0; ; ++pass) {
        auto report = critic_.inspect(
            runtime_.unit(),
            bridge_.frame_tree(),
            renderer_.framebuffer(),
            renderer_.background_color());
        result.reports.push_back(report);
        if (report.healthy) {
            result.healthy = true;
            return result;
        }
        if (pass >= max_repairs) return result;

        auto patch = repair_.propose(runtime_.unit(), report);
        if (patch.operations.empty()) return result;
        bridge_.apply_patch(patch);
        result.repaired = true;
        ++result.repair_passes;
    }
}

std::string issue_kind_name(IssueKind kind) {
    switch (kind) {
        case IssueKind::UndersizedInteractive: return "undersized-interactive";
        case IssueKind::HeavilyClipped: return "heavily-clipped";
        case IssueKind::SparseFrame: return "sparse-frame";
    }
    return "unknown";
}

} // namespace spiral::visual
