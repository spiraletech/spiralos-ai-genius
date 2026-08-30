#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace spiral::genius {

enum class Project {
    General,
    SpiralOS,
    Hakui,
    EtherPlay,
    EtherBeat,
};

enum class MaltState {
    Pass,
    Malt,
    Failsafe,
    Halt,
};

enum class AumMode {
    Create,
    Preserve,
    Transform,
};

enum class MindNotch {
    Observe,
    Recall,
    Analyze,
    Diagnose,
    Design,
    Verify,
    Reflect,
    Communicate,
};

enum class CodeNotch {
    Inspect,
    Trace,
    Explain,
    Plan,
    Implement,
    Compile,
    Test,
    Repair,
};

struct LiratelFrame {
    std::string source;
    std::string energy;
    std::string direction;
    std::string intent;
    std::string outcome;
    std::vector<std::string> sigils;
};

struct SteamTelemetry {
    float pressure = 0.0F;
    float heat = 0.0F;
    float uncertainty = 0.0F;
    float contradiction = 0.0F;
    float evidence = 0.0F;
    float tool_need = 0.0F;
    bool redline = false;
};

struct LambdaState {
    float score = 0.0F;
    float contradiction_penalty = 0.0F;
    bool stable = false;
};

struct Context {
    std::string host_context;
    std::string last_topic;
    std::string last_reply;
    float coherence = 0.5F;
    float confidence = 0.5F;
    float focus = 0.5F;
    float curiosity = 0.5F;
    std::size_t memory_count = 0;
};

struct Trace {
    Project project = Project::General;
    MaltState malt = MaltState::Pass;
    AumMode aum = AumMode::Preserve;
    MindNotch mind = MindNotch::Observe;
    CodeNotch code = CodeNotch::Explain;
    LiratelFrame liratel;
    SteamTelemetry steam;
    LambdaState lambda;
    bool coding_question = false;
    bool project_question = false;
    bool tool_required = false;
    bool project_profile_verified = false;
    std::string topic;
    std::string evidence_summary;
};

class Kernel final {
public:
    [[nodiscard]] Trace evaluate(std::string_view input, const Context& context) const;
    [[nodiscard]] std::string answer(std::string_view input, const Context& context, const Trace& trace) const;

    [[nodiscard]] static std::string project_name(Project value);
    [[nodiscard]] static std::string malt_name(MaltState value);
    [[nodiscard]] static std::string aum_name(AumMode value);
    [[nodiscard]] static std::string mind_name(MindNotch value);
    [[nodiscard]] static std::string code_name(CodeNotch value);
    [[nodiscard]] static std::string hologram(const Trace& trace);
};

} // namespace spiral::genius
