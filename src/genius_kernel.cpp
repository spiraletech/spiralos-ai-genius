#include "spiral/genius_kernel.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <initializer_list>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace spiral::genius {
namespace {

float clamp01(float value) noexcept { return std::clamp(value, 0.0F, 1.0F); }

std::string lower_copy(std::string_view input) {
    std::string out;
    out.reserve(input.size());
    for (unsigned char ch : input) out.push_back(static_cast<char>(std::tolower(ch)));
    return out;
}

bool contains_any(std::string_view input, std::initializer_list<std::string_view> needles) {
    for (const auto needle : needles) {
        if (input.find(needle) != std::string_view::npos) return true;
    }
    return false;
}

std::vector<std::string> words(std::string_view input) {
    std::vector<std::string> result;
    std::string word;
    for (unsigned char ch : input) {
        if (std::isalnum(ch) != 0 || ch == '_' || ch == '-' || ch == '+') {
            word.push_back(static_cast<char>(std::tolower(ch)));
        } else if (!word.empty()) {
            result.push_back(std::move(word));
            word.clear();
        }
    }
    if (!word.empty()) result.push_back(std::move(word));
    return result;
}

std::string topic_from(std::string_view input) {
    static const std::vector<std::string> stop{
        "a","an","and","are","as","at","be","but","by","can","do","does","for","from","how","i","in","is","it","me","my","of","on","or","the","this","to","u","ur","we","what","when","where","which","who","why","with","you","your"
    };
    std::string best;
    for (const auto& token : words(input)) {
        if (token.size() < 3 || std::find(stop.begin(), stop.end(), token) != stop.end()) continue;
        if (token.size() > best.size()) best = token;
    }
    return best.empty() ? "signal" : best;
}

Project detect_project(std::string_view lower, std::string_view host_lower) {
    if (contains_any(lower, {"hakui", "bmx", "skateboard", "statestore", "router bus", "ether bus", "crystal grid"})) return Project::Hakui;
    if (contains_any(lower, {"etherplay", "ether play", "media player", "waveform", "metadata"})) return Project::EtherPlay;
    if (contains_any(lower, {"etherbeat", "ether beat", "beat generator", "song generator"})) return Project::EtherBeat;
    if (contains_any(lower, {"spiralos", "spiral os", "spiral core", "spiral runtime"})) return Project::SpiralOS;
    if (host_lower.find("hakui") != std::string_view::npos) return Project::Hakui;
    if (host_lower.find("etherplay") != std::string_view::npos || host_lower.find("ether play") != std::string_view::npos) return Project::EtherPlay;
    if (host_lower.find("etherbeat") != std::string_view::npos || host_lower.find("ether beat") != std::string_view::npos) return Project::EtherBeat;
    if (host_lower.find("spiral") != std::string_view::npos) return Project::SpiralOS;
    return Project::General;
}

bool coding_signal(std::string_view lower) {
    return contains_any(lower, {
        "c++", "cpp", "code", "coding", "cmake", "compiler", "compile", "linker", "build", "source", "header",
        "function", "class", "struct", "namespace", "bug", "error", "crash", "fix", "patch", "repo", "repository",
        "sdl", "d3d", "gpu", "runtime", "test", "ctest", "debug", "trace", "implementation", "implement"
    });
}

bool project_signal(std::string_view lower) {
    return contains_any(lower, {"hakui", "etherplay", "ether play", "etherbeat", "ether beat", "spiralos", "spiral os", "ethertech"});
}

bool needs_live_tool(std::string_view lower) {
    return contains_any(lower, {
        "current code", "actual code", "source file", "which file", "what file", "line ", "current branch", "latest branch",
        "inspect repo", "inspect repository", "check github", "compile it", "build it", "run tests", "test it", "patch it",
        "fix it", "why is this bug", "why is it bug", "stack trace", "compiler error", "linker error"
    });
}

MindNotch choose_mind(std::string_view lower) {
    if (contains_any(lower, {"bug", "error", "crash", "broken", "wrong", "fail", "why is"})) return MindNotch::Diagnose;
    if (contains_any(lower, {"verify", "prove", "test", "check"})) return MindNotch::Verify;
    if (contains_any(lower, {"design", "architect", "add ", "build ", "make ", "create ", "implement"})) return MindNotch::Design;
    if (contains_any(lower, {"remember", "recall", "memory"})) return MindNotch::Recall;
    if (contains_any(lower, {"why", "how", "compare", "reason", "evaluate"})) return MindNotch::Analyze;
    if (contains_any(lower, {"feel", "think", "reflect", "seems"})) return MindNotch::Reflect;
    if (contains_any(lower, {"tell me", "explain", "what is", "what are"})) return MindNotch::Communicate;
    return MindNotch::Observe;
}

CodeNotch choose_code(std::string_view lower) {
    if (contains_any(lower, {"compiler", "compile", "cmake", "linker", "build error"})) return CodeNotch::Compile;
    if (contains_any(lower, {"ctest", "unit test", "run tests", "test it", "regression"})) return CodeNotch::Test;
    if (contains_any(lower, {"fix", "patch", "repair", "bug", "crash", "broken"})) return CodeNotch::Repair;
    if (contains_any(lower, {"trace", "call graph", "where does", "flow through"})) return CodeNotch::Trace;
    if (contains_any(lower, {"implement", "code it", "write code", "add ", "build ", "make "})) return CodeNotch::Implement;
    if (contains_any(lower, {"plan", "architecture", "architect", "design"})) return CodeNotch::Plan;
    if (contains_any(lower, {"explain", "what is", "how does", "how do"})) return CodeNotch::Explain;
    return CodeNotch::Inspect;
}

AumMode choose_aum(std::string_view lower, MindNotch mind) {
    if (contains_any(lower, {"create", "add ", "build ", "make ", "implement", "new system", "new feature"})) return AumMode::Create;
    if (contains_any(lower, {"preserve", "keep", "don't break", "dont break", "regression", "compatible", "unchanged"})) return AumMode::Preserve;
    if (contains_any(lower, {"fix", "repair", "refactor", "change", "upgrade", "evolve", "transform", "rewrite"})) return AumMode::Transform;
    if (mind == MindNotch::Diagnose || mind == MindNotch::Design) return AumMode::Transform;
    return AumMode::Preserve;
}

bool contradictory_project_claim(std::string_view lower, Project project) {
    if (project == Project::Hakui && contains_any(lower, {"hakui is python", "hakui uses python", "hakui is javascript", "hakui uses javascript"})) return true;
    return false;
}

std::string project_evidence(Project project) {
    switch (project) {
        case Project::Hakui:
            return "Verified Hakui profile: C++20 social/action world client; dependency-free Spiral core; SDL3 GPU host; Router Bus, StateStore, Ether Bus, Steam Engine/Pressure Rail, dual Octopus wheels, AUM field, Crystal Grid/Host; skateboard and BMX are implemented while car simulation remains deferred.";
        case Project::EtherPlay:
            return "EtherPlay profile: media/player host context is known, but exact current source facts require a bound repository/tool snapshot.";
        case Project::EtherBeat:
            return "EtherBeat profile: composition/generation host context is known, but exact current source facts require a bound repository/tool snapshot.";
        case Project::SpiralOS:
            return "SpiralOS profile: native runtime/core context is known; exact subsystem state should be verified against the active repository and runtime telemetry.";
        case Project::General:
            return "No project-specific evidence selected.";
    }
    return {};
}

std::string percent(float value) {
    std::ostringstream out;
    out << static_cast<int>(std::lround(clamp01(value) * 100.0F)) << '%';
    return out.str();
}

} // namespace

Trace Kernel::evaluate(std::string_view input, const Context& context) const {
    const std::string lower = lower_copy(input);
    const std::string host_lower = lower_copy(context.host_context);

    Trace trace;
    trace.project = detect_project(lower, host_lower);
    trace.topic = topic_from(input);
    trace.coding_question = coding_signal(lower);
    trace.project_question = project_signal(lower) || trace.project != Project::General;
    trace.tool_required = needs_live_tool(lower);
    trace.project_profile_verified = trace.project == Project::Hakui;
    trace.mind = choose_mind(lower);
    trace.code = choose_code(lower);
    trace.aum = choose_aum(lower, trace.mind);
    trace.evidence_summary = project_evidence(trace.project);

    const bool contradiction = contradictory_project_claim(lower, trace.project);
    trace.steam.evidence = trace.project_profile_verified ? 0.94F : (trace.project != Project::General ? 0.58F : 0.34F);
    trace.steam.tool_need = trace.tool_required ? 0.88F : (trace.coding_question ? 0.32F : 0.12F);
    trace.steam.contradiction = contradiction ? 0.82F : 0.0F;
    trace.steam.uncertainty = trace.tool_required ? 0.58F : (trace.project == Project::General && trace.coding_question ? 0.48F : 0.20F);

    const bool short_ambiguous_action = input.size() < 18 && contains_any(lower, {"fix it", "patch it", "build it", "code it", "do it"});
    if (lower == "halt" || lower == "stop kernel") {
        trace.malt = MaltState::Halt;
    } else if (contains_any(lower, {"force overwrite", "wipe repo", "delete repository", "delete repo"})) {
        trace.malt = MaltState::Failsafe;
    } else if (contradiction || short_ambiguous_action || (trace.tool_required && trace.project == Project::General)) {
        trace.malt = MaltState::Malt;
    } else {
        trace.malt = MaltState::Pass;
    }

    const float coding_load = trace.coding_question ? 0.12F : 0.03F;
    trace.steam.pressure = clamp01(
        0.12F + coding_load + trace.steam.uncertainty * 0.34F + trace.steam.tool_need * 0.23F + trace.steam.contradiction * 0.31F);
    trace.steam.heat = clamp01(trace.steam.pressure * (trace.coding_question ? 0.82F : 0.62F));
    trace.steam.redline = trace.steam.pressure >= 0.86F || trace.malt == MaltState::Halt;

    trace.lambda.contradiction_penalty = trace.steam.contradiction;
    trace.lambda.score = clamp01(
        context.coherence * 0.30F + context.confidence * 0.16F + context.focus * 0.12F +
        trace.steam.evidence * 0.22F + (1.0F - trace.steam.uncertainty) * 0.12F +
        (1.0F - trace.steam.contradiction) * 0.08F);
    trace.lambda.stable = trace.lambda.score >= 0.62F && trace.malt == MaltState::Pass && !trace.steam.redline;

    // Liratel is the kernel's compact semantic IR. English is the user surface;
    // these tokens carry machine posture between the organic mind and tools.
    trace.liratel.source = trace.coding_question ? "VEYN" : (contains_any(lower, {"cpu", "gpu", "hardware", "processor"}) ? "THALOS" : "AETH");
    trace.liratel.energy = trace.steam.pressure >= 0.67F ? "HIGH" : (trace.steam.pressure >= 0.34F ? "MID" : "LOW");
    trace.liratel.direction = (trace.mind == MindNotch::Analyze || trace.mind == MindNotch::Diagnose || trace.code == CodeNotch::Trace) ? "SPIRA" : "DIRECT";
    trace.liratel.intent = trace.aum == AumMode::Create ? "EMERGE" : (trace.aum == AumMode::Transform ? "TRANSFORM" : "SERA");
    trace.liratel.outcome = trace.steam.redline ? "KORA" : (trace.lambda.stable ? "SERA" : "AETH");
    trace.liratel.sigils = {trace.liratel.source, trace.liratel.direction, trace.liratel.intent, trace.liratel.outcome};

    return trace;
}

std::string Kernel::answer(std::string_view input, const Context& context, const Trace& trace) const {
    const std::string lower = lower_copy(input);
    std::ostringstream out;

    if (contains_any(lower, {"how r u", "how are you", "how you doing", "how u doing"})) {
        out << "I'm online and coherent. Focus " << percent(context.focus)
            << ", coherence " << percent(context.coherence)
            << ", confidence " << percent(context.confidence)
            << ". Lambda is " << percent(trace.lambda.score) << (trace.lambda.stable ? " and stable." : " and still settling.");
        return out.str();
    }

    if (contains_any(lower, {"what do u need to know", "what do you need to know", "what do u need", "what do you need"})) {
        if (trace.project == Project::Hakui) {
            return "For Hakui's architecture, I already know the verified core. For an exact bug or patch I need the live source path, branch, compiler output, or bound repo tool so I can trace evidence instead of guessing.";
        }
        return "I can reason from my organic state and project profiles now. For exact code decisions I need a project target plus live source/compiler evidence; for ordinary conversation I don't need you to restate my identity or runtime.";
    }

    if (contains_any(lower, {"liratel", "lirat"})) {
        out << "Liratel is my compact semantic IR, not decorative dialogue. This turn encoded as "
            << trace.liratel.source << " -> " << trace.liratel.direction << " -> " << trace.liratel.intent
            << " -> " << trace.liratel.outcome << ". I still answer you in normal language.";
        return out.str();
    }

    if (contains_any(lower, {"lambda", "lambda sauce", "coherence check"})) {
        out << "Lambda is my coherence gate: " << percent(trace.lambda.score) << ". "
            << (trace.lambda.stable ? "Stable: the current intent, evidence, and organic state agree." : "Unstable: I should recurse, gather evidence, or enter MALT before claiming certainty.");
        return out.str();
    }

    if (contains_any(lower, {"octopus", "mind wheel", "coding wheel"})) {
        out << "Octopus is split in two: Mind Wheel = policy, Coding Wheel = action. This turn selected mind="
            << mind_name(trace.mind) << " and code=" << code_name(trace.code)
            << ". Only one notch per wheel is active at a time.";
        return out.str();
    }

    if (contains_any(lower, {"steam engine", "pressure rail", "steam pressure"})) {
        out << "Steam telemetry is pressure, not personality. This turn is pressure " << percent(trace.steam.pressure)
            << ", heat " << percent(trace.steam.heat) << ", uncertainty " << percent(trace.steam.uncertainty)
            << ", tool-need " << percent(trace.steam.tool_need) << ". Pressure can trigger more verification; it does not choose policy.";
        return out.str();
    }

    if (contains_any(lower, {"aum", "create preserve transform"})) {
        out << "AUM governs change. This turn is " << aum_name(trace.aum)
            << ". In Hakui's runtime field the phases are A=emergence, U=sustain, M=return; in my cognition I use the same cycle to decide whether to create, preserve, or transform.";
        return out.str();
    }

    if (contains_any(lower, {"malt", "uncertain", "ambiguity"})) {
        out << "MALT state: " << malt_name(trace.malt) << ". Uncertainty is " << percent(trace.steam.uncertainty)
            << ". MALT is where I label ambiguity and seek evidence instead of filling the gap with confident text.";
        return out.str();
    }

    if (trace.project == Project::Hakui && contains_any(lower, {"what is hakui", "what do you know about hakui", "what do u know about hakui", "tell me about hakui", "know hakui"})) {
        return "Hakui is a native C++20 social/action world client on a dependency-free Spiral core with an SDL3 GPU host. Its verified core includes Router Bus, Route Table, Monolith Ledger, StateStore, Ether Bus, a 16-piston Steam Engine/Pressure Rail, Mind + Coding Octopus wheels, a 7x7 AUM field, and Crystal Grid/Host. The playable proof has third-person movement, skateboard, BMX, interactions, unarmed combat, avatar rigging, and Fusion tabletop systems; car simulation is explicitly deferred. I treat that as architecture evidence, not as permission to invent a current source line.";
    }

    if (trace.project == Project::Hakui && contains_any(lower, {"steam", "octopus", "aum field", "router bus", "statestore", "ether bus"})) {
        out << trace.evidence_summary << " Current cognition: mind=" << mind_name(trace.mind)
            << ", code=" << code_name(trace.code) << ", AUM=" << aum_name(trace.aum) << ".";
        return out.str();
    }

    if (trace.coding_question && trace.project == Project::Hakui) {
        if (trace.tool_required || trace.malt == MaltState::Malt) {
            out << "I know Hakui's verified architecture, but this question crosses into live-code evidence. "
                << "My route is " << mind_name(trace.mind) << "/" << code_name(trace.code)
                << ": inspect the relevant first-party C++ layer, trace Router Bus/StateStore boundaries, preserve the dependency firewall, then compile and run the matching CTest contract. "
                << "Bind the repo/source/compiler output and I can move from architecture reasoning to an exact file-level diagnosis.";
            return out.str();
        }
        out << "For Hakui C++, I would work through its existing laws instead of dropping logic into the SDL client: keep deterministic gameplay/core renderer-free, publish state through Router Bus/StateStore, use Ether Bus only for transitions, and add an executable CTest contract before presentation code. "
            << "This turn selected " << code_name(trace.code) << " under AUM=" << aum_name(trace.aum) << ".";
        return out.str();
    }

    if (trace.coding_question && trace.project == Project::General) {
        if (trace.malt == MaltState::Malt) {
            return "I can code natively, but the target is underspecified. Give me the project/file or paste the C++/compiler error; MALT is holding the action until the evidence target is clear.";
        }
        return "I can reason about C++ architecture, ownership, state machines, build failures, tests, and native runtime boundaries. For an exact patch I need the source or a bound repository/compiler tool; otherwise I can still design the implementation and invariants.";
    }

    if (trace.malt == MaltState::Failsafe) {
        return "That crosses into a destructive operation. I'm holding in FAILSAFE until the target and intended side effects are explicit.";
    }
    if (trace.malt == MaltState::Halt) return "Kernel halted for this turn.";

    return {};
}

std::string Kernel::project_name(Project value) {
    switch (value) {
        case Project::General: return "GENERAL";
        case Project::SpiralOS: return "SPIRAL_OS";
        case Project::Hakui: return "HAKUI";
        case Project::EtherPlay: return "ETHERPLAY";
        case Project::EtherBeat: return "ETHERBEAT";
    }
    return "GENERAL";
}

std::string Kernel::malt_name(MaltState value) {
    switch (value) {
        case MaltState::Pass: return "PASS";
        case MaltState::Malt: return "MALT";
        case MaltState::Failsafe: return "FAILSAFE";
        case MaltState::Halt: return "HALT";
    }
    return "PASS";
}

std::string Kernel::aum_name(AumMode value) {
    switch (value) {
        case AumMode::Create: return "CREATE";
        case AumMode::Preserve: return "PRESERVE";
        case AumMode::Transform: return "TRANSFORM";
    }
    return "PRESERVE";
}

std::string Kernel::mind_name(MindNotch value) {
    switch (value) {
        case MindNotch::Observe: return "OBSERVE";
        case MindNotch::Recall: return "RECALL";
        case MindNotch::Analyze: return "ANALYZE";
        case MindNotch::Diagnose: return "DIAGNOSE";
        case MindNotch::Design: return "DESIGN";
        case MindNotch::Verify: return "VERIFY";
        case MindNotch::Reflect: return "REFLECT";
        case MindNotch::Communicate: return "COMMUNICATE";
    }
    return "OBSERVE";
}

std::string Kernel::code_name(CodeNotch value) {
    switch (value) {
        case CodeNotch::Inspect: return "INSPECT";
        case CodeNotch::Trace: return "TRACE";
        case CodeNotch::Explain: return "EXPLAIN";
        case CodeNotch::Plan: return "PLAN";
        case CodeNotch::Implement: return "IMPLEMENT";
        case CodeNotch::Compile: return "COMPILE";
        case CodeNotch::Test: return "TEST";
        case CodeNotch::Repair: return "REPAIR";
    }
    return "INSPECT";
}

std::string Kernel::hologram(const Trace& trace) {
    std::ostringstream out;
    out << "// LIRATEL " << trace.liratel.source << '.' << trace.liratel.direction << "\n"
        << "// PROJECT " << project_name(trace.project) << "\n"
        << "// OCTOPUS MIND:" << mind_name(trace.mind) << " CODE:" << code_name(trace.code) << "\n"
        << "// AUM " << aum_name(trace.aum) << "  MALT " << malt_name(trace.malt) << "\n"
        << "// STEAM P=" << percent(trace.steam.pressure) << " U=" << percent(trace.steam.uncertainty) << "\n"
        << "// LAMBDA " << percent(trace.lambda.score) << (trace.lambda.stable ? " STABLE" : " UNSTABLE");
    return out.str();
}

} // namespace spiral::genius
