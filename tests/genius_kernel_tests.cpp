#include "spiral/ether_ai.hpp"
#include "spiral/genius_kernel.hpp"

#include <cassert>
#include <iostream>
#include <string>

int main() {
    using namespace spiral;

    genius::Kernel kernel;
    genius::Context context;
    context.host_context = "Spiral Ether AI standalone Windows host";
    context.coherence = 0.82F;
    context.confidence = 0.72F;
    context.focus = 0.79F;
    context.curiosity = 0.77F;
    context.memory_count = 12;

    const auto hakui_trace = kernel.evaluate("what do u know about hakui", context);
    assert(hakui_trace.project == genius::Project::Hakui);
    assert(hakui_trace.project_profile_verified);
    assert(hakui_trace.malt == genius::MaltState::Pass);
    assert(hakui_trace.liratel.source == "AETH" || hakui_trace.liratel.source == "VEYN");
    assert(hakui_trace.lambda.score > 0.60F);

    const std::string hakui = kernel.answer("what do u know about hakui", context, hakui_trace);
    assert(hakui.find("C++20") != std::string::npos);
    assert(hakui.find("Router Bus") != std::string::npos);
    assert(hakui.find("Steam Engine") != std::string::npos);
    assert(hakui.find("Octopus") != std::string::npos);
    assert(hakui.find("AUM") != std::string::npos);
    assert(hakui.find("BMX") != std::string::npos);

    const auto code_trace = kernel.evaluate("how should we code a new interaction in hakui", context);
    assert(code_trace.project == genius::Project::Hakui);
    assert(code_trace.coding_question);
    assert(code_trace.mind == genius::MindNotch::Design || code_trace.mind == genius::MindNotch::Analyze);
    assert(code_trace.code == genius::CodeNotch::Implement || code_trace.code == genius::CodeNotch::Explain);
    assert(code_trace.aum == genius::AumMode::Create);

    const std::string code = kernel.answer("how should we code a new interaction in hakui", context, code_trace);
    assert(code.find("StateStore") != std::string::npos);
    assert(code.find("Router Bus") != std::string::npos);
    assert(code.find("CTest") != std::string::npos);
    assert(code.find("dependency") != std::string::npos);

    const auto live_trace = kernel.evaluate("inspect repo and fix the current hakui source file bug", context);
    assert(live_trace.project == genius::Project::Hakui);
    assert(live_trace.tool_required);
    const std::string live = kernel.answer("inspect repo and fix the current hakui source file bug", context, live_trace);
    assert(live.find("live-code evidence") != std::string::npos);
    assert(live.find("exact file-level diagnosis") != std::string::npos);

    const std::string hologram = genius::Kernel::hologram(code_trace);
    assert(hologram.find("LIRATEL") != std::string::npos);
    assert(hologram.find("OCTOPUS") != std::string::npos);
    assert(hologram.find("AUM") != std::string::npos);
    assert(hologram.find("LAMBDA") != std::string::npos);

    // Full portable app path: ORGANIC must be authored by the Genius kernel.
    ether_ai::Runtime runtime(ether_ai::standalone_host(), "");
    runtime.reset_organic_state();

    const std::string hello = runtime.send("how r u");
    assert(hello.find("online and coherent") != std::string::npos);
    assert(hello.find("don't have enough native learned knowledge") == std::string::npos);

    const std::string runtime_hakui = runtime.send("what do u know about hakui");
    assert(runtime_hakui.find("native C++20") != std::string::npos);
    assert(runtime_hakui.find("Octopus") != std::string::npos);

    const auto status = runtime.status();
    assert(status.shell.cognition_project == "HAKUI");
    assert(!status.shell.cognition_mind.empty());
    assert(!status.shell.cognition_code.empty());
    assert(status.shell.cognition_lambda > 0.0F);

    const std::string trace = runtime.command("/trace");
    assert(trace.find("PROJECT HAKUI") != std::string::npos);
    assert(trace.find("LIRATEL") != std::string::npos);
    assert(trace.find("LAMBDA") != std::string::npos);

    const std::string need = runtime.send("what do u need to know");
    assert(need.find("project target") != std::string::npos || need.find("live source") != std::string::npos);

    std::cout << "Liratel Genius cognition regression passed\n";
    return 0;
}
