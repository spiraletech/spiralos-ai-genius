#include "spiral/xenon_os.hpp"

#include <cassert>
#include <string>

int main() {
    using namespace spiral::xenon;

    ToolBus bus = make_default_tool_bus();
    assert(bus.contains("hakui.inspect_world"));
    assert(bus.contains("etherplay.analyze_audio"));
    assert(bus.contains("etherbeat.get_arrangement"));
    assert(bus.capabilities().size() >= 9);

    ToolIntent intent;
    intent.host = "etherplay";
    intent.action = "analyze_audio";
    intent.arguments["track"] = "test.wav";
    const auto result = bus.dispatch(intent);
    assert(result.success);
    assert(result.data.at("host") == "etherplay");
    assert(result.data.at("capability") == "analyze_audio");

    SpiralContext context;
    context.host = "ETHERPLAY";
    context.host_context = "audio host";
    context.local_datetime = "Sunday, 2026-08-30 02:00:00";
    context.organic_topic = "music";
    context.recent_turns.push_back({"user", "remember BlueCube"});
    const std::string prompt = build_cortex_prompt(context, "what day is it?");
    assert(prompt.find("2026-08-30") != std::string::npos);
    assert(prompt.find("ORGANIC") != std::string::npos);
    assert(prompt.find("what day is it?") != std::string::npos);

    LocalCortex cortex;
    std::string error;
    assert(!cortex.configure_gguf("not-a-model.bundle", {}, &error));
    assert(!error.empty());

    return 0;
}
