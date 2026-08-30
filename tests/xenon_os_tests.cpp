#include "spiral/xenon_os.hpp"

#include <cassert>
#include <string>

int main() {
    using namespace spiral::xenon;

    ToolBus bus = make_default_tool_bus();
    assert(bus.contains("hakui.inspect_world"));
    assert(bus.contains("hakui.move_to"));
    assert(bus.contains("etherplay.analyze_audio"));
    assert(bus.contains("etherplay.seek"));
    assert(bus.contains("etherbeat.get_arrangement"));
    assert(bus.contains("etherbeat.generate_midi"));
    assert(bus.contains("etherbeat.export_song"));
    assert(bus.capabilities().size() >= 20);

    ToolIntent write;
    write.host = "etherbeat";
    write.action = "generate_midi";
    const auto blocked = bus.dispatch(write);
    assert(!blocked.success);
    assert(blocked.message.find("permission gate") != std::string::npos);

    EngineBridge bridge;
    const std::string endpoint = EngineBridge::endpoint_for("etherplay");
    assert(endpoint.find("etherplay") != std::string::npos);
    const auto probe = bridge.probe("etherplay");
    assert(probe.host == "etherplay");
    assert(!probe.endpoint.empty());

    ToolIntent read;
    read.host = "etherplay";
    read.action = "analyze_audio";
    read.arguments["track"] = "test.wav";
    const auto result = bus.dispatch(read);
    assert(result.data.at("tool") == "etherplay.analyze_audio");
    // A CI runner normally has no EtherPlay pipe server, so failure is truthful.
    if (!result.success) assert(result.message.find("offline") != std::string::npos || result.message.find("unavailable") != std::string::npos);

    SpiralContext context;
    context.host = "ETHERPLAY";
    context.host_context = "audio host";
    context.local_datetime = "Sunday, 2026-08-30 02:00:00";
    context.organic_topic = "music";
    context.recent_turns.push_back({"user", "remember BlueCube"});
    context.recent_tool_results.push_back(result);
    const std::string prompt = build_cortex_prompt(context, "what day is it?");
    assert(prompt.find("2026-08-30") != std::string::npos);
    assert(prompt.find("ORGANIC") != std::string::npos);
    assert(prompt.find("what day is it?") != std::string::npos);
    assert(prompt.find("TOOL:") != std::string::npos);

    LocalCortex cortex;
    std::string error;
    assert(!cortex.configure_gguf("not-a-model.bundle", {}, &error));
    assert(!error.empty());

    return 0;
}
