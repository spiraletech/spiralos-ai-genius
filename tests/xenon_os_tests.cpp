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

    ToolIntent write{"etherbeat","generate_midi",{}};
    const auto blocked = bus.dispatch(write);
    assert(!blocked.success);
    assert(blocked.message.find("permission gate") != std::string::npos);

    EngineBridge bridge;
    const std::string endpoint = EngineBridge::endpoint_for("etherplay");
    assert(endpoint.find("etherplay") != std::string::npos);
    const auto probe = bridge.probe("etherplay");
    assert(probe.host == "etherplay");
    assert(!probe.endpoint.empty());

    ToolIntent read{"etherplay","analyze_audio",{{"track","test.wav"}}};
    const auto result = bus.dispatch(read);
    assert(result.data.at("tool") == "etherplay.analyze_audio");
    if (!result.success) assert(result.message.find("offline") != std::string::npos || result.message.find("unavailable") != std::string::npos);

    SpiralContext context;
    context.host = "ETHERPLAY";
    context.host_context = "audio host";
    context.local_datetime = "Sunday, August 30, 2026 03:12:00 AM";
    context.organic_topic = "music";
    context.relevant_memories.push_back("The test object is called BlueCube.");
    context.recent_turns.push_back({"user","remember BlueCube"});
    context.recent_tool_results.push_back(result);
    const std::string prompt = build_cortex_prompt(context,"what is my test object called?");
    assert(prompt.find("BlueCube") != std::string::npos);
    assert(prompt.find("ORGANIC") != std::string::npos);
    assert(prompt.find("TOOL_RESULT:") != std::string::npos);
    assert(prompt.find("TOOL_CALL") != std::string::npos);
    assert(prompt.find("A greeting needs only a friendly greeting") != std::string::npos);
    assert(prompt.find("Never invent packages") != std::string::npos);

    const auto call = parse_tool_call("TOOL_CALL etherplay.analyze_audio track=test.wav mode=full");
    assert(call.has_value());
    assert(call->host == "etherplay");
    assert(call->action == "analyze_audio");
    assert(call->arguments.at("track") == "test.wav");

    const std::string cleaned = clean_cortex_output("llama_model_loader: noise\nmain: noise\nASSISTANT: Specific useful answer.\n");
    assert(cleaned == "Specific useful answer.");
    const std::string single_turn = clean_cortex_output(
        "build : test\n> USER: hello\nASSISTANT:\nUsable shell answer.\n\n[ Prompt: 500 t/s | Generation: 100 t/s ]\nExiting...\n");
    assert(single_turn == "Usable shell answer.");
    const std::string truncated_prompt = clean_cortex_output(
        "build : test\n> SYSTEM: long prompt ... (truncated)\nClean generated answer.\n[ Prompt: 500 t/s ]\n");
    assert(truncated_prompt == "Clean generated answer.");

    const std::string date = current_local_date_answer();
    assert(date.rfind("Today is ",0) == 0);

    LocalCortex cortex;
    assert(cortex.state() == CortexState::Offline);
    std::string error;
    assert(!cortex.configure_gguf("not-a-model.bundle",{},&error));
    assert(!error.empty());

    return 0;
}
