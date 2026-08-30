#include "spiral/ether_ai.hpp"

#include <cassert>
#include <filesystem>
#include <iostream>
#include <string>

int main() {
    using namespace spiral;

    const std::filesystem::path state_path = "spiral_ether_ai_runtime_test.organic";
    std::filesystem::remove(state_path);
    std::filesystem::remove(state_path.string() + ".memory");

    ether_ai::Runtime runtime(ether_ai::standalone_host(), state_path.string());
    auto status = runtime.status();
    assert(status.host.kind == ether_ai::HostKind::StandaloneWindows);
    assert(status.shell.mode == genius::ShellMode::Gpt);

    if (runtime.backend() == genius::GptBackend::Auto) {
        const std::string hello = runtime.send("hello spiral");
        assert(hello.find("LANGUAGE CORTEX: OFFLINE / LIMITED MODE") == std::string::npos);
        assert(hello.find("Organic mode is running locally") != std::string::npos);
        assert(runtime.status().shell.organic_turns == 1);
        assert(runtime.status().shell.organic_memories >= 1);

        const std::string sup = runtime.send("sup");
        assert(sup.find("LANGUAGE CORTEX: OFFLINE / LIMITED MODE") == std::string::npos);
    }

    const std::string day = runtime.send("what day is it?");
    assert(day.rfind("Today is ", 0) == 0);

    // Reproduce the exact Windows composer edge case: case changes + surrounding CR/LF/whitespace.
    const std::string day_windows = runtime.send("  WHAT DAY IS IT \r\n");
    assert(day_windows.rfind("Today is ", 0) == 0);
    assert(day_windows.find("native learned knowledge") == std::string::npos);

    const std::string date_windows = runtime.send("WHAT DATE IS IT???\r\n");
    assert(date_windows.rfind("Today is ", 0) == 0);

    auto history = runtime.history();
    assert(history.size() >= 2);
    assert(history[history.size() - 2].role == "user");
    assert(history.back().role == "assistant");

    runtime.set_host(ether_ai::etherplay_host());
    assert(runtime.host().kind == ether_ai::HostKind::EtherPlay);
    runtime.set_host(ether_ai::hakui_host());
    assert(ether_ai::host_kind_name(runtime.host().kind) == "HAKUI");
    runtime.set_host(ether_ai::etherbeat_host());
    assert(runtime.host().kind == ether_ai::HostKind::EtherBeat);

    const auto tools = runtime.tool_capabilities();
    assert(tools.size() >= 20);

    xenon::ToolIntent write{"etherbeat", "generate_midi", {}};
    const auto blocked = runtime.dispatch_tool(write, false);
    assert(!blocked.success);
    assert(blocked.message.find("permission gate") != std::string::npos);

    const std::size_t memories_before_clear = runtime.status().shell.organic_memories;
    runtime.clear();
    assert(runtime.history().empty());
    assert(runtime.status().shell.organic_memories == memories_before_clear);

    runtime.reset_organic_state();
    assert(runtime.status().shell.organic_turns == 0);
    assert(runtime.status().shell.organic_memories == 0);

    std::filesystem::remove(state_path);
    std::filesystem::remove(state_path.string() + ".memory");
    std::cout << "Ether AI L27D HF3 conversational routing tests passed\n";
    return 0;
}