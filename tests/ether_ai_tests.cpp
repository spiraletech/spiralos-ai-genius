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
    assert(status.host.name == "Spiral Ether AI");
    assert(status.shell.mode == genius::ShellMode::Gpt);
    assert(status.shell.gpt_backend == genius::GptBackend::Auto);
    assert(genius::gpt_backend_name(status.shell.gpt_backend) == "ORGANIC");
    assert(status.shell.organic_turns == 0);

    const std::string organic_reply = runtime.send("hello spiral");
    assert(!organic_reply.empty());
    status = runtime.status();
    assert(status.shell.organic_turns == 1);
    assert(status.shell.organic_memories == 1);
    assert(std::filesystem::exists(state_path));
    assert(std::filesystem::exists(state_path.string() + ".memory"));

    auto history = runtime.history();
    assert(history.size() == 2);
    assert(history[0].role == "user");
    assert(history[0].content == "hello spiral");
    assert(history[1].role == "assistant");

    // App-level regression for the exact bad responses reported from the Windows UI.
    const std::string state_reply = runtime.send("what is the local state");
    assert(state_reply.find("My local organic state right now") != std::string::npos);
    assert(state_reply.find("focus") != std::string::npos);
    assert(state_reply.find("For broad factual knowledge") == std::string::npos);

    const std::string processor_reply = runtime.send("what is ur processor");
    assert(processor_reply.find("native C++") != std::string::npos);
    assert(processor_reply.find("CPU") != std::string::npos);
    assert(processor_reply.find("what is the local state") == std::string::npos);

    runtime.set_backend(genius::GptBackend::SpiralLocal);
    const std::string local_reply = runtime.send("hello local cortex");
    assert(local_reply.find("no trained language-model bundle") != std::string::npos);
    assert(runtime.status().shell.organic_turns == 3);

    runtime.set_backend(genius::GptBackend::Auto);
    runtime.set_host(ether_ai::etherplay_host());
    status = runtime.status();
    assert(status.host.kind == ether_ai::HostKind::EtherPlay);
    assert(status.host.name == "EtherPlay");
    assert(runtime.host().kind == ether_ai::HostKind::EtherPlay);
    const std::string host_reply = runtime.send("what host are you running in?");
    assert(host_reply.find("EtherPlay") != std::string::npos);
    assert(runtime.status().shell.organic_turns == 4);

    runtime.set_host(ether_ai::hakui_host());
    assert(runtime.host().kind == ether_ai::HostKind::Hakui);
    assert(ether_ai::host_kind_name(runtime.host().kind) == "HAKUI");

    runtime.set_host(ether_ai::etherbeat_host());
    assert(runtime.host().kind == ether_ai::HostKind::EtherBeat);

    const std::size_t memories_before_clear = runtime.status().shell.organic_memories;
    runtime.clear();
    assert(runtime.history().empty());
    assert(runtime.status().shell.organic_memories == memories_before_clear);

    runtime.reset_organic_state();
    assert(runtime.status().shell.organic_turns == 0);
    assert(runtime.status().shell.organic_memories == 0);

    std::filesystem::remove(state_path);
    std::filesystem::remove(state_path.string() + ".memory");

    std::cout << "Ether AI organic portable runtime response tests passed\n";
    return 0;
}
