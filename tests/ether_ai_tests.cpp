#include "spiral/ether_ai.hpp"

#include <cassert>
#include <iostream>
#include <string>

int main() {
    using namespace spiral;

    ether_ai::Runtime runtime(ether_ai::standalone_host());
    auto status = runtime.status();
    assert(status.host.kind == ether_ai::HostKind::StandaloneWindows);
    assert(status.host.name == "Spiral Ether AI");
    assert(status.shell.mode == genius::ShellMode::Gpt);
    assert(status.shell.gpt_backend == genius::GptBackend::Auto);

    runtime.set_backend(genius::GptBackend::SpiralLocal);
    const std::string reply = runtime.send("hello");
    assert(reply.find("no trained language-model bundle") != std::string::npos);
    auto history = runtime.history();
    assert(history.size() == 2);
    assert(history[0].role == "user");
    assert(history[0].content == "hello");
    assert(history[1].role == "assistant");

    runtime.set_host(ether_ai::etherplay_host());
    status = runtime.status();
    assert(status.host.kind == ether_ai::HostKind::EtherPlay);
    assert(status.host.name == "EtherPlay");
    assert(runtime.host().kind == ether_ai::HostKind::EtherPlay);

    runtime.set_host(ether_ai::hakui_host());
    assert(runtime.host().kind == ether_ai::HostKind::Hakui);
    assert(ether_ai::host_kind_name(runtime.host().kind) == "HAKUI");

    runtime.set_host(ether_ai::etherbeat_host());
    assert(runtime.host().kind == ether_ai::HostKind::EtherBeat);

    runtime.clear();
    assert(runtime.history().empty());

    std::cout << "Ether AI portable runtime tests passed\n";
    return 0;
}
