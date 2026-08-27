#include "spiral/genius_shell.hpp"

#include <cassert>
#include <iostream>
#include <string>

int main() {
    using namespace spiral::genius;

    GeniusShell shell;
    assert(shell.mode() == ShellMode::Gpt);
    assert(!shell.should_exit());

    const auto banner = shell.banner_text();
    assert(banner.find("SPIRAL AI GENIUS / L23 SHELL") != std::string::npos);
    assert(banner.find("GPT MODE") != std::string::npos);

    const auto status = shell.status_text();
    assert(status.find("mode: GPT") != std::string::npos);
    assert(status.find("language model: NOT LOADED") != std::string::npos);

    const auto no_model = shell.handle_line("hello spiral");
    assert(no_model.find("no trained Spiral language-model bundle") != std::string::npos);
    assert(no_model.find("will not fake intelligence") != std::string::npos);
    assert(shell.history().size() == 2);

    assert(shell.handle_line("/clear") == "conversation history cleared.");
    assert(shell.history().empty());

    const auto native_switch = shell.handle_line("/native");
    assert(native_switch.find("NATIVE MODE enabled") != std::string::npos);
    assert(shell.mode() == ShellMode::Native);
    const auto native_reply = shell.handle_line("probe this runtime");
    assert(native_reply.find("NATIVE MODE / substrate inspection") != std::string::npos);
    assert(native_reply.find("input bytes:") != std::string::npos);

    const auto gpt_switch = shell.handle_line("/gpt");
    assert(gpt_switch.find("GPT MODE enabled") != std::string::npos);
    assert(shell.mode() == ShellMode::Gpt);

    const auto temp = shell.handle_line("/temperature 0.35");
    assert(temp.find("0.35") != std::string::npos);
    assert(shell.status().temperature > 0.34F && shell.status().temperature < 0.36F);
    assert(shell.handle_line("/temperature 9") == "usage: /temperature <0.05..2.0>");

    assert(shell.handle_line("/max 64") == "max new tokens = 64");
    assert(shell.status().max_new_tokens == 64);
    assert(shell.handle_line("/max 99999") == "usage: /max <1..2048>");

    const auto gpu = shell.handle_line("/gpu");
    assert(gpu.find("GPU:") != std::string::npos);

    const auto memory = shell.handle_line("/memory");
    assert(memory.find("MemoryStore") != std::string::npos);
    const auto agent = shell.handle_line("/agent");
    assert(agent.find("AgentEngine") != std::string::npos);
    const auto render = shell.handle_line("/render");
    assert(render.find("framebuffer") != std::string::npos);

    const auto help = shell.handle_line("/help");
    assert(help.find("/gpt") != std::string::npos);
    assert(help.find("/load <bundle>") != std::string::npos);

    assert(shell.handle_line("/exit") == "Spiral sleeping.");
    assert(shell.should_exit());

    std::cout << "Genius shell GPT mode regression passed\n";
    return 0;
}
