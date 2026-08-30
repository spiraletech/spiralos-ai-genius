#include "spiral/genius_shell.hpp"

#include <cassert>
#include <filesystem>
#include <iostream>
#include <string>

int main() {
    using namespace spiral::genius;

    const std::filesystem::path state_path = "spiral_shell_organic_test.bin";
    std::filesystem::remove(state_path);
    std::filesystem::remove(state_path.string() + ".memory");

    GeniusShell shell;
    shell.set_organic_state_path(state_path.string(), false);
    assert(shell.mode() == ShellMode::Gpt);
    assert(shell.gpt_backend() == GptBackend::Auto);
    assert(gpt_backend_name(shell.gpt_backend()) == "ORGANIC");
    assert(!shell.should_exit());

    const auto banner = shell.banner_text();
    assert(banner.find("SPIRAL ETHER AI / ORGANIC STATE") != std::string::npos);
    assert(banner.find("without an API key") != std::string::npos);

    const auto status = shell.status_text();
    assert(status.find("mode: CHAT") != std::string::npos);
    assert(status.find("brain: ORGANIC") != std::string::npos);
    assert(status.find("organic: ONLINE") != std::string::npos);

    // Default chat is entirely native and must evolve organic state without any API/model.
    const auto organic_reply = shell.handle_line("hello spiral");
    assert(!organic_reply.empty());
    assert(shell.history().size() == 2);
    assert(shell.status().organic_turns == 1);
    assert(shell.status().organic_memories == 1);
    assert(std::filesystem::exists(state_path));

    const auto mind = shell.handle_line("/organic");
    assert(mind.find("ORGANIC MIND / ONLINE") != std::string::npos);
    assert(mind.find("memories: 1") != std::string::npos);

    // Local cortex is explicit and never silently replaces the organic default.
    assert(shell.handle_line("/backend spiral") == "brain = SPIRAL_LOCAL_CORTEX");
    const auto no_model = shell.handle_line("hello local cortex");
    assert(no_model.find("no trained language-model bundle") != std::string::npos);
    assert(shell.status().organic_turns == 1);

    assert(shell.handle_line("/clear") == "visible conversation history cleared; organic durable memory preserved.");
    assert(shell.history().empty());
    assert(shell.status().organic_memories == 1);

    assert(shell.handle_line("/backend organic") == "brain = ORGANIC");
    assert(shell.handle_line("/backend openai") == "brain = OPENAI_BRIDGE (explicit network mode)");
    assert(shell.handle_line("/backend") == "brain = OPENAI_BRIDGE");
    assert(shell.handle_line("/backend spiral") == "brain = SPIRAL_LOCAL_CORTEX");
    assert(shell.handle_line("/backend nonsense") == "usage: /backend organic|spiral|openai");

    const auto openai = shell.handle_line("/openai");
    assert(openai.find("OPENAI BRIDGE:") != std::string::npos);
    assert(openai.find("optional") != std::string::npos);
    assert(shell.handle_line("/gptmodel gpt-test") == "OpenAI bridge model = gpt-test");
    assert(shell.handle_line("/gptmodel") == "OpenAI bridge model = gpt-test");

    const auto native_switch = shell.handle_line("/native");
    assert(native_switch.find("NATIVE MODE enabled") != std::string::npos);
    assert(shell.mode() == ShellMode::Native);
    const auto native_reply = shell.handle_line("probe this runtime");
    assert(native_reply.find("NATIVE MODE / substrate inspection") != std::string::npos);
    assert(native_reply.find("organic mind: ONLINE") != std::string::npos);

    const auto chat_switch = shell.handle_line("/gpt");
    assert(chat_switch.find("CHAT MODE enabled") != std::string::npos);
    assert(shell.mode() == ShellMode::Gpt);

    const auto temp = shell.handle_line("/temperature 0.35");
    assert(temp.find("0.35") != std::string::npos);
    assert(shell.status().temperature > 0.34F && shell.status().temperature < 0.36F);
    assert(shell.handle_line("/temperature 9") == "usage: /temperature <0.05..2.0>");

    assert(shell.handle_line("/max 64") == "local cortex max new tokens = 64");
    assert(shell.status().max_new_tokens == 64);
    assert(shell.handle_line("/max 99999") == "usage: /max <1..2048>");

    const auto gpu = shell.handle_line("/gpu");
    assert(gpu.find("GPU:") != std::string::npos);
    const auto memory = shell.handle_line("/memory");
    assert(memory.find("ORGANIC MEMORY") != std::string::npos);
    const auto agent = shell.handle_line("/agent");
    assert(agent.find("AgentEngine") != std::string::npos);
    const auto render = shell.handle_line("/render");
    assert(render.find("native Spiral Ether AI window") != std::string::npos);

    const auto help = shell.handle_line("/help");
    assert(help.find("organic|spiral|openai") != std::string::npos);
    assert(help.find("/organic") != std::string::npos);
    assert(help.find("/load <bundle>") != std::string::npos);

    assert(shell.handle_line("/resetorganic") == "organic state + durable memory reset.");
    assert(shell.status().organic_turns == 0);
    assert(shell.status().organic_memories == 0);

    assert(shell.handle_line("/exit") == "Spiral state saved. Sleeping.");
    assert(shell.should_exit());

    std::filesystem::remove(state_path);
    std::filesystem::remove(state_path.string() + ".memory");

    std::cout << "Genius shell organic-state regression passed\n";
    return 0;
}
