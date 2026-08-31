#include "spiral/ether_ai.hpp"
#include "spiral/gguf.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

namespace {

struct Options {
    std::filesystem::path model;
    std::filesystem::path runtime;
    std::string prompt;
    bool one_shot = false;
};

std::filesystem::path executable_directory() {
#ifdef _WIN32
    std::wstring buffer(32768, L'\0');
    const DWORD length = GetModuleFileNameW(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
    if (length != 0 && length < buffer.size()) {
        buffer.resize(length);
        return std::filesystem::path(buffer).parent_path();
    }
#endif
    std::error_code error;
    const auto current = std::filesystem::current_path(error);
    return error ? std::filesystem::path{} : current;
}

std::optional<std::filesystem::path> discover_model(const std::filesystem::path& root) {
    const std::filesystem::path qwen = root / "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf";
    if (std::filesystem::exists(qwen)) return qwen;
    const std::filesystem::path preferred = root / "SmolLM2-135M-Instruct-Q4_K_M.gguf";
    if (std::filesystem::exists(preferred)) return preferred;
    const std::filesystem::path generic = root / "SpiralCortex.gguf";
    if (std::filesystem::exists(generic)) return generic;
    std::error_code error;
    for (const auto& entry : std::filesystem::directory_iterator(root, error)) {
        if (error) break;
        if (!entry.is_regular_file()) continue;
        std::string extension = entry.path().extension().string();
        std::transform(extension.begin(), extension.end(), extension.begin(),
            [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
        if (extension == ".gguf") return entry.path();
    }
    return std::nullopt;
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string_view argument(argv[i]);
        if (argument == "--model" && i + 1 < argc) options.model = argv[++i];
        else if (argument == "--runtime" && i + 1 < argc) options.runtime = argv[++i];
        else if (argument == "--prompt" && i + 1 < argc) { options.prompt = argv[++i]; options.one_shot = true; }
        else if (!argument.starts_with("--") && options.model.empty()) options.model = argv[i];
        else throw std::runtime_error("unknown or incomplete argument: " + std::string(argument));
    }
    return options;
}

void configure_runtime_path(const std::filesystem::path& path) {
    if (path.empty()) return;
#ifdef _WIN32
    _putenv_s("SPIRAL_LLAMA_EXE", path.string().c_str());
#else
    setenv("SPIRAL_LLAMA_EXE", path.string().c_str(), 1);
#endif
}

void print_model_summary(const spiral::gguf::ModelFile& model, const std::filesystem::path& path) {
    std::cout << "model: " << path.filename().string() << '\n';
    if (const auto* architecture = model.find("general.architecture"))
        std::cout << "architecture: " << spiral::gguf::value_summary(*architecture) << '\n';
    if (const auto* name = model.find("general.name"))
        std::cout << "name: " << spiral::gguf::value_summary(*name) << '\n';
    std::cout << "GGUF v" << model.version << " | " << model.tensors.size() << " tensors\n";
}

bool load_model(spiral::ether_ai::Runtime& runtime, spiral::gguf::Reader& reader,
                const std::filesystem::path& path, std::string& error) {
    if (!reader.open(path, &error)) return false;
    return runtime.load_local_model(path.string(), &error);
}

void print_help() {
    std::cout <<
        "Commands:\n"
        "  /help                 show this command list\n"
        "  /status               show cortex, memory, and runtime status\n"
        "  /inspect              show validated GGUF model information\n"
        "  /load <model.gguf>    switch to another local model\n"
        "  /temperature <n>      set sampling temperature (0.05..2.0)\n"
        "  /max <n>              set maximum reply tokens (1..2048)\n"
        "  /system <text>        set additional system context\n"
        "  /history              show this session's conversation\n"
        "  /save <file.txt>      save the conversation transcript\n"
        "  /clear                clear visible conversation history\n"
        "  /memory               inspect durable ORGANIC memory\n"
        "  /tools                list XENON tool capabilities\n"
        "  /exit                 close the shell\n";
}

std::string trim(std::string value) {
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return {};
    const auto last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

std::pair<std::string, std::string> split_command(const std::string& line) {
    const auto split = line.find_first_of(" \t");
    if (split == std::string::npos) return {line, {}};
    return {line.substr(0, split), trim(line.substr(split + 1))};
}

void print_history(const spiral::ether_ai::Runtime& runtime, std::ostream& out) {
    const auto history = runtime.history();
    if (history.empty()) { out << "No conversation yet.\n"; return; }
    for (const auto& turn : history)
        out << (turn.role == "assistant" ? "Spiral: " : "You: ") << turn.content << "\n\n";
}

bool handle_command(const std::string& line, spiral::ether_ai::Runtime& runtime,
                    spiral::gguf::Reader& reader, std::filesystem::path& model_path,
                    std::string& system_context) {
    const auto [command, argument] = split_command(line);
    if (command == "/exit" || command == "/quit") return false;
    if (command == "/help" || command == "/?") print_help();
    else if (command == "/clear") { runtime.clear(); std::cout << "Conversation cleared; durable memory preserved.\n"; }
    else if (command == "/history") print_history(runtime, std::cout);
    else if (command == "/status") std::cout << runtime.command("/xenon") << '\n' << runtime.command("/memory") << '\n';
    else if (command == "/tools") std::cout << runtime.command("/tools") << '\n';
    else if (command == "/memory") std::cout << runtime.command("/memory") << '\n';
    else if (command == "/inspect") print_model_summary(reader.model(), model_path);
    else if (command == "/load") {
        if (argument.empty()) std::cout << "Usage: /load <model.gguf>\n";
        else {
            std::string error;
            const std::filesystem::path candidate(argument);
            if (!load_model(runtime, reader, candidate, error)) std::cout << "Model load failed: " << error << '\n';
            else { model_path = candidate; std::cout << "Loaded "; print_model_summary(reader.model(), model_path); }
        }
    } else if (command == "/temperature") {
        try {
            const float value = std::stof(argument);
            if (value < 0.05F || value > 2.0F) throw std::out_of_range("temperature");
            runtime.configure_local_generation(runtime.local_max_new_tokens(), value);
            std::cout << "Temperature: " << value << '\n';
        } catch (...) { std::cout << "Usage: /temperature <0.05..2.0>\n"; }
    } else if (command == "/max") {
        try {
            const auto value = static_cast<std::size_t>(std::stoull(argument));
            if (value == 0 || value > 2048) throw std::out_of_range("max");
            runtime.configure_local_generation(value, runtime.local_temperature());
            std::cout << "Maximum reply tokens: " << value << '\n';
        } catch (...) { std::cout << "Usage: /max <1..2048>\n"; }
    } else if (command == "/system") {
        if (argument.empty()) std::cout << "System context: " << (system_context.empty() ? "default" : system_context) << '\n';
        else {
            system_context = argument;
            auto host = spiral::ether_ai::standalone_host();
            host.context = system_context;
            runtime.set_host(std::move(host));
            std::cout << "System context updated.\n";
        }
    } else if (command == "/save") {
        if (argument.empty()) std::cout << "Usage: /save <file.txt>\n";
        else {
            std::ofstream output(argument, std::ios::binary | std::ios::trunc);
            if (!output) std::cout << "Could not create transcript file.\n";
            else { print_history(runtime, output); std::cout << "Saved transcript to " << argument << '\n'; }
        }
    } else std::cout << "Unknown command. Use /help.\n";
    return true;
}

} // namespace

int main(int argc, char** argv) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
    DWORD mode = 0;
    const HANDLE output = GetStdHandle(STD_OUTPUT_HANDLE);
    if (GetConsoleMode(output, &mode)) SetConsoleMode(output, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
#endif
    try {
        Options options = parse_options(argc, argv);
        const auto root = executable_directory();
        if (options.runtime.empty()) options.runtime = root / "llama-cli.exe";
        configure_runtime_path(options.runtime);
        if (options.model.empty()) {
            const auto discovered = discover_model(root);
            if (discovered) options.model = *discovered;
        }
        if (options.model.empty()) {
            std::cerr << "No GGUF model found. Put a model beside SpiralAIShell.exe or use --model <file.gguf>.\n";
            return 2;
        }
        if (!std::filesystem::exists(options.runtime)) {
            std::cerr << "Local inference runtime not found: " << options.runtime.string() << '\n';
            return 2;
        }

        spiral::ether_ai::Runtime runtime(spiral::ether_ai::standalone_host(), (root / "SpiralAIShell.organic").string());
        runtime.configure_local_generation(384, 0.62F);
        spiral::gguf::Reader reader;
        std::string error;
        if (!load_model(runtime, reader, options.model, error)) {
            std::cerr << "Model load failed: " << error << '\n';
            return 2;
        }

        if (options.one_shot) {
            const std::string reply = runtime.send(options.prompt);
            if (reply.empty() || reply.starts_with("LANGUAGE CORTEX ERROR")) {
                std::cerr << (reply.empty() ? "The model returned no reply." : reply) << '\n';
                return 1;
            }
            std::cout << reply << '\n';
            return 0;
        }

        std::cout << "\x1b[1;36mSPIRAL AI SHELL\x1b[0m / local conversational cortex\n";
        print_model_summary(reader.model(), options.model);
        std::cout << "Persistent ORGANIC memory: online | XENON tools: online\n"
                     "Type normally to chat. Use /help for commands.\n\n";

        std::filesystem::path model_path = options.model;
        std::string system_context;
        std::string line;
        while (true) {
            std::cout << "\x1b[1;32mYou >\x1b[0m " << std::flush;
            if (!std::getline(std::cin, line)) break;
            line = trim(std::move(line));
            if (line.empty()) continue;
            if (line.front() == '/') {
                if (!handle_command(line, runtime, reader, model_path, system_context)) break;
                std::cout << '\n';
                continue;
            }
            std::cout << "\x1b[2mThinking...\x1b[0m\r" << std::flush;
            const std::string reply = runtime.send(line);
            std::cout << "\x1b[2K\r\x1b[1;36mSpiral >\x1b[0m " << reply << "\n\n";
        }
        std::cout << "Spiral state saved.\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "Spiral AI Shell error: " << exception.what() << '\n';
        return 2;
    }
}
