#include "spiral/genius_shell.hpp"

#include <iostream>
#include <string>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

int main(int argc, char** argv) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif

    spiral::genius::GeniusShell shell;

    if (argc >= 3 && std::string(argv[1]) == "--model") {
        std::string error;
        if (!shell.load_model(argv[2], &error)) {
            std::cerr << "MODEL LOAD FAILED: " << error << "\n";
        }
    }

    std::cout << shell.banner_text() << "\n\n";
    std::cout << shell.status_text() << "\n\n";

    std::string line;
    while (!shell.should_exit()) {
        std::cout << '[' << spiral::genius::shell_mode_name(shell.mode()) << "] spiral> " << std::flush;
        if (!std::getline(std::cin, line)) break;
        const std::string response = shell.handle_line(line);
        if (!response.empty()) std::cout << "\nspiral> " << response << "\n\n";
    }

    return 0;
}
