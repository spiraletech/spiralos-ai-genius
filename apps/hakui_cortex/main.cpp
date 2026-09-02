#include "spiral/ether_ai.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace {

constexpr std::array<char, 8> kRequestMagic{{'S','P','H','K','U','I','0','1'}};
constexpr std::array<char, 8> kResponseMagic{{'S','P','H','R','E','P','0','1'}};
constexpr std::uint32_t kPing = 1;
constexpr std::uint32_t kAsk = 2;
constexpr std::uint32_t kResponseOk = 0;
constexpr std::uint32_t kResponseBadRequest = 1;
constexpr std::uint32_t kResponseRuntimeError = 2;
constexpr std::uint32_t kFlagLocalModelLoaded = 1U << 0U;
constexpr std::size_t kMaxContextBytes = 64U * 1024U;
constexpr std::size_t kMaxPromptBytes = 8U * 1024U;
constexpr std::size_t kMaxReplyBytes = 1024U * 1024U;
constexpr std::uint16_t kDefaultPort = 47691;

#ifdef _WIN32
using SocketHandle = SOCKET;
constexpr SocketHandle kInvalidSocket = INVALID_SOCKET;
#else
using SocketHandle = int;
constexpr SocketHandle kInvalidSocket = -1;
#endif

void close_socket(SocketHandle socket) noexcept {
    if (socket == kInvalidSocket) return;
#ifdef _WIN32
    closesocket(socket);
#else
    close(socket);
#endif
}

class SocketRuntime final {
public:
    SocketRuntime() {
#ifdef _WIN32
        WSADATA data{};
        ok_ = WSAStartup(MAKEWORD(2, 2), &data) == 0;
#else
        ok_ = true;
#endif
    }

    ~SocketRuntime() {
#ifdef _WIN32
        if (ok_) WSACleanup();
#endif
    }

    SocketRuntime(const SocketRuntime&) = delete;
    SocketRuntime& operator=(const SocketRuntime&) = delete;

    [[nodiscard]] bool ok() const noexcept { return ok_; }

private:
    bool ok_ = false;
};

bool send_all(SocketHandle socket, const void* data, std::size_t size) {
    const auto* bytes = static_cast<const char*>(data);
    std::size_t sent = 0;
    while (sent < size) {
        const std::size_t remaining = size - sent;
        const int chunk = static_cast<int>(std::min<std::size_t>(
            remaining,
            static_cast<std::size_t>(std::numeric_limits<int>::max())
        ));
#ifdef _WIN32
        const int written = send(socket, bytes + sent, chunk, 0);
#else
        const int written = static_cast<int>(send(socket, bytes + sent, static_cast<std::size_t>(chunk), 0));
#endif
        if (written <= 0) return false;
        sent += static_cast<std::size_t>(written);
    }
    return true;
}

bool receive_all(SocketHandle socket, void* data, std::size_t size) {
    auto* bytes = static_cast<char*>(data);
    std::size_t received = 0;
    while (received < size) {
        const std::size_t remaining = size - received;
        const int chunk = static_cast<int>(std::min<std::size_t>(
            remaining,
            static_cast<std::size_t>(std::numeric_limits<int>::max())
        ));
#ifdef _WIN32
        const int read = recv(socket, bytes + received, chunk, 0);
#else
        const int read = static_cast<int>(recv(socket, bytes + received, static_cast<std::size_t>(chunk), 0));
#endif
        if (read <= 0) return false;
        received += static_cast<std::size_t>(read);
    }
    return true;
}

bool send_u32(SocketHandle socket, std::uint32_t value) {
    const std::uint32_t wire = htonl(value);
    return send_all(socket, &wire, sizeof(wire));
}

bool receive_u32(SocketHandle socket, std::uint32_t& value) {
    std::uint32_t wire = 0;
    if (!receive_all(socket, &wire, sizeof(wire))) return false;
    value = ntohl(wire);
    return true;
}

struct Request {
    std::uint32_t kind = 0;
    std::string context;
    std::string prompt;
};

bool receive_request(SocketHandle socket, Request& request) {
    std::array<char, 8> magic{};
    if (!receive_all(socket, magic.data(), magic.size()) || magic != kRequestMagic) {
        return false;
    }

    std::uint32_t contextBytes = 0;
    std::uint32_t promptBytes = 0;
    if (!receive_u32(socket, request.kind) ||
        !receive_u32(socket, contextBytes) ||
        !receive_u32(socket, promptBytes)) {
        return false;
    }

    if (contextBytes > kMaxContextBytes || promptBytes > kMaxPromptBytes) {
        return false;
    }

    request.context.assign(contextBytes, '\0');
    request.prompt.assign(promptBytes, '\0');
    if (contextBytes > 0 &&
        !receive_all(socket, request.context.data(), request.context.size())) {
        return false;
    }
    if (promptBytes > 0 &&
        !receive_all(socket, request.prompt.data(), request.prompt.size())) {
        return false;
    }
    return true;
}

bool send_response(
    SocketHandle socket,
    std::uint32_t status,
    bool localModelLoaded,
    std::string_view model,
    std::string_view text
) {
    if (model.size() > std::numeric_limits<std::uint32_t>::max() ||
        text.size() > kMaxReplyBytes ||
        text.size() > std::numeric_limits<std::uint32_t>::max()) {
        return false;
    }

    const std::uint32_t flags = localModelLoaded ? kFlagLocalModelLoaded : 0U;
    if (!send_all(socket, kResponseMagic.data(), kResponseMagic.size()) ||
        !send_u32(socket, status) ||
        !send_u32(socket, flags) ||
        !send_u32(socket, static_cast<std::uint32_t>(model.size())) ||
        !send_u32(socket, static_cast<std::uint32_t>(text.size()))) {
        return false;
    }
    if (!model.empty() && !send_all(socket, model.data(), model.size())) return false;
    if (!text.empty() && !send_all(socket, text.data(), text.size())) return false;
    return true;
}

struct Options {
    std::uint16_t port = kDefaultPort;
    std::string modelPath;
    std::string organicStatePath;
};

std::optional<std::uint16_t> parse_port(std::string_view value) {
    try {
        const unsigned long parsed = std::stoul(std::string(value));
        if (parsed == 0 || parsed > 65535UL) return std::nullopt;
        return static_cast<std::uint16_t>(parsed);
    } catch (...) {
        return std::nullopt;
    }
}

bool parse_options(int argc, char** argv, Options& options) {
    for (int index = 1; index < argc; ++index) {
        const std::string_view arg = argv[index];
        if (arg == "--port" && index + 1 < argc) {
            const auto parsed = parse_port(argv[++index]);
            if (!parsed) return false;
            options.port = *parsed;
        } else if (arg == "--model" && index + 1 < argc) {
            options.modelPath = argv[++index];
        } else if (arg == "--organic-state" && index + 1 < argc) {
            options.organicStatePath = argv[++index];
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "SpiralHakuiCortex [--port 47691] [--model model.gguf] "
                   "[--organic-state path]\n";
            std::exit(0);
        } else {
            return false;
        }
    }
    return true;
}

std::string model_label(const spiral::ether_ai::Status& status) {
    if (!status.xenon_model_path.empty()) return status.xenon_model_path;
    return status.xenon_local_cortex_loaded ? "local-cortex" : "organic/default";
}

void serve_client(SocketHandle client, spiral::ether_ai::Runtime& runtime) {
    Request request;
    if (!receive_request(client, request)) {
        const auto status = runtime.status();
        (void)send_response(
            client,
            kResponseBadRequest,
            status.xenon_local_cortex_loaded,
            model_label(status),
            "invalid SPIRAL_HAKUI/1 request"
        );
        return;
    }

    if (request.kind == kPing) {
        const auto status = runtime.status();
        (void)send_response(
            client,
            kResponseOk,
            status.xenon_local_cortex_loaded,
            model_label(status),
            "Spiral Ether AI HAKUI cortex ready"
        );
        return;
    }

    if (request.kind != kAsk || request.prompt.empty()) {
        const auto status = runtime.status();
        (void)send_response(
            client,
            kResponseBadRequest,
            status.xenon_local_cortex_loaded,
            model_label(status),
            "ASK requires a non-empty prompt"
        );
        return;
    }

    try {
        auto host = spiral::ether_ai::hakui_host();
        host.context = std::move(request.context);
        runtime.set_host(std::move(host));

        const std::string reply = runtime.send(request.prompt);
        const auto status = runtime.status();
        (void)send_response(
            client,
            kResponseOk,
            status.xenon_local_cortex_loaded,
            model_label(status),
            reply
        );
    } catch (const std::exception& exception) {
        const auto status = runtime.status();
        (void)send_response(
            client,
            kResponseRuntimeError,
            status.xenon_local_cortex_loaded,
            model_label(status),
            exception.what()
        );
    } catch (...) {
        const auto status = runtime.status();
        (void)send_response(
            client,
            kResponseRuntimeError,
            status.xenon_local_cortex_loaded,
            model_label(status),
            "unknown Spiral Ether AI runtime failure"
        );
    }
}

} // namespace

int main(int argc, char** argv) {
    Options options;
    if (!parse_options(argc, argv, options)) {
        std::cerr << "usage: SpiralHakuiCortex [--port 47691] [--model model.gguf] "
                     "[--organic-state path]\n";
        return 2;
    }

    if (options.modelPath.empty()) {
        if (const char* model = std::getenv("SPIRAL_HAKUI_MODEL"); model && *model) {
            options.modelPath = model;
        }
    }

    SocketRuntime sockets;
    if (!sockets.ok()) {
        std::cerr << "[SPIRAL HAKUI] socket runtime initialization failed\n";
        return 3;
    }

    spiral::ether_ai::Runtime runtime(
        spiral::ether_ai::hakui_host(),
        options.organicStatePath
    );
    if (!options.modelPath.empty()) {
        std::string error;
        if (!runtime.load_local_model(options.modelPath, &error)) {
            std::cerr << "[SPIRAL HAKUI] local model load failed // " << error << '\n';
        }
    }

    SocketHandle server = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (server == kInvalidSocket) {
        std::cerr << "[SPIRAL HAKUI] socket creation failed\n";
        return 4;
    }

    int reuse = 1;
#ifdef _WIN32
    (void)setsockopt(server, SOL_SOCKET, SO_REUSEADDR,
        reinterpret_cast<const char*>(&reuse), sizeof(reuse));
#else
    (void)setsockopt(server, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
#endif

    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(options.port);
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);

    if (bind(server, reinterpret_cast<const sockaddr*>(&address), sizeof(address)) != 0) {
        std::cerr << "[SPIRAL HAKUI] bind failed // loopback port " << options.port << '\n';
        close_socket(server);
        return 5;
    }
    if (listen(server, 4) != 0) {
        std::cerr << "[SPIRAL HAKUI] listen failed\n";
        close_socket(server);
        return 6;
    }

    const auto status = runtime.status();
    std::cout
        << "[SPIRAL HAKUI] CORTEX BRIDGE // ONLINE // 127.0.0.1:"
        << options.port
        << " // host=Hakui // local_model="
        << (status.xenon_local_cortex_loaded ? "loaded" : "not-loaded")
        << '\n';

    for (;;) {
        sockaddr_in clientAddress{};
#ifdef _WIN32
        int clientLength = sizeof(clientAddress);
#else
        socklen_t clientLength = sizeof(clientAddress);
#endif
        SocketHandle client = accept(
            server,
            reinterpret_cast<sockaddr*>(&clientAddress),
            &clientLength
        );
        if (client == kInvalidSocket) {
            std::cerr << "[SPIRAL HAKUI] accept failed\n";
            continue;
        }
        serve_client(client, runtime);
        close_socket(client);
    }
}
