#include "spiral/openai_backend.hpp"

#include <cctype>
#include <cstdlib>
#include <iomanip>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <winhttp.h>
#endif

namespace spiral::openai {
namespace {

std::string json_escape(std::string_view value) {
    std::ostringstream out;
    for (const unsigned char ch : value) {
        switch (ch) {
            case '"': out << "\\\""; break;
            case '\\': out << "\\\\"; break;
            case '\b': out << "\\b"; break;
            case '\f': out << "\\f"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (ch < 0x20U) {
                    out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                        << static_cast<unsigned int>(ch) << std::dec;
                } else {
                    out << static_cast<char>(ch);
                }
                break;
        }
    }
    return out.str();
}

void append_utf8(std::string& out, std::uint32_t codepoint) {
    if (codepoint <= 0x7FU) {
        out.push_back(static_cast<char>(codepoint));
    } else if (codepoint <= 0x7FFU) {
        out.push_back(static_cast<char>(0xC0U | (codepoint >> 6U)));
        out.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
    } else if (codepoint <= 0xFFFFU) {
        out.push_back(static_cast<char>(0xE0U | (codepoint >> 12U)));
        out.push_back(static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3FU)));
        out.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
    } else if (codepoint <= 0x10FFFFU) {
        out.push_back(static_cast<char>(0xF0U | (codepoint >> 18U)));
        out.push_back(static_cast<char>(0x80U | ((codepoint >> 12U) & 0x3FU)));
        out.push_back(static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3FU)));
        out.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
    }
}

int hex_digit(char ch) {
    if (ch >= '0' && ch <= '9') return ch - '0';
    if (ch >= 'a' && ch <= 'f') return 10 + ch - 'a';
    if (ch >= 'A' && ch <= 'F') return 10 + ch - 'A';
    return -1;
}

std::optional<std::uint32_t> parse_hex4(std::string_view text, std::size_t offset) {
    if (offset + 4 > text.size()) return std::nullopt;
    std::uint32_t value = 0;
    for (std::size_t i = 0; i < 4; ++i) {
        const int digit = hex_digit(text[offset + i]);
        if (digit < 0) return std::nullopt;
        value = (value << 4U) | static_cast<std::uint32_t>(digit);
    }
    return value;
}

std::optional<std::pair<std::string, std::size_t>> parse_json_string(
    std::string_view json,
    std::size_t quote_position) {
    if (quote_position >= json.size() || json[quote_position] != '"') return std::nullopt;
    std::string out;
    for (std::size_t i = quote_position + 1; i < json.size(); ++i) {
        const char ch = json[i];
        if (ch == '"') return std::pair<std::string, std::size_t>{std::move(out), i + 1};
        if (ch != '\\') {
            out.push_back(ch);
            continue;
        }
        if (++i >= json.size()) return std::nullopt;
        switch (json[i]) {
            case '"': out.push_back('"'); break;
            case '\\': out.push_back('\\'); break;
            case '/': out.push_back('/'); break;
            case 'b': out.push_back('\b'); break;
            case 'f': out.push_back('\f'); break;
            case 'n': out.push_back('\n'); break;
            case 'r': out.push_back('\r'); break;
            case 't': out.push_back('\t'); break;
            case 'u': {
                const auto high = parse_hex4(json, i + 1);
                if (!high.has_value()) return std::nullopt;
                i += 4;
                std::uint32_t codepoint = *high;
                if (codepoint >= 0xD800U && codepoint <= 0xDBFFU &&
                    i + 6 < json.size() && json[i + 1] == '\\' && json[i + 2] == 'u') {
                    const auto low = parse_hex4(json, i + 3);
                    if (low.has_value() && *low >= 0xDC00U && *low <= 0xDFFFU) {
                        codepoint = 0x10000U + ((codepoint - 0xD800U) << 10U) + (*low - 0xDC00U);
                        i += 6;
                    }
                }
                append_utf8(out, codepoint);
                break;
            }
            default: return std::nullopt;
        }
    }
    return std::nullopt;
}

std::optional<std::pair<std::string, std::size_t>> find_string_field(
    std::string_view json,
    std::string_view key,
    std::size_t start = 0) {
    const std::string needle = "\"" + std::string(key) + "\"";
    std::size_t position = start;
    while ((position = json.find(needle, position)) != std::string_view::npos) {
        std::size_t cursor = position + needle.size();
        while (cursor < json.size() && std::isspace(static_cast<unsigned char>(json[cursor])) != 0) ++cursor;
        if (cursor >= json.size() || json[cursor] != ':') {
            position += needle.size();
            continue;
        }
        ++cursor;
        while (cursor < json.size() && std::isspace(static_cast<unsigned char>(json[cursor])) != 0) ++cursor;
        if (cursor >= json.size() || json[cursor] != '"') {
            position += needle.size();
            continue;
        }
        return parse_json_string(json, cursor);
    }
    return std::nullopt;
}

std::optional<std::string> extract_output_text(std::string_view json) {
    std::size_t cursor = 0;
    while (true) {
        const auto type = find_string_field(json, "type", cursor);
        if (!type.has_value()) return std::nullopt;
        cursor = type->second;
        if (type->first != "output_text") continue;
        const auto text = find_string_field(json, "text", cursor);
        if (text.has_value()) return text->first;
    }
}

std::string extract_error_message(std::string_view json) {
    const auto message = find_string_field(json, "message");
    return message.has_value() ? message->first : "OpenAI API returned an error response";
}

#ifdef _WIN32

class InternetHandle final {
public:
    InternetHandle() = default;
    explicit InternetHandle(HINTERNET handle) : handle_(handle) {}
    ~InternetHandle() { reset(); }
    InternetHandle(const InternetHandle&) = delete;
    InternetHandle& operator=(const InternetHandle&) = delete;
    InternetHandle(InternetHandle&& other) noexcept : handle_(other.handle_) { other.handle_ = nullptr; }
    InternetHandle& operator=(InternetHandle&& other) noexcept {
        if (this != &other) {
            reset();
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }
    [[nodiscard]] HINTERNET get() const noexcept { return handle_; }
    [[nodiscard]] explicit operator bool() const noexcept { return handle_ != nullptr; }
private:
    void reset() noexcept {
        if (handle_ != nullptr) WinHttpCloseHandle(handle_);
        handle_ = nullptr;
    }
    HINTERNET handle_ = nullptr;
};

std::wstring widen_ascii(std::string_view value) {
    return std::wstring(value.begin(), value.end());
}

std::string winhttp_error(const char* operation) {
    return std::string(operation) + " failed (WinHTTP error " + std::to_string(GetLastError()) + ")";
}

#endif

} // namespace

bool ResponsesBackend::platform_supported() noexcept {
#ifdef _WIN32
    return true;
#else
    return false;
#endif
}

bool ResponsesBackend::api_key_present() noexcept {
    const char* key = std::getenv("OPENAI_API_KEY");
    return key != nullptr && *key != '\0';
}

std::string ResponsesBackend::default_model() {
    const char* configured = std::getenv("SPIRAL_GPT_MODEL");
    if (configured != nullptr && *configured != '\0') return configured;
    return "gpt-5.6";
}

ResponseResult ResponsesBackend::respond(
    std::string_view instructions,
    std::string_view input,
    std::string_view model) const {
    ResponseResult result;
#ifndef _WIN32
    (void)instructions; (void)input; (void)model;
    result.error = "OpenAI GPT backend is only implemented for the Windows shell at this rung";
    return result;
#else
    const char* key = std::getenv("OPENAI_API_KEY");
    if (key == nullptr || *key == '\0') {
        result.error = "OPENAI_API_KEY is not set";
        return result;
    }

    const std::string resolved_model = model.empty() ? default_model() : std::string(model);
    const std::string body =
        "{\"model\":\"" + json_escape(resolved_model) +
        "\",\"instructions\":\"" + json_escape(instructions) +
        "\",\"input\":\"" + json_escape(input) +
        "\",\"store\":false}";

    InternetHandle session(WinHttpOpen(
        L"SpiralOS-AI-Genius/0.0.25",
        WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY,
        WINHTTP_NO_PROXY_NAME,
        WINHTTP_NO_PROXY_BYPASS,
        0));
    if (!session) { result.error = winhttp_error("WinHttpOpen"); return result; }
    WinHttpSetTimeouts(session.get(), 10000, 10000, 30000, 90000);

    InternetHandle connection(WinHttpConnect(
        session.get(), L"api.openai.com", INTERNET_DEFAULT_HTTPS_PORT, 0));
    if (!connection) { result.error = winhttp_error("WinHttpConnect"); return result; }

    InternetHandle request(WinHttpOpenRequest(
        connection.get(),
        L"POST",
        L"/v1/responses",
        nullptr,
        WINHTTP_NO_REFERER,
        WINHTTP_DEFAULT_ACCEPT_TYPES,
        WINHTTP_FLAG_SECURE));
    if (!request) { result.error = winhttp_error("WinHttpOpenRequest"); return result; }

    const std::wstring authorization = L"Authorization: Bearer " + widen_ascii(key);
    if (!WinHttpAddRequestHeaders(
            request.get(), authorization.c_str(), static_cast<DWORD>(-1L),
            WINHTTP_ADDREQ_FLAG_ADD | WINHTTP_ADDREQ_FLAG_REPLACE)) {
        result.error = winhttp_error("WinHttpAddRequestHeaders authorization");
        return result;
    }
    constexpr wchar_t content_type[] = L"Content-Type: application/json";
    if (!WinHttpAddRequestHeaders(
            request.get(), content_type, static_cast<DWORD>(-1L),
            WINHTTP_ADDREQ_FLAG_ADD | WINHTTP_ADDREQ_FLAG_REPLACE)) {
        result.error = winhttp_error("WinHttpAddRequestHeaders content-type");
        return result;
    }

    if (!WinHttpSendRequest(
            request.get(),
            WINHTTP_NO_ADDITIONAL_HEADERS,
            0,
            const_cast<char*>(body.data()),
            static_cast<DWORD>(body.size()),
            static_cast<DWORD>(body.size()),
            0)) {
        result.error = winhttp_error("WinHttpSendRequest");
        return result;
    }
    if (!WinHttpReceiveResponse(request.get(), nullptr)) {
        result.error = winhttp_error("WinHttpReceiveResponse");
        return result;
    }

    DWORD status = 0;
    DWORD status_size = sizeof(status);
    if (WinHttpQueryHeaders(
            request.get(),
            WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER,
            WINHTTP_HEADER_NAME_BY_INDEX,
            &status,
            &status_size,
            WINHTTP_NO_HEADER_INDEX)) {
        result.http_status = static_cast<int>(status);
    }

    std::string response;
    constexpr std::size_t max_response_bytes = 16U * 1024U * 1024U;
    while (true) {
        DWORD available = 0;
        if (!WinHttpQueryDataAvailable(request.get(), &available)) {
            result.error = winhttp_error("WinHttpQueryDataAvailable");
            return result;
        }
        if (available == 0) break;
        if (response.size() + available > max_response_bytes) {
            result.error = "OpenAI API response exceeded 16 MiB safety limit";
            return result;
        }
        const std::size_t old_size = response.size();
        response.resize(old_size + available);
        DWORD read = 0;
        if (!WinHttpReadData(request.get(), response.data() + old_size, available, &read)) {
            result.error = winhttp_error("WinHttpReadData");
            return result;
        }
        response.resize(old_size + read);
        if (read == 0) break;
    }

    if (result.http_status < 200 || result.http_status >= 300) {
        result.error = extract_error_message(response);
        return result;
    }

    const auto output_text = extract_output_text(response);
    if (!output_text.has_value()) {
        result.error = "OpenAI response contained no output_text item";
        return result;
    }
    result.ok = true;
    result.text = *output_text;
    return result;
#endif
}

} // namespace spiral::openai
