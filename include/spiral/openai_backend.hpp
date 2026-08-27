#pragma once

#include <string>
#include <string_view>

namespace spiral::openai {

struct ResponseResult {
    bool ok = false;
    int http_status = 0;
    std::string text;
    std::string error;
};

class ResponsesBackend final {
public:
    [[nodiscard]] static bool platform_supported() noexcept;
    [[nodiscard]] static bool api_key_present() noexcept;
    [[nodiscard]] static std::string default_model();

    [[nodiscard]] ResponseResult respond(
        std::string_view instructions,
        std::string_view input,
        std::string_view model = {}) const;
};

} // namespace spiral::openai
