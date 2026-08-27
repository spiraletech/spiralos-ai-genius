#pragma once

#include "spiral/gpu.hpp"

#include <algorithm>
#include <cctype>
#include <string>
#include <string_view>

namespace spiral::gpu {

[[nodiscard]] inline bool adapter_name_is_software(std::string_view adapter_name) {
    std::string normalized(adapter_name);
    std::transform(normalized.begin(), normalized.end(), normalized.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return normalized.find("microsoft basic render driver") != std::string::npos ||
        normalized.find("warp") != std::string::npos ||
        normalized.find("software") != std::string::npos;
}

[[nodiscard]] inline bool is_physical_gpu(const NativeGpuCapabilities& capabilities) {
    return capabilities.available && capabilities.hardware_accelerated &&
        !adapter_name_is_software(capabilities.adapter_name);
}

[[nodiscard]] inline std::string_view adapter_execution_class(const NativeGpuCapabilities& capabilities) {
    if (!capabilities.available) return "unavailable";
    if (adapter_name_is_software(capabilities.adapter_name)) return "software";
    if (capabilities.hardware_accelerated) return "physical-hardware";
    return "compatibility-software";
}

} // namespace spiral::gpu
