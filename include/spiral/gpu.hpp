#pragma once

#include "spiral/device.hpp"

#include <memory>
#include <string>
#include <string_view>

namespace spiral::gpu {

enum class NativeGpuBackend {
    D3D11,
};

struct NativeGpuCapabilities {
    bool available = false;
    bool hardware_accelerated = false;
    NativeGpuBackend backend = NativeGpuBackend::D3D11;
    std::string adapter_name;
    std::string feature_level;
};

class D3D11GpuDevice final : public device::Device {
public:
    D3D11GpuDevice();
    ~D3D11GpuDevice() override;

    D3D11GpuDevice(const D3D11GpuDevice&) = delete;
    D3D11GpuDevice& operator=(const D3D11GpuDevice&) = delete;
    D3D11GpuDevice(D3D11GpuDevice&&) = delete;
    D3D11GpuDevice& operator=(D3D11GpuDevice&&) = delete;

    [[nodiscard]] static std::unique_ptr<D3D11GpuDevice> try_create(std::string* error = nullptr) noexcept;
    [[nodiscard]] static bool platform_supported() noexcept;

    [[nodiscard]] NativeGpuCapabilities capabilities() const;
    [[nodiscard]] device::DeviceInfo info() const override;
    [[nodiscard]] device::BufferHandle create_buffer(device::BufferDesc desc) override;
    void destroy_buffer(device::BufferHandle handle) override;
    void upload(
        device::BufferHandle handle,
        std::span<const std::byte> data,
        std::size_t offset = 0) override;
    [[nodiscard]] std::vector<std::byte> download(
        device::BufferHandle handle,
        std::size_t offset = 0,
        std::size_t size_bytes = 0) const override;
    void submit(const device::CommandList& commands) override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    friend class D3D11FramebufferPresenter;
};

[[nodiscard]] std::string_view native_gpu_backend_name(NativeGpuBackend backend) noexcept;

} // namespace spiral::gpu
