#pragma once

#include "spiral/gpu.hpp"
#include "spiral/raster.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace spiral::host {

struct NativeWindowConfig {
    std::size_t width = 960;
    std::size_t height = 540;
    std::string title = "Spiral";
    bool visible = true;
    bool resizable = true;
};

class NativeWindowHost final {
public:
    NativeWindowHost();
    ~NativeWindowHost();

    NativeWindowHost(const NativeWindowHost&) = delete;
    NativeWindowHost& operator=(const NativeWindowHost&) = delete;
    NativeWindowHost(NativeWindowHost&&) = delete;
    NativeWindowHost& operator=(NativeWindowHost&&) = delete;

    [[nodiscard]] static bool platform_supported() noexcept;

    void create(const NativeWindowConfig& config);
    void show(bool visible);
    void resize(std::size_t width, std::size_t height);
    void pump_events();

    [[nodiscard]] bool created() const noexcept;
    [[nodiscard]] bool close_requested() const noexcept;
    [[nodiscard]] std::size_t client_width() const noexcept;
    [[nodiscard]] std::size_t client_height() const noexcept;
    [[nodiscard]] void* native_handle() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace spiral::host

namespace spiral::gpu {

class D3D11FramebufferPresenter final {
public:
    D3D11FramebufferPresenter(D3D11GpuDevice& gpu, host::NativeWindowHost& window);
    ~D3D11FramebufferPresenter();

    D3D11FramebufferPresenter(const D3D11FramebufferPresenter&) = delete;
    D3D11FramebufferPresenter& operator=(const D3D11FramebufferPresenter&) = delete;
    D3D11FramebufferPresenter(D3D11FramebufferPresenter&&) = delete;
    D3D11FramebufferPresenter& operator=(D3D11FramebufferPresenter&&) = delete;

    void present(const raster::Framebuffer& framebuffer, bool vsync = false);

    [[nodiscard]] std::uint64_t frame_count() const noexcept;
    [[nodiscard]] std::uint64_t last_present_hash() const noexcept;
    [[nodiscard]] std::size_t width() const noexcept;
    [[nodiscard]] std::size_t height() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace spiral::gpu
