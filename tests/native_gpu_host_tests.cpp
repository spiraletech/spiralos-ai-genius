#include "spiral/device.hpp"
#include "spiral/gpu.hpp"
#include "spiral/native_host.hpp"
#include "spiral/raster.hpp"

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

int main() {
    using namespace spiral;

    assert(gpu::native_gpu_backend_name(gpu::NativeGpuBackend::D3D11) == "d3d11");

#ifdef _WIN32
    assert(gpu::D3D11GpuDevice::platform_supported());
    assert(host::NativeWindowHost::platform_supported());

    std::string create_error;
    auto native_gpu = gpu::D3D11GpuDevice::try_create(&create_error);
    assert(native_gpu != nullptr);
    assert(create_error.empty());

    const auto caps = native_gpu->capabilities();
    assert(caps.available);
    assert(!caps.adapter_name.empty());
    assert(!caps.feature_level.empty());
    assert(native_gpu->info().kind == device::DeviceKind::Gpu);

    const auto source = native_gpu->create_buffer(device::BufferDesc{
        64,
        device::BufferUsage::Transfer,
        true});
    const auto destination = native_gpu->create_buffer(device::BufferDesc{
        64,
        device::BufferUsage::Transfer,
        true});

    std::array<std::byte, 64> pattern{};
    for (std::size_t i = 0; i < pattern.size(); ++i) {
        pattern[i] = std::byte{static_cast<unsigned char>(i)};
    }
    native_gpu->upload(source, pattern);

    device::CommandQueue queue(*native_gpu);
    device::CommandList commands;
    commands.fill_buffer(destination, std::byte{0xEE});
    commands.copy_buffer(source, destination, 16, 8, 16);
    queue.submit(commands);
    assert(queue.submission_count() == 1);

    const auto downloaded = native_gpu->download(destination);
    assert(downloaded.size() == 64);
    for (std::size_t i = 0; i < downloaded.size(); ++i) {
        if (i >= 16 && i < 32) {
            assert(downloaded[i] == pattern[i - 8]);
        } else {
            assert(downloaded[i] == std::byte{0xEE});
        }
    }

    device::CommandList invalid;
    invalid.copy_buffer(source, destination, 60, 8, 0);
    bool rejected = false;
    try {
        queue.submit(invalid);
    } catch (const std::out_of_range&) {
        rejected = true;
    }
    assert(rejected);
    assert(queue.submission_count() == 1);

    const auto gpu_info = native_gpu->info();
    assert(gpu_info.buffer_count == 2);
    assert(gpu_info.allocated_bytes == 128);

    host::NativeWindowHost window;
    window.create(host::NativeWindowConfig{
        96,
        64,
        "Spiral L22 native host regression",
        false,
        false});
    assert(window.created());
    assert(window.native_handle() != nullptr);
    assert(window.client_width() == 96);
    assert(window.client_height() == 64);
    assert(!window.close_requested());
    window.pump_events();

    raster::Framebuffer framebuffer(96, 64, raster::Rgba8{8, 10, 14, 255});
    for (std::size_t y = 8; y < 56; ++y) {
        for (std::size_t x = 12; x < 84; ++x) {
            framebuffer.set_pixel(x, y, raster::Rgba8{
                static_cast<std::uint8_t>((x * 3U) & 0xFFU),
                static_cast<std::uint8_t>((y * 5U) & 0xFFU),
                180,
                255});
        }
    }
    const auto framebuffer_hash = framebuffer.hash64();

    gpu::D3D11FramebufferPresenter presenter(*native_gpu, window);
    presenter.present(framebuffer, false);
    assert(presenter.frame_count() == 1);
    assert(presenter.last_present_hash() == framebuffer_hash);
    assert(presenter.width() == 96);
    assert(presenter.height() == 64);

    framebuffer.set_pixel(0, 0, raster::Rgba8{255, 0, 255, 255});
    presenter.present(framebuffer, false);
    assert(presenter.frame_count() == 2);
    assert(presenter.last_present_hash() == framebuffer.hash64());

    native_gpu->destroy_buffer(source);
    native_gpu->destroy_buffer(destination);
    assert(native_gpu->info().buffer_count == 0);

    std::cout << "L22 backend: " << gpu::native_gpu_backend_name(caps.backend) << '\n';
    std::cout << "L22 adapter: " << caps.adapter_name << '\n';
    std::cout << "L22 feature level: " << caps.feature_level << '\n';
    std::cout << "L22 hardware accelerated: " << (caps.hardware_accelerated ? "yes" : "WARP fallback") << '\n';
    std::cout << "L22 native GPU/window test passed\n";
#else
    assert(!gpu::D3D11GpuDevice::platform_supported());
    assert(!host::NativeWindowHost::platform_supported());

    std::string error;
    auto native_gpu = gpu::D3D11GpuDevice::try_create(&error);
    assert(native_gpu == nullptr);
    assert(!error.empty());

    host::NativeWindowHost window;
    assert(!window.created());
    assert(window.native_handle() == nullptr);
    bool rejected = false;
    try {
        window.create(host::NativeWindowConfig{64, 64, "unsupported", false, false});
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    assert(rejected);

    std::cout << "L22 D3D11/Win32 path correctly unavailable on this platform\n";
#endif

    return 0;
}
