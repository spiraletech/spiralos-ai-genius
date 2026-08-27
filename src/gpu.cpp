#include "spiral/gpu.hpp"
#include "spiral/native_host.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <d3d11.h>
#include <dxgi.h>
#endif

namespace spiral::gpu {
namespace {

std::size_t resolved_size(std::size_t total, std::size_t offset, std::size_t requested) {
    if (offset > total) throw std::out_of_range("gpu buffer offset exceeds buffer size");
    const std::size_t size = requested == 0 ? total - offset : requested;
    if (size > total - offset) throw std::out_of_range("gpu buffer range exceeds allocation");
    return size;
}

#ifdef _WIN32

[[noreturn]] void throw_hresult(const char* operation, HRESULT hr) {
    std::ostringstream stream;
    stream << operation << " failed with HRESULT 0x" << std::hex << std::uppercase
           << static_cast<unsigned long>(hr);
    throw std::runtime_error(stream.str());
}

UINT checked_uint(std::size_t value, const char* field) {
    if (value > static_cast<std::size_t>(std::numeric_limits<UINT>::max())) {
        throw std::overflow_error(std::string(field) + " exceeds D3D11 UINT range");
    }
    return static_cast<UINT>(value);
}

UINT rounded_buffer_size(std::size_t logical_size) {
    if (logical_size == 0) throw std::invalid_argument("gpu buffer size must be non-zero");
    if (logical_size > static_cast<std::size_t>(std::numeric_limits<UINT>::max()) - 3U) {
        throw std::overflow_error("gpu buffer allocation exceeds D3D11 range");
    }
    return static_cast<UINT>((logical_size + 3U) & ~std::size_t{3U});
}

std::string utf8_from_wide(const wchar_t* text) {
    if (text == nullptr || *text == L'\0') return {};
    const int needed = WideCharToMultiByte(CP_UTF8, 0, text, -1, nullptr, 0, nullptr, nullptr);
    if (needed <= 1) return {};
    std::string result(static_cast<std::size_t>(needed), '\0');
    const int written = WideCharToMultiByte(CP_UTF8, 0, text, -1, result.data(), needed, nullptr, nullptr);
    if (written <= 1) return {};
    result.resize(static_cast<std::size_t>(written - 1));
    return result;
}

std::string feature_level_name(D3D_FEATURE_LEVEL level) {
    switch (level) {
        case D3D_FEATURE_LEVEL_11_1: return "11.1";
        case D3D_FEATURE_LEVEL_11_0: return "11.0";
        case D3D_FEATURE_LEVEL_10_1: return "10.1";
        case D3D_FEATURE_LEVEL_10_0: return "10.0";
        case D3D_FEATURE_LEVEL_9_3: return "9.3";
        case D3D_FEATURE_LEVEL_9_2: return "9.2";
        case D3D_FEATURE_LEVEL_9_1: return "9.1";
        default: return "unknown";
    }
}

#endif

} // namespace

struct D3D11GpuDevice::Impl {
#ifdef _WIN32
    struct Buffer {
        device::BufferDesc desc;
        std::size_t logical_size = 0;
        UINT physical_size = 0;
        ID3D11Buffer* resource = nullptr;
    };

    ID3D11Device* device = nullptr;
    ID3D11DeviceContext* context = nullptr;
    D3D_FEATURE_LEVEL feature_level = D3D_FEATURE_LEVEL_9_1;
    bool hardware_accelerated = false;
    std::string adapter_name;
    device::BufferHandle next_handle = 1;
    std::map<device::BufferHandle, Buffer> buffers;

    ~Impl() {
        for (auto& [handle, buffer] : buffers) {
            (void)handle;
            if (buffer.resource != nullptr) buffer.resource->Release();
        }
        if (context != nullptr) context->Release();
        if (device != nullptr) device->Release();
    }

    Buffer& require_buffer(device::BufferHandle handle) {
        const auto it = buffers.find(handle);
        if (it == buffers.end()) throw std::invalid_argument("gpu buffer handle not found");
        return it->second;
    }

    const Buffer& require_buffer(device::BufferHandle handle) const {
        const auto it = buffers.find(handle);
        if (it == buffers.end()) throw std::invalid_argument("gpu buffer handle not found");
        return it->second;
    }

    static void require_range(const Buffer& buffer, std::size_t offset, std::size_t count) {
        if (offset > buffer.logical_size || count > buffer.logical_size - offset) {
            throw std::out_of_range("gpu buffer range exceeds allocation");
        }
    }

    void update_bytes(Buffer& buffer, std::span<const std::byte> data, std::size_t offset) {
        require_range(buffer, offset, data.size());
        if (data.empty()) return;
        D3D11_BOX box{};
        box.left = checked_uint(offset, "gpu upload offset");
        box.right = checked_uint(offset + data.size(), "gpu upload end");
        box.top = 0;
        box.bottom = 1;
        box.front = 0;
        box.back = 1;
        context->UpdateSubresource(buffer.resource, 0, &box, data.data(), 0, 0);
    }

    void fill_bytes(Buffer& buffer, std::byte value, std::size_t offset, std::size_t count) {
        require_range(buffer, offset, count);
        if (count == 0) return;

        const std::uint8_t byte_value = std::to_integer<std::uint8_t>(value);
        const std::uint32_t repeated =
            static_cast<std::uint32_t>(byte_value) |
            (static_cast<std::uint32_t>(byte_value) << 8U) |
            (static_cast<std::uint32_t>(byte_value) << 16U) |
            (static_cast<std::uint32_t>(byte_value) << 24U);

        const std::size_t aligned_begin = (offset + 3U) & ~std::size_t{3U};
        const std::size_t end = offset + count;
        const std::size_t aligned_end = end & ~std::size_t{3U};

        const std::size_t head_end = std::min(end, aligned_begin);
        if (head_end > offset) {
            const std::vector<std::byte> head(head_end - offset, value);
            update_bytes(buffer, head, offset);
        }

        if (aligned_end > aligned_begin) {
            D3D11_UNORDERED_ACCESS_VIEW_DESC view_desc{};
            view_desc.Format = DXGI_FORMAT_R32_TYPELESS;
            view_desc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
            view_desc.Buffer.FirstElement = checked_uint(aligned_begin / 4U, "gpu fill first element");
            view_desc.Buffer.NumElements = checked_uint((aligned_end - aligned_begin) / 4U, "gpu fill element count");
            view_desc.Buffer.Flags = D3D11_BUFFER_UAV_FLAG_RAW;

            ID3D11UnorderedAccessView* view = nullptr;
            const HRESULT hr = device->CreateUnorderedAccessView(buffer.resource, &view_desc, &view);
            if (FAILED(hr)) throw_hresult("ID3D11Device::CreateUnorderedAccessView", hr);
            const UINT values[4] = {repeated, repeated, repeated, repeated};
            context->ClearUnorderedAccessViewUint(view, values);
            view->Release();
        }

        const std::size_t tail_begin = std::max(offset, aligned_end);
        if (end > tail_begin) {
            const std::vector<std::byte> tail(end - tail_begin, value);
            update_bytes(buffer, tail, tail_begin);
        }
    }
#endif
};

D3D11GpuDevice::D3D11GpuDevice() : impl_(std::make_unique<Impl>()) {
#ifdef _WIN32
    constexpr std::array<D3D_FEATURE_LEVEL, 4> levels{
        D3D_FEATURE_LEVEL_11_1,
        D3D_FEATURE_LEVEL_11_0,
        D3D_FEATURE_LEVEL_10_1,
        D3D_FEATURE_LEVEL_10_0,
    };

    auto create = [&](D3D_DRIVER_TYPE type, bool hardware) -> HRESULT {
        impl_->device = nullptr;
        impl_->context = nullptr;
        impl_->feature_level = D3D_FEATURE_LEVEL_9_1;
        HRESULT hr = D3D11CreateDevice(
            nullptr,
            type,
            nullptr,
            D3D11_CREATE_DEVICE_BGRA_SUPPORT,
            levels.data(),
            static_cast<UINT>(levels.size()),
            D3D11_SDK_VERSION,
            &impl_->device,
            &impl_->feature_level,
            &impl_->context);
        if (hr == E_INVALIDARG) {
            hr = D3D11CreateDevice(
                nullptr,
                type,
                nullptr,
                D3D11_CREATE_DEVICE_BGRA_SUPPORT,
                levels.data() + 1,
                static_cast<UINT>(levels.size() - 1),
                D3D11_SDK_VERSION,
                &impl_->device,
                &impl_->feature_level,
                &impl_->context);
        }
        if (SUCCEEDED(hr)) impl_->hardware_accelerated = hardware;
        return hr;
    };

    HRESULT hr = create(D3D_DRIVER_TYPE_HARDWARE, true);
    if (FAILED(hr)) hr = create(D3D_DRIVER_TYPE_WARP, false);
    if (FAILED(hr)) throw_hresult("D3D11CreateDevice hardware/WARP", hr);

    IDXGIDevice* dxgi_device = nullptr;
    if (SUCCEEDED(impl_->device->QueryInterface(__uuidof(IDXGIDevice), reinterpret_cast<void**>(&dxgi_device)))) {
        IDXGIAdapter* adapter = nullptr;
        if (SUCCEEDED(dxgi_device->GetAdapter(&adapter))) {
            DXGI_ADAPTER_DESC desc{};
            if (SUCCEEDED(adapter->GetDesc(&desc))) impl_->adapter_name = utf8_from_wide(desc.Description);
            adapter->Release();
        }
        dxgi_device->Release();
    }
    if (impl_->adapter_name.empty()) {
        impl_->adapter_name = impl_->hardware_accelerated ? "D3D11 hardware adapter" : "D3D11 WARP adapter";
    }
#else
    throw std::runtime_error("D3D11 GPU backend is only available on Windows");
#endif
}

D3D11GpuDevice::~D3D11GpuDevice() = default;

std::unique_ptr<D3D11GpuDevice> D3D11GpuDevice::try_create(std::string* error) noexcept {
    try {
        auto result = std::make_unique<D3D11GpuDevice>();
        if (error != nullptr) error->clear();
        return result;
    } catch (const std::exception& exception) {
        if (error != nullptr) *error = exception.what();
        return nullptr;
    } catch (...) {
        if (error != nullptr) *error = "unknown D3D11 GPU initialization failure";
        return nullptr;
    }
}

bool D3D11GpuDevice::platform_supported() noexcept {
#ifdef _WIN32
    return true;
#else
    return false;
#endif
}

NativeGpuCapabilities D3D11GpuDevice::capabilities() const {
    NativeGpuCapabilities result;
#ifdef _WIN32
    result.available = impl_ != nullptr && impl_->device != nullptr;
    result.hardware_accelerated = impl_->hardware_accelerated;
    result.backend = NativeGpuBackend::D3D11;
    result.adapter_name = impl_->adapter_name;
    result.feature_level = feature_level_name(impl_->feature_level);
#endif
    return result;
}

device::DeviceInfo D3D11GpuDevice::info() const {
    device::DeviceInfo result;
#ifdef _WIN32
    result.name = "spiral-d3d11:" + impl_->adapter_name;
    result.kind = device::DeviceKind::Gpu;
    result.buffer_count = impl_->buffers.size();
    for (const auto& [handle, buffer] : impl_->buffers) {
        (void)handle;
        result.allocated_bytes += buffer.logical_size;
    }
#else
    result.name = "spiral-d3d11-unavailable";
    result.kind = device::DeviceKind::Gpu;
#endif
    return result;
}

device::BufferHandle D3D11GpuDevice::create_buffer(device::BufferDesc desc) {
#ifdef _WIN32
    if (desc.size_bytes == 0) throw std::invalid_argument("gpu buffer size must be non-zero");
    if (impl_->next_handle == 0) throw std::overflow_error("gpu buffer handle overflow");

    D3D11_BUFFER_DESC native_desc{};
    native_desc.ByteWidth = rounded_buffer_size(desc.size_bytes);
    native_desc.Usage = D3D11_USAGE_DEFAULT;
    native_desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
    switch (desc.usage) {
        case device::BufferUsage::Vertex: native_desc.BindFlags |= D3D11_BIND_VERTEX_BUFFER; break;
        case device::BufferUsage::Index: native_desc.BindFlags |= D3D11_BIND_INDEX_BUFFER; break;
        case device::BufferUsage::Storage: native_desc.BindFlags |= D3D11_BIND_SHADER_RESOURCE; break;
        case device::BufferUsage::Generic:
        case device::BufferUsage::Transfer:
            break;
    }
    native_desc.CPUAccessFlags = 0;
    native_desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS;
    native_desc.StructureByteStride = 0;

    ID3D11Buffer* resource = nullptr;
    const HRESULT hr = impl_->device->CreateBuffer(&native_desc, nullptr, &resource);
    if (FAILED(hr)) throw_hresult("ID3D11Device::CreateBuffer", hr);

    const device::BufferHandle handle = impl_->next_handle++;
    impl_->buffers.emplace(handle, Impl::Buffer{desc, desc.size_bytes, native_desc.ByteWidth, resource});
    return handle;
#else
    (void)desc;
    throw std::runtime_error("D3D11 GPU backend is unavailable on this platform");
#endif
}

void D3D11GpuDevice::destroy_buffer(device::BufferHandle handle) {
#ifdef _WIN32
    const auto it = impl_->buffers.find(handle);
    if (it == impl_->buffers.end()) throw std::invalid_argument("gpu buffer handle not found");
    if (it->second.resource != nullptr) it->second.resource->Release();
    impl_->buffers.erase(it);
#else
    (void)handle;
    throw std::runtime_error("D3D11 GPU backend is unavailable on this platform");
#endif
}

void D3D11GpuDevice::upload(
    device::BufferHandle handle,
    std::span<const std::byte> data,
    std::size_t offset) {
#ifdef _WIN32
    auto& buffer = impl_->require_buffer(handle);
    if (!buffer.desc.host_visible) throw std::logic_error("gpu buffer is not host visible");
    impl_->update_bytes(buffer, data, offset);
#else
    (void)handle; (void)data; (void)offset;
    throw std::runtime_error("D3D11 GPU backend is unavailable on this platform");
#endif
}

std::vector<std::byte> D3D11GpuDevice::download(
    device::BufferHandle handle,
    std::size_t offset,
    std::size_t size_bytes) const {
#ifdef _WIN32
    const auto& buffer = impl_->require_buffer(handle);
    if (!buffer.desc.host_visible) throw std::logic_error("gpu buffer is not host visible");
    const std::size_t count = resolved_size(buffer.logical_size, offset, size_bytes);
    Impl::require_range(buffer, offset, count);

    D3D11_BUFFER_DESC staging_desc{};
    staging_desc.ByteWidth = buffer.physical_size;
    staging_desc.Usage = D3D11_USAGE_STAGING;
    staging_desc.BindFlags = 0;
    staging_desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    staging_desc.MiscFlags = 0;
    staging_desc.StructureByteStride = 0;

    ID3D11Buffer* staging = nullptr;
    HRESULT hr = impl_->device->CreateBuffer(&staging_desc, nullptr, &staging);
    if (FAILED(hr)) throw_hresult("ID3D11Device::CreateBuffer staging", hr);
    impl_->context->CopyResource(staging, buffer.resource);

    D3D11_MAPPED_SUBRESOURCE mapped{};
    hr = impl_->context->Map(staging, 0, D3D11_MAP_READ, 0, &mapped);
    if (FAILED(hr)) {
        staging->Release();
        throw_hresult("ID3D11DeviceContext::Map", hr);
    }

    const auto* begin = static_cast<const std::byte*>(mapped.pData) + offset;
    std::vector<std::byte> result(begin, begin + static_cast<std::ptrdiff_t>(count));
    impl_->context->Unmap(staging, 0);
    staging->Release();
    return result;
#else
    (void)handle; (void)offset; (void)size_bytes;
    throw std::runtime_error("D3D11 GPU backend is unavailable on this platform");
#endif
}

void D3D11GpuDevice::submit(const device::CommandList& commands) {
#ifdef _WIN32
    for (const auto& command : commands.commands()) {
        switch (command.kind) {
            case device::CommandKind::FillBuffer: {
                auto& destination = impl_->require_buffer(command.destination);
                const std::size_t count = resolved_size(
                    destination.logical_size,
                    command.destination_offset,
                    command.size_bytes);
                impl_->fill_bytes(destination, command.fill_value, command.destination_offset, count);
                break;
            }
            case device::CommandKind::CopyBuffer: {
                const auto& source = impl_->require_buffer(command.source);
                auto& destination = impl_->require_buffer(command.destination);
                Impl::require_range(source, command.source_offset, command.size_bytes);
                Impl::require_range(destination, command.destination_offset, command.size_bytes);

                const UINT source_offset = checked_uint(command.source_offset, "gpu copy source offset");
                const UINT destination_offset = checked_uint(command.destination_offset, "gpu copy destination offset");
                const UINT copy_size = checked_uint(command.size_bytes, "gpu copy size");

                D3D11_BOX box{};
                box.left = source_offset;
                box.right = source_offset + copy_size;
                box.top = 0;
                box.bottom = 1;
                box.front = 0;
                box.back = 1;

                if (command.source == command.destination) {
                    D3D11_BUFFER_DESC temp_desc{};
                    temp_desc.ByteWidth = rounded_buffer_size(command.size_bytes);
                    temp_desc.Usage = D3D11_USAGE_DEFAULT;
                    ID3D11Buffer* temporary = nullptr;
                    HRESULT hr = impl_->device->CreateBuffer(&temp_desc, nullptr, &temporary);
                    if (FAILED(hr)) throw_hresult("ID3D11Device::CreateBuffer temporary copy", hr);
                    impl_->context->CopySubresourceRegion(temporary, 0, 0, 0, 0, source.resource, 0, &box);
                    D3D11_BOX temp_box{};
                    temp_box.left = 0;
                    temp_box.right = copy_size;
                    temp_box.top = 0;
                    temp_box.bottom = 1;
                    temp_box.front = 0;
                    temp_box.back = 1;
                    impl_->context->CopySubresourceRegion(
                        destination.resource, 0, destination_offset, 0, 0, temporary, 0, &temp_box);
                    temporary->Release();
                } else {
                    impl_->context->CopySubresourceRegion(
                        destination.resource, 0, destination_offset, 0, 0, source.resource, 0, &box);
                }
                break;
            }
        }
    }
    impl_->context->Flush();
#else
    (void)commands;
    throw std::runtime_error("D3D11 GPU backend is unavailable on this platform");
#endif
}

std::string_view native_gpu_backend_name(NativeGpuBackend backend) noexcept {
    switch (backend) {
        case NativeGpuBackend::D3D11: return "d3d11";
    }
    return "unknown";
}

struct D3D11FramebufferPresenter::Impl {
#ifdef _WIN32
    D3D11GpuDevice* gpu = nullptr;
    host::NativeWindowHost* window = nullptr;
    IDXGISwapChain* swap_chain = nullptr;
#endif
    std::uint64_t frames = 0;
    std::uint64_t last_hash = 0;
    std::size_t width = 0;
    std::size_t height = 0;

    ~Impl() {
#ifdef _WIN32
        if (swap_chain != nullptr) swap_chain->Release();
#endif
    }
};

D3D11FramebufferPresenter::D3D11FramebufferPresenter(
    D3D11GpuDevice& gpu_device,
    host::NativeWindowHost& window_host)
    : impl_(std::make_unique<Impl>()) {
#ifdef _WIN32
    if (!window_host.created() || window_host.native_handle() == nullptr) {
        throw std::invalid_argument("D3D11 presenter requires a created native window");
    }
    if (gpu_device.impl_ == nullptr || gpu_device.impl_->device == nullptr) {
        throw std::invalid_argument("D3D11 presenter requires an initialized GPU device");
    }

    impl_->gpu = &gpu_device;
    impl_->window = &window_host;
    impl_->width = std::max<std::size_t>(1, window_host.client_width());
    impl_->height = std::max<std::size_t>(1, window_host.client_height());

    IDXGIDevice* dxgi_device = nullptr;
    HRESULT hr = gpu_device.impl_->device->QueryInterface(
        __uuidof(IDXGIDevice), reinterpret_cast<void**>(&dxgi_device));
    if (FAILED(hr)) throw_hresult("ID3D11Device::QueryInterface IDXGIDevice", hr);

    IDXGIAdapter* adapter = nullptr;
    hr = dxgi_device->GetAdapter(&adapter);
    dxgi_device->Release();
    if (FAILED(hr)) throw_hresult("IDXGIDevice::GetAdapter", hr);

    IDXGIFactory* factory = nullptr;
    hr = adapter->GetParent(__uuidof(IDXGIFactory), reinterpret_cast<void**>(&factory));
    adapter->Release();
    if (FAILED(hr)) throw_hresult("IDXGIAdapter::GetParent IDXGIFactory", hr);

    DXGI_SWAP_CHAIN_DESC desc{};
    desc.BufferDesc.Width = checked_uint(impl_->width, "swapchain width");
    desc.BufferDesc.Height = checked_uint(impl_->height, "swapchain height");
    desc.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.BufferDesc.RefreshRate.Numerator = 0;
    desc.BufferDesc.RefreshRate.Denominator = 1;
    desc.SampleDesc.Count = 1;
    desc.SampleDesc.Quality = 0;
    desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    desc.BufferCount = 1;
    desc.OutputWindow = reinterpret_cast<HWND>(window_host.native_handle());
    desc.Windowed = TRUE;
    desc.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;
    desc.Flags = 0;

    hr = factory->CreateSwapChain(gpu_device.impl_->device, &desc, &impl_->swap_chain);
    if (SUCCEEDED(hr)) {
        factory->MakeWindowAssociation(desc.OutputWindow, DXGI_MWA_NO_ALT_ENTER);
    }
    factory->Release();
    if (FAILED(hr)) throw_hresult("IDXGIFactory::CreateSwapChain", hr);
#else
    (void)gpu_device;
    (void)window_host;
    throw std::runtime_error("D3D11 framebuffer presentation is only available on Windows");
#endif
}

D3D11FramebufferPresenter::~D3D11FramebufferPresenter() = default;

void D3D11FramebufferPresenter::present(const raster::Framebuffer& framebuffer, bool vsync) {
#ifdef _WIN32
    static_assert(sizeof(raster::Rgba8) == 4, "RGBA framebuffer pixels must be tightly packed");
    if (framebuffer.width() == 0 || framebuffer.height() == 0) {
        throw std::invalid_argument("cannot present an empty framebuffer");
    }

    if (framebuffer.width() != impl_->width || framebuffer.height() != impl_->height) {
        impl_->window->resize(framebuffer.width(), framebuffer.height());
        const HRESULT resize_hr = impl_->swap_chain->ResizeBuffers(
            1,
            checked_uint(framebuffer.width(), "swapchain width"),
            checked_uint(framebuffer.height(), "swapchain height"),
            DXGI_FORMAT_R8G8B8A8_UNORM,
            0);
        if (FAILED(resize_hr)) throw_hresult("IDXGISwapChain::ResizeBuffers", resize_hr);
        impl_->width = framebuffer.width();
        impl_->height = framebuffer.height();
    }

    ID3D11Texture2D* back_buffer = nullptr;
    HRESULT hr = impl_->swap_chain->GetBuffer(
        0,
        __uuidof(ID3D11Texture2D),
        reinterpret_cast<void**>(&back_buffer));
    if (FAILED(hr)) throw_hresult("IDXGISwapChain::GetBuffer", hr);

    const UINT row_pitch = checked_uint(framebuffer.width() * sizeof(raster::Rgba8), "framebuffer row pitch");
    impl_->gpu->impl_->context->UpdateSubresource(
        back_buffer,
        0,
        nullptr,
        framebuffer.pixels().data(),
        row_pitch,
        0);
    back_buffer->Release();

    hr = impl_->swap_chain->Present(vsync ? 1U : 0U, 0);
    if (FAILED(hr)) throw_hresult("IDXGISwapChain::Present", hr);
    ++impl_->frames;
    impl_->last_hash = framebuffer.hash64();
#else
    (void)framebuffer;
    (void)vsync;
    throw std::runtime_error("D3D11 framebuffer presentation is unavailable on this platform");
#endif
}

std::uint64_t D3D11FramebufferPresenter::frame_count() const noexcept {
    return impl_ == nullptr ? 0 : impl_->frames;
}

std::uint64_t D3D11FramebufferPresenter::last_present_hash() const noexcept {
    return impl_ == nullptr ? 0 : impl_->last_hash;
}

std::size_t D3D11FramebufferPresenter::width() const noexcept {
    return impl_ == nullptr ? 0 : impl_->width;
}

std::size_t D3D11FramebufferPresenter::height() const noexcept {
    return impl_ == nullptr ? 0 : impl_->height;
}

} // namespace spiral::gpu
