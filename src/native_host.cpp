#include "spiral/native_host.hpp"

#include <algorithm>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

namespace spiral::host {

struct NativeWindowHost::Impl {
#ifdef _WIN32
    HWND hwnd = nullptr;
    bool close_requested = false;
    std::size_t client_width = 0;
    std::size_t client_height = 0;

    inline static constexpr wchar_t kWindowClassName[] = L"SpiralNativeWindowHost";

    static std::wstring utf16_from_utf8(const std::string& text) {
        if (text.empty()) return L"Spiral";
        const int needed = MultiByteToWideChar(
            CP_UTF8,
            MB_ERR_INVALID_CHARS,
            text.data(),
            static_cast<int>(text.size()),
            nullptr,
            0);
        if (needed <= 0) throw std::invalid_argument("window title is not valid UTF-8");
        std::wstring result(static_cast<std::size_t>(needed), L'\0');
        MultiByteToWideChar(
            CP_UTF8,
            MB_ERR_INVALID_CHARS,
            text.data(),
            static_cast<int>(text.size()),
            result.data(),
            needed);
        return result;
    }

    static LRESULT CALLBACK window_proc(HWND hwnd_value, UINT message, WPARAM wparam, LPARAM lparam) {
        auto* state = reinterpret_cast<Impl*>(GetWindowLongPtrW(hwnd_value, GWLP_USERDATA));
        if (message == WM_NCCREATE) {
            const auto* create = reinterpret_cast<const CREATESTRUCTW*>(lparam);
            state = static_cast<Impl*>(create->lpCreateParams);
            SetWindowLongPtrW(hwnd_value, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(state));
        }

        switch (message) {
            case WM_CLOSE:
                if (state != nullptr) state->close_requested = true;
                ShowWindow(hwnd_value, SW_HIDE);
                return 0;
            case WM_SIZE:
                if (state != nullptr) {
                    RECT rect{};
                    if (GetClientRect(hwnd_value, &rect)) {
                        state->client_width = static_cast<std::size_t>(std::max<LONG>(0, rect.right - rect.left));
                        state->client_height = static_cast<std::size_t>(std::max<LONG>(0, rect.bottom - rect.top));
                    }
                }
                break;
            default:
                break;
        }
        return DefWindowProcW(hwnd_value, message, wparam, lparam);
    }

    static void ensure_window_class() {
        static std::once_flag once;
        static std::exception_ptr registration_error;
        std::call_once(once, [] {
            WNDCLASSEXW window_class{};
            window_class.cbSize = sizeof(window_class);
            window_class.style = CS_HREDRAW | CS_VREDRAW;
            window_class.lpfnWndProc = &Impl::window_proc;
            window_class.hInstance = GetModuleHandleW(nullptr);
            window_class.hCursor = LoadCursorW(nullptr, MAKEINTRESOURCEW(32512));
            window_class.lpszClassName = kWindowClassName;
            if (RegisterClassExW(&window_class) == 0) {
                registration_error = std::make_exception_ptr(
                    std::runtime_error("RegisterClassExW failed for Spiral window host"));
            }
        });
        if (registration_error) std::rethrow_exception(registration_error);
    }

    static int checked_int(std::size_t value, const char* field) {
        if (value == 0 || value > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            throw std::invalid_argument(std::string(field) + " must fit a positive Win32 int");
        }
        return static_cast<int>(value);
    }
#endif
};

NativeWindowHost::NativeWindowHost() : impl_(std::make_unique<Impl>()) {}

NativeWindowHost::~NativeWindowHost() {
#ifdef _WIN32
    if (impl_ != nullptr && impl_->hwnd != nullptr) {
        DestroyWindow(impl_->hwnd);
        impl_->hwnd = nullptr;
    }
#endif
}

bool NativeWindowHost::platform_supported() noexcept {
#ifdef _WIN32
    return true;
#else
    return false;
#endif
}

void NativeWindowHost::create(const NativeWindowConfig& config) {
#ifdef _WIN32
    if (impl_->hwnd != nullptr) throw std::logic_error("native window already created");
    if (config.width == 0 || config.height == 0) throw std::invalid_argument("native window dimensions must be non-zero");
    Impl::ensure_window_class();

    const DWORD style = config.resizable
        ? WS_OVERLAPPEDWINDOW
        : (WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX);
    RECT outer{
        0,
        0,
        Impl::checked_int(config.width, "window width"),
        Impl::checked_int(config.height, "window height")};
    if (!AdjustWindowRect(&outer, style, FALSE)) throw std::runtime_error("AdjustWindowRect failed");
    const auto title = Impl::utf16_from_utf8(config.title);

    impl_->hwnd = CreateWindowExW(
        0,
        Impl::kWindowClassName,
        title.c_str(),
        style,
        CW_USEDEFAULT,
        CW_USEDEFAULT,
        outer.right - outer.left,
        outer.bottom - outer.top,
        nullptr,
        nullptr,
        GetModuleHandleW(nullptr),
        impl_.get());
    if (impl_->hwnd == nullptr) throw std::runtime_error("CreateWindowExW failed for Spiral window host");

    impl_->close_requested = false;
    RECT client{};
    if (GetClientRect(impl_->hwnd, &client)) {
        impl_->client_width = static_cast<std::size_t>(client.right - client.left);
        impl_->client_height = static_cast<std::size_t>(client.bottom - client.top);
    }
    ShowWindow(impl_->hwnd, config.visible ? SW_SHOW : SW_HIDE);
    UpdateWindow(impl_->hwnd);
#else
    (void)config;
    throw std::runtime_error("native Win32 window host is unavailable on this platform");
#endif
}

void NativeWindowHost::show(bool visible) {
#ifdef _WIN32
    if (impl_->hwnd == nullptr) throw std::logic_error("native window has not been created");
    ShowWindow(impl_->hwnd, visible ? SW_SHOW : SW_HIDE);
#else
    (void)visible;
    throw std::runtime_error("native Win32 window host is unavailable on this platform");
#endif
}

void NativeWindowHost::resize(std::size_t width, std::size_t height) {
#ifdef _WIN32
    if (impl_->hwnd == nullptr) throw std::logic_error("native window has not been created");
    if (width == 0 || height == 0) throw std::invalid_argument("native window dimensions must be non-zero");
    const LONG_PTR style = GetWindowLongPtrW(impl_->hwnd, GWL_STYLE);
    RECT outer{
        0,
        0,
        Impl::checked_int(width, "window width"),
        Impl::checked_int(height, "window height")};
    if (!AdjustWindowRect(&outer, static_cast<DWORD>(style), FALSE)) throw std::runtime_error("AdjustWindowRect failed");
    if (!SetWindowPos(
            impl_->hwnd,
            nullptr,
            0,
            0,
            outer.right - outer.left,
            outer.bottom - outer.top,
            SWP_NOMOVE | SWP_NOZORDER | SWP_NOACTIVATE)) {
        throw std::runtime_error("SetWindowPos failed while resizing native window");
    }
    RECT client{};
    if (GetClientRect(impl_->hwnd, &client)) {
        impl_->client_width = static_cast<std::size_t>(client.right - client.left);
        impl_->client_height = static_cast<std::size_t>(client.bottom - client.top);
    }
#else
    (void)width; (void)height;
    throw std::runtime_error("native Win32 window host is unavailable on this platform");
#endif
}

void NativeWindowHost::pump_events() {
#ifdef _WIN32
    MSG message{};
    while (PeekMessageW(&message, nullptr, 0, 0, PM_REMOVE)) {
        TranslateMessage(&message);
        DispatchMessageW(&message);
    }
#endif
}

bool NativeWindowHost::created() const noexcept {
#ifdef _WIN32
    return impl_ != nullptr && impl_->hwnd != nullptr;
#else
    return false;
#endif
}

bool NativeWindowHost::close_requested() const noexcept {
#ifdef _WIN32
    return impl_ != nullptr && impl_->close_requested;
#else
    return false;
#endif
}

std::size_t NativeWindowHost::client_width() const noexcept {
#ifdef _WIN32
    return impl_ == nullptr ? 0 : impl_->client_width;
#else
    return 0;
#endif
}

std::size_t NativeWindowHost::client_height() const noexcept {
#ifdef _WIN32
    return impl_ == nullptr ? 0 : impl_->client_height;
#else
    return 0;
#endif
}

void* NativeWindowHost::native_handle() const noexcept {
#ifdef _WIN32
    return impl_ == nullptr ? nullptr : reinterpret_cast<void*>(impl_->hwnd);
#else
    return nullptr;
#endif
}

} // namespace spiral::host
