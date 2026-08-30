#include "spiral/ether_ai.hpp"

#ifdef _WIN32

#define NOMINMAX
#include <windows.h>
#include <commctrl.h>
#include <d2d1.h>
#include <dwrite.h>
#include <dwmapi.h>
#include <shellapi.h>
#include <wrl/client.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using Microsoft::WRL::ComPtr;

namespace {

constexpr UINT WM_ETHER_RESPONSE = WM_APP + 41;
constexpr UINT_PTR kAnimationTimer = 77;
constexpr float kHeaderHeight = 72.0F;
constexpr float kLeftRailWidth = 214.0F;
constexpr float kRightRailWidth = 250.0F;
constexpr float kComposerHeight = 112.0F;

std::wstring utf8_to_wide(std::string_view value) {
    if (value.empty()) return {};
    const int required = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, value.data(), static_cast<int>(value.size()), nullptr, 0);
    if (required <= 0) return L"[invalid UTF-8]";
    std::wstring out(static_cast<std::size_t>(required), L'\0');
    MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, value.data(), static_cast<int>(value.size()), out.data(), required);
    return out;
}

std::string wide_to_utf8(std::wstring_view value) {
    if (value.empty()) return {};
    const int required = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, value.data(), static_cast<int>(value.size()), nullptr, 0, nullptr, nullptr);
    if (required <= 0) return {};
    std::string out(static_cast<std::size_t>(required), '\0');
    WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, value.data(), static_cast<int>(value.size()), out.data(), required, nullptr, nullptr);
    return out;
}

D2D1_COLOR_F color(std::uint32_t rgb, float alpha = 1.0F) {
    return D2D1::ColorF(
        static_cast<float>((rgb >> 16U) & 0xFFU) / 255.0F,
        static_cast<float>((rgb >> 8U) & 0xFFU) / 255.0F,
        static_cast<float>(rgb & 0xFFU) / 255.0F,
        alpha);
}

D2D1_RECT_F inset(D2D1_RECT_F rect, float amount) {
    return D2D1::RectF(rect.left + amount, rect.top + amount, rect.right - amount, rect.bottom - amount);
}

struct UiMessage {
    bool user = false;
    std::wstring text;
};

struct ResponsePacket {
    std::wstring reply;
    spiral::ether_ai::Status status;
};

class App final {
public:
    App() : runtime_(spiral::ether_ai::standalone_host()) {
        cached_status_ = runtime_.status();
        messages_.push_back(UiMessage{
            false,
            L"Spiral Ether AI online. This is the standalone Windows host for the same AI runtime that can be embedded into EtherPlay, Hakui, EtherBeat, or another Spiral application. Backend is AUTO."});
    }

    ~App() {
        closing_.store(true);
        if (worker_.joinable()) worker_.join();
        if (edit_brush_ != nullptr) DeleteObject(edit_brush_);
        if (edit_font_ != nullptr) DeleteObject(edit_font_);
        if (edit_ != nullptr) RemovePropW(edit_, L"SPIRAL_ETHER_APP");
        CoUninitialize();
    }

    int run(HINSTANCE instance, int show) {
        instance_ = instance;
        if (FAILED(CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED))) return 2;
        if (!create_factories()) return 3;
        if (!create_window(show)) return 4;

        MSG message{};
        while (GetMessageW(&message, nullptr, 0, 0) > 0) {
            TranslateMessage(&message);
            DispatchMessageW(&message);
        }
        return static_cast<int>(message.wParam);
    }

private:
    static LRESULT CALLBACK window_proc(HWND hwnd, UINT msg, WPARAM wparam, LPARAM lparam) {
        App* self = reinterpret_cast<App*>(GetWindowLongPtrW(hwnd, GWLP_USERDATA));
        if (msg == WM_NCCREATE) {
            const auto* create = reinterpret_cast<CREATESTRUCTW*>(lparam);
            self = static_cast<App*>(create->lpCreateParams);
            SetWindowLongPtrW(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(self));
            self->hwnd_ = hwnd;
        }
        return self != nullptr ? self->handle_message(msg, wparam, lparam)
                               : DefWindowProcW(hwnd, msg, wparam, lparam);
    }

    static LRESULT CALLBACK edit_proc(HWND hwnd, UINT msg, WPARAM wparam, LPARAM lparam) {
        auto* self = static_cast<App*>(GetPropW(hwnd, L"SPIRAL_ETHER_APP"));
        if (self != nullptr && msg == WM_KEYDOWN && wparam == VK_RETURN) {
            if ((GetKeyState(VK_SHIFT) & 0x8000) == 0) {
                self->send_from_composer();
                return 0;
            }
        }
        if (self != nullptr && msg == WM_KEYDOWN && wparam == VK_ESCAPE) {
            SetWindowTextW(hwnd, L"");
            return 0;
        }
        return self != nullptr && self->old_edit_proc_ != nullptr
            ? CallWindowProcW(self->old_edit_proc_, hwnd, msg, wparam, lparam)
            : DefWindowProcW(hwnd, msg, wparam, lparam);
    }

    bool create_factories() {
        if (FAILED(D2D1CreateFactory(D2D1_FACTORY_TYPE_SINGLE_THREADED, d2d_factory_.GetAddressOf()))) return false;
        if (FAILED(DWriteCreateFactory(DWRITE_FACTORY_TYPE_SHARED, __uuidof(IDWriteFactory), reinterpret_cast<IUnknown**>(write_factory_.GetAddressOf())))) return false;

        const wchar_t* family = L"Segoe UI";
        if (FAILED(write_factory_->CreateTextFormat(family, nullptr, DWRITE_FONT_WEIGHT_NORMAL, DWRITE_FONT_STYLE_NORMAL,
                                                     DWRITE_FONT_STRETCH_NORMAL, 15.0F, L"en-us", body_format_.GetAddressOf()))) return false;
        if (FAILED(write_factory_->CreateTextFormat(family, nullptr, DWRITE_FONT_WEIGHT_SEMI_BOLD, DWRITE_FONT_STYLE_NORMAL,
                                                     DWRITE_FONT_STRETCH_NORMAL, 13.0F, L"en-us", small_format_.GetAddressOf()))) return false;
        if (FAILED(write_factory_->CreateTextFormat(family, nullptr, DWRITE_FONT_WEIGHT_SEMI_BOLD, DWRITE_FONT_STYLE_NORMAL,
                                                     DWRITE_FONT_STRETCH_NORMAL, 21.0F, L"en-us", title_format_.GetAddressOf()))) return false;
        if (FAILED(write_factory_->CreateTextFormat(family, nullptr, DWRITE_FONT_WEIGHT_BOLD, DWRITE_FONT_STYLE_NORMAL,
                                                     DWRITE_FONT_STRETCH_NORMAL, 11.0F, L"en-us", micro_format_.GetAddressOf()))) return false;
        body_format_->SetWordWrapping(DWRITE_WORD_WRAPPING_WRAP);
        small_format_->SetWordWrapping(DWRITE_WORD_WRAPPING_WRAP);
        return true;
    }

    bool create_window(int show) {
        WNDCLASSEXW wc{};
        wc.cbSize = sizeof(wc);
        wc.hInstance = instance_;
        wc.lpfnWndProc = &App::window_proc;
        wc.lpszClassName = L"SpiralEtherAIWindow";
        wc.hCursor = LoadCursorW(nullptr, IDC_ARROW);
        wc.hIcon = LoadIconW(nullptr, IDI_APPLICATION);
        wc.hbrBackground = nullptr;
        if (RegisterClassExW(&wc) == 0 && GetLastError() != ERROR_CLASS_ALREADY_EXISTS) return false;

        hwnd_ = CreateWindowExW(
            WS_EX_APPWINDOW | WS_EX_ACCEPTFILES,
            wc.lpszClassName,
            L"Spiral Ether AI",
            WS_POPUP | WS_THICKFRAME | WS_MINIMIZEBOX | WS_MAXIMIZEBOX,
            CW_USEDEFAULT, CW_USEDEFAULT, 1320, 820,
            nullptr, nullptr, instance_, this);
        if (hwnd_ == nullptr) return false;

        const MARGINS margins{1, 1, 1, 1};
        DwmExtendFrameIntoClientArea(hwnd_, &margins);
        SetWindowPos(hwnd_, nullptr, 0, 0, 0, 0,
                     SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_FRAMECHANGED);
        DragAcceptFiles(hwnd_, TRUE);

        edit_brush_ = CreateSolidBrush(RGB(12, 24, 31));
        edit_font_ = CreateFontW(-17, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE, DEFAULT_CHARSET,
                                 OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY,
                                 DEFAULT_PITCH | FF_DONTCARE, L"Segoe UI");
        edit_ = CreateWindowExW(
            0, L"EDIT", L"",
            WS_CHILD | WS_VISIBLE | ES_LEFT | ES_MULTILINE | ES_AUTOVSCROLL,
            0, 0, 100, 40, hwnd_, reinterpret_cast<HMENU>(1001), instance_, nullptr);
        if (edit_ == nullptr) return false;
        SendMessageW(edit_, WM_SETFONT, reinterpret_cast<WPARAM>(edit_font_), TRUE);
        SendMessageW(edit_, EM_SETMARGINS, EC_LEFTMARGIN | EC_RIGHTMARGIN, MAKELPARAM(9, 9));
        SetPropW(edit_, L"SPIRAL_ETHER_APP", this);
        old_edit_proc_ = reinterpret_cast<WNDPROC>(SetWindowLongPtrW(edit_, GWLP_WNDPROC, reinterpret_cast<LONG_PTR>(&App::edit_proc)));

        SetTimer(hwnd_, kAnimationTimer, 50, nullptr);
        ShowWindow(hwnd_, show);
        UpdateWindow(hwnd_);
        SetFocus(edit_);
        return true;
    }

    LRESULT handle_message(UINT msg, WPARAM wparam, LPARAM lparam) {
        switch (msg) {
            case WM_NCCALCSIZE:
                if (wparam != 0) return 0;
                break;
            case WM_NCHITTEST:
                return hit_test_frame(GET_X_LPARAM(lparam), GET_Y_LPARAM(lparam));
            case WM_SIZE:
                resize_target(LOWORD(lparam), HIWORD(lparam));
                layout_edit();
                InvalidateRect(hwnd_, nullptr, FALSE);
                return 0;
            case WM_GETMINMAXINFO: {
                auto* info = reinterpret_cast<MINMAXINFO*>(lparam);
                info->ptMinTrackSize.x = 980;
                info->ptMinTrackSize.y = 680;
                return 0;
            }
            case WM_PAINT:
                paint();
                return 0;
            case WM_ERASEBKGND:
                return 1;
            case WM_TIMER:
                animation_phase_ += 0.08F;
                if (animation_phase_ > 1000.0F) animation_phase_ = 0.0F;
                if (busy_.load()) InvalidateRect(hwnd_, nullptr, FALSE);
                return 0;
            case WM_MOUSEWHEEL: {
                const float delta = static_cast<float>(GET_WHEEL_DELTA_WPARAM(wparam)) / 120.0F;
                scroll_offset_ = std::clamp(scroll_offset_ - delta * 56.0F, 0.0F, std::max(0.0F, content_height_ - chat_view_height_));
                InvalidateRect(hwnd_, nullptr, FALSE);
                return 0;
            }
            case WM_LBUTTONUP:
                on_click(GET_X_LPARAM(lparam), GET_Y_LPARAM(lparam));
                return 0;
            case WM_COMMAND:
                return 0;
            case WM_CTLCOLOREDIT: {
                const HDC dc = reinterpret_cast<HDC>(wparam);
                SetTextColor(dc, RGB(226, 240, 244));
                SetBkColor(dc, RGB(12, 24, 31));
                return reinterpret_cast<LRESULT>(edit_brush_);
            }
            case WM_DROPFILES:
                on_drop(reinterpret_cast<HDROP>(wparam));
                return 0;
            case WM_ETHER_RESPONSE:
                on_response(reinterpret_cast<ResponsePacket*>(lparam));
                return 0;
            case WM_CLOSE:
                closing_.store(true);
                DestroyWindow(hwnd_);
                return 0;
            case WM_DESTROY:
                KillTimer(hwnd_, kAnimationTimer);
                PostQuitMessage(0);
                return 0;
        }
        return DefWindowProcW(hwnd_, msg, wparam, lparam);
    }

    LRESULT hit_test_frame(int screen_x, int screen_y) const {
        POINT point{screen_x, screen_y};
        ScreenToClient(hwnd_, &point);
        RECT client{};
        GetClientRect(hwnd_, &client);
        constexpr int edge = 8;
        const bool left = point.x < edge;
        const bool right = point.x >= client.right - edge;
        const bool top = point.y < edge;
        const bool bottom = point.y >= client.bottom - edge;
        if (top && left) return HTTOPLEFT;
        if (top && right) return HTTOPRIGHT;
        if (bottom && left) return HTBOTTOMLEFT;
        if (bottom && right) return HTBOTTOMRIGHT;
        if (left) return HTLEFT;
        if (right) return HTRIGHT;
        if (top) return HTTOP;
        if (bottom) return HTBOTTOM;
        if (point.y < static_cast<int>(kHeaderHeight) && point.x < client.right - 170) return HTCAPTION;
        return HTCLIENT;
    }

    void ensure_render_target() {
        if (target_ != nullptr) return;
        RECT rect{};
        GetClientRect(hwnd_, &rect);
        const D2D1_SIZE_U size = D2D1::SizeU(std::max(1L, rect.right), std::max(1L, rect.bottom));
        if (FAILED(d2d_factory_->CreateHwndRenderTarget(
                D2D1::RenderTargetProperties(D2D1_RENDER_TARGET_TYPE_DEFAULT),
                D2D1::HwndRenderTargetProperties(hwnd_, size),
                target_.GetAddressOf()))) return;

        target_->CreateSolidColorBrush(color(0xE3F5F7), text_brush_.GetAddressOf());
        target_->CreateSolidColorBrush(color(0x91B7BD), muted_brush_.GetAddressOf());
        target_->CreateSolidColorBrush(color(0x65DCE5), cyan_brush_.GetAddressOf());
        target_->CreateSolidColorBrush(color(0xE4B86A), amber_brush_.GetAddressOf());
        target_->CreateSolidColorBrush(color(0x0B151C, 0.95F), panel_brush_.GetAddressOf());
        target_->CreateSolidColorBrush(color(0x142630, 0.95F), panel2_brush_.GetAddressOf());
        target_->CreateSolidColorBrush(color(0x223B47, 0.95F), border_brush_.GetAddressOf());
        target_->CreateSolidColorBrush(color(0x193944, 0.94F), user_brush_.GetAddressOf());
        target_->CreateSolidColorBrush(color(0x0C1A22, 0.96F), assistant_brush_.GetAddressOf());
        target_->CreateSolidColorBrush(color(0xD75A5A), danger_brush_.GetAddressOf());

        D2D1_GRADIENT_STOP stops[3] = {
            {0.0F, color(0x071015)},
            {0.54F, color(0x0A171E)},
            {1.0F, color(0x10232B)},
        };
        ComPtr<ID2D1GradientStopCollection> stop_collection;
        if (SUCCEEDED(target_->CreateGradientStopCollection(stops, 3, stop_collection.GetAddressOf()))) {
            target_->CreateLinearGradientBrush(
                D2D1::LinearGradientBrushProperties(D2D1::Point2F(0, 0), D2D1::Point2F(1200, 800)),
                stop_collection.Get(), background_gradient_.GetAddressOf());
        }
    }

    void discard_target() {
        background_gradient_.Reset();
        danger_brush_.Reset();
        assistant_brush_.Reset();
        user_brush_.Reset();
        border_brush_.Reset();
        panel2_brush_.Reset();
        panel_brush_.Reset();
        amber_brush_.Reset();
        cyan_brush_.Reset();
        muted_brush_.Reset();
        text_brush_.Reset();
        target_.Reset();
    }

    void resize_target(UINT width, UINT height) {
        if (target_ != nullptr && width > 0 && height > 0) {
            if (FAILED(target_->Resize(D2D1::SizeU(width, height)))) discard_target();
        }
    }

    void layout_edit() {
        if (edit_ == nullptr) return;
        RECT rect{};
        GetClientRect(hwnd_, &rect);
        const int width = rect.right - rect.left;
        const int height = rect.bottom - rect.top;
        const int left = static_cast<int>(kLeftRailWidth + 28.0F);
        const int right_panel = width > 1120 ? static_cast<int>(kRightRailWidth + 26.0F) : 26;
        const int send_width = 96;
        const int edit_width = std::max(220, width - left - right_panel - send_width - 24);
        const int top = height - static_cast<int>(kComposerHeight) + 28;
        SetWindowPos(edit_, nullptr, left + 14, top + 8, edit_width - 20, 56, SWP_NOZORDER | SWP_NOACTIVATE);
    }

    float measure_text(const std::wstring& text, IDWriteTextFormat* format, float width) const {
        ComPtr<IDWriteTextLayout> layout;
        if (FAILED(write_factory_->CreateTextLayout(text.c_str(), static_cast<UINT32>(text.size()), format,
                                                     std::max(1.0F, width), 2000.0F, layout.GetAddressOf()))) return 24.0F;
        DWRITE_TEXT_METRICS metrics{};
        if (FAILED(layout->GetMetrics(&metrics))) return 24.0F;
        return metrics.height;
    }

    void draw_text(const std::wstring& text, IDWriteTextFormat* format, ID2D1Brush* brush, D2D1_RECT_F rect) {
        target_->DrawTextW(text.c_str(), static_cast<UINT32>(text.size()), format, rect, brush,
                           D2D1_DRAW_TEXT_OPTIONS_CLIP, DWRITE_MEASURING_MODE_NATURAL);
    }

    void rounded_panel(D2D1_RECT_F rect, float radius, ID2D1Brush* fill, ID2D1Brush* stroke = nullptr, float stroke_width = 1.0F) {
        const D2D1_ROUNDED_RECT rounded{rect, radius, radius};
        if (fill != nullptr) target_->FillRoundedRectangle(rounded, fill);
        if (stroke != nullptr) target_->DrawRoundedRectangle(rounded, stroke, stroke_width);
    }

    void paint() {
        PAINTSTRUCT ps{};
        BeginPaint(hwnd_, &ps);
        ensure_render_target();
        if (target_ == nullptr) {
            EndPaint(hwnd_, &ps);
            return;
        }

        RECT client{};
        GetClientRect(hwnd_, &client);
        const float width = static_cast<float>(client.right);
        const float height = static_cast<float>(client.bottom);
        const bool wide = width > 1120.0F;
        const float right_width = wide ? kRightRailWidth : 0.0F;

        target_->BeginDraw();
        target_->SetTransform(D2D1::Matrix3x2F::Identity());
        if (background_gradient_ != nullptr) target_->FillRectangle(D2D1::RectF(0, 0, width, height), background_gradient_.Get());
        else target_->Clear(color(0x071015));

        draw_atmosphere(width, height);
        draw_header(width);
        draw_left_rail(height);
        if (wide) draw_right_rail(width, height);
        draw_chat(D2D1::RectF(kLeftRailWidth + 18.0F, kHeaderHeight + 14.0F,
                              width - right_width - 18.0F, height - kComposerHeight));
        draw_composer(width, height, wide);

        const HRESULT end = target_->EndDraw();
        if (end == D2DERR_RECREATE_TARGET) discard_target();
        EndPaint(hwnd_, &ps);
    }

    void draw_atmosphere(float width, float height) {
        // Sparse HUD grid and horizon lines: Halo-era mood, original geometry.
        for (float x = kLeftRailWidth; x < width; x += 72.0F) {
            border_brush_->SetOpacity(0.10F);
            target_->DrawLine(D2D1::Point2F(x, kHeaderHeight), D2D1::Point2F(x, height), border_brush_.Get(), 1.0F);
        }
        for (float y = kHeaderHeight; y < height; y += 72.0F) {
            border_brush_->SetOpacity(0.08F);
            target_->DrawLine(D2D1::Point2F(kLeftRailWidth, y), D2D1::Point2F(width, y), border_brush_.Get(), 1.0F);
        }
        border_brush_->SetOpacity(1.0F);

        cyan_brush_->SetOpacity(0.18F);
        target_->DrawLine(D2D1::Point2F(kLeftRailWidth + 12, kHeaderHeight + 4),
                          D2D1::Point2F(width - 16, kHeaderHeight + 4), cyan_brush_.Get(), 1.0F);
        cyan_brush_->SetOpacity(1.0F);
    }

    void draw_header(float width) {
        target_->FillRectangle(D2D1::RectF(0, 0, width, kHeaderHeight), panel_brush_.Get());
        target_->DrawLine(D2D1::Point2F(0, kHeaderHeight - 1), D2D1::Point2F(width, kHeaderHeight - 1), border_brush_.Get(), 1.0F);

        draw_text(L"SPIRAL // ETHER AI", title_format_.Get(), text_brush_.Get(), D2D1::RectF(28, 15, 320, 45));
        draw_text(L"NATIVE INTELLIGENCE HOST", micro_format_.Get(), cyan_brush_.Get(), D2D1::RectF(29, 44, 310, 63));

        const std::wstring backend = utf8_to_wide(spiral::genius::gpt_backend_name(cached_status_.shell.gpt_backend));
        backend_rect_ = D2D1::RectF(width - 418, 18, width - 238, 53);
        rounded_panel(backend_rect_, 5.0F, panel2_brush_.Get(), cyan_brush_.Get());
        draw_text(L"BACKEND  " + backend, micro_format_.Get(), text_brush_.Get(), inset(backend_rect_, 10));

        min_rect_ = D2D1::RectF(width - 154, 14, width - 108, 58);
        max_rect_ = D2D1::RectF(width - 104, 14, width - 58, 58);
        close_rect_ = D2D1::RectF(width - 54, 14, width - 8, 58);
        rounded_panel(min_rect_, 4, panel2_brush_.Get(), border_brush_.Get());
        rounded_panel(max_rect_, 4, panel2_brush_.Get(), border_brush_.Get());
        rounded_panel(close_rect_, 4, panel2_brush_.Get(), border_brush_.Get());
        draw_text(L"—", small_format_.Get(), muted_brush_.Get(), D2D1::RectF(min_rect_.left + 15, min_rect_.top + 9, min_rect_.right, min_rect_.bottom));
        draw_text(IsZoomed(hwnd_) ? L"◱" : L"□", small_format_.Get(), muted_brush_.Get(), D2D1::RectF(max_rect_.left + 15, max_rect_.top + 9, max_rect_.right, max_rect_.bottom));
        draw_text(L"×", title_format_.Get(), danger_brush_.Get(), D2D1::RectF(close_rect_.left + 13, close_rect_.top + 4, close_rect_.right, close_rect_.bottom));
    }

    void draw_left_rail(float height) {
        target_->FillRectangle(D2D1::RectF(0, kHeaderHeight, kLeftRailWidth, height), panel_brush_.Get());
        target_->DrawLine(D2D1::Point2F(kLeftRailWidth - 1, kHeaderHeight), D2D1::Point2F(kLeftRailWidth - 1, height), border_brush_.Get(), 1.0F);

        new_chat_rect_ = D2D1::RectF(18, kHeaderHeight + 20, kLeftRailWidth - 18, kHeaderHeight + 62);
        rounded_panel(new_chat_rect_, 5, panel2_brush_.Get(), cyan_brush_.Get());
        draw_text(L"+  NEW SESSION", small_format_.Get(), text_brush_.Get(), inset(new_chat_rect_, 12));

        draw_text(L"HOST CONTEXT", micro_format_.Get(), muted_brush_.Get(), D2D1::RectF(20, kHeaderHeight + 92, kLeftRailWidth - 12, kHeaderHeight + 112));
        const std::pair<spiral::ether_ai::HostKind, const wchar_t*> hosts[] = {
            {spiral::ether_ai::HostKind::StandaloneWindows, L"WINDOWS"},
            {spiral::ether_ai::HostKind::EtherPlay, L"ETHERPLAY"},
            {spiral::ether_ai::HostKind::Hakui, L"HAKUI"},
            {spiral::ether_ai::HostKind::EtherBeat, L"ETHERBEAT"},
        };
        const float start = kHeaderHeight + 122;
        for (std::size_t i = 0; i < 4; ++i) {
            host_rects_[i] = D2D1::RectF(14, start + static_cast<float>(i) * 46, kLeftRailWidth - 14,
                                         start + 38 + static_cast<float>(i) * 46);
            const bool active = cached_status_.host.kind == hosts[i].first;
            if (active) rounded_panel(host_rects_[i], 4, panel2_brush_.Get(), cyan_brush_.Get());
            else target_->DrawRoundedRectangle(D2D1::RoundedRect(host_rects_[i], 4, 4), border_brush_.Get(), 1.0F);
            if (active) target_->FillRectangle(D2D1::RectF(host_rects_[i].left, host_rects_[i].top, host_rects_[i].left + 3, host_rects_[i].bottom), cyan_brush_.Get());
            draw_text(hosts[i].second, small_format_.Get(), active ? text_brush_.Get() : muted_brush_.Get(), inset(host_rects_[i], 12));
        }

        draw_text(L"PORTABLE CORE", micro_format_.Get(), muted_brush_.Get(), D2D1::RectF(20, height - 142, kLeftRailWidth - 12, height - 120));
        draw_text(L"One C++ session runtime\nshared across Spiral hosts.", small_format_.Get(), muted_brush_.Get(), D2D1::RectF(20, height - 116, kLeftRailWidth - 18, height - 72));
        cyan_brush_->SetOpacity(0.55F);
        target_->DrawLine(D2D1::Point2F(20, height - 48), D2D1::Point2F(kLeftRailWidth - 20, height - 48), cyan_brush_.Get(), 1.0F);
        cyan_brush_->SetOpacity(1.0F);
        draw_text(L"SPIRAL ETHER TECH", micro_format_.Get(), cyan_brush_.Get(), D2D1::RectF(20, height - 38, kLeftRailWidth - 12, height - 16));
    }

    void draw_right_rail(float width, float height) {
        const float left = width - kRightRailWidth;
        target_->FillRectangle(D2D1::RectF(left, kHeaderHeight, width, height), panel_brush_.Get());
        target_->DrawLine(D2D1::Point2F(left, kHeaderHeight), D2D1::Point2F(left, height), border_brush_.Get(), 1.0F);

        draw_text(L"SYSTEM STATE", micro_format_.Get(), muted_brush_.Get(), D2D1::RectF(left + 20, kHeaderHeight + 22, width - 20, kHeaderHeight + 42));
        D2D1_RECT_F card = D2D1::RectF(left + 16, kHeaderHeight + 52, width - 16, kHeaderHeight + 210);
        rounded_panel(card, 6, panel2_brush_.Get(), border_brush_.Get());

        const auto& s = cached_status_.shell;
        const std::wstring backend = utf8_to_wide(spiral::genius::gpt_backend_name(s.gpt_backend));
        const std::wstring host = utf8_to_wide(spiral::ether_ai::host_kind_name(cached_status_.host.kind));
        std::wstring gpu = s.gpu_available ? utf8_to_wide(s.gpu_adapter) : L"OFFLINE";
        if (gpu.size() > 27) gpu = gpu.substr(0, 27) + L"…";
        const std::wstring lines =
            L"HOST      " + host + L"\n" +
            L"BACKEND   " + backend + L"\n" +
            L"OPENAI    " + std::wstring(s.openai_key_present ? L"KEY PRESENT" : L"NO KEY") + L"\n" +
            L"LOCAL     " + std::wstring(s.model_loaded ? L"MODEL LOADED" : L"NO MODEL") + L"\n" +
            L"GPU       " + gpu;
        draw_text(lines, small_format_.Get(), text_brush_.Get(), inset(card, 14));

        draw_text(L"SESSION", micro_format_.Get(), muted_brush_.Get(), D2D1::RectF(left + 20, kHeaderHeight + 238, width - 20, kHeaderHeight + 258));
        D2D1_RECT_F session = D2D1::RectF(left + 16, kHeaderHeight + 268, width - 16, kHeaderHeight + 346);
        rounded_panel(session, 6, panel2_brush_.Get(), border_brush_.Get());
        draw_text(L"TURNS  " + std::to_wstring(cached_status_.shell.conversation_turns) + L"\nMODE   GPT-STYLE CHAT",
                  small_format_.Get(), text_brush_.Get(), inset(session, 14));

        draw_text(L"EMBED TARGETS", micro_format_.Get(), muted_brush_.Get(), D2D1::RectF(left + 20, kHeaderHeight + 376, width - 20, kHeaderHeight + 396));
        D2D1_RECT_F targets = D2D1::RectF(left + 16, kHeaderHeight + 406, width - 16, kHeaderHeight + 526);
        rounded_panel(targets, 6, panel2_brush_.Get(), border_brush_.Get());
        draw_text(L"ETHERPLAY   media assistant\nHAKUI       world assistant\nETHERBEAT   music assistant\nCUSTOM      host-neutral API",
                  small_format_.Get(), muted_brush_.Get(), inset(targets, 14));

        if (busy_.load()) {
            const float pulse = 0.45F + 0.35F * std::sin(animation_phase_);
            cyan_brush_->SetOpacity(pulse);
            draw_text(L"● THINKING", micro_format_.Get(), cyan_brush_.Get(), D2D1::RectF(left + 20, height - 58, width - 18, height - 32));
            cyan_brush_->SetOpacity(1.0F);
        } else {
            draw_text(L"● READY", micro_format_.Get(), cyan_brush_.Get(), D2D1::RectF(left + 20, height - 58, width - 18, height - 32));
        }
    }

    void draw_chat(D2D1_RECT_F rect) {
        chat_view_height_ = rect.bottom - rect.top;
        rounded_panel(rect, 8, assistant_brush_.Get(), border_brush_.Get());
        const D2D1_RECT_F clip = inset(rect, 12);
        target_->PushAxisAlignedClip(clip, D2D1_ANTIALIAS_MODE_PER_PRIMITIVE);

        const float available_width = clip.right - clip.left;
        const float bubble_width = std::min(720.0F, available_width - 64.0F);
        float y = clip.top + 12.0F - scroll_offset_;
        content_height_ = 0.0F;

        for (const auto& message : messages_) {
            const float text_width = std::max(120.0F, bubble_width - 30.0F);
            const float text_height = measure_text(message.text, body_format_.Get(), text_width);
            const float bubble_height = text_height + 32.0F;
            const float x = message.user ? clip.right - bubble_width - 16.0F : clip.left + 16.0F;
            const D2D1_RECT_F bubble = D2D1::RectF(x, y, x + bubble_width, y + bubble_height);
            if (bubble.bottom >= clip.top && bubble.top <= clip.bottom) {
                rounded_panel(bubble, 8, message.user ? user_brush_.Get() : panel2_brush_.Get(),
                              message.user ? cyan_brush_.Get() : border_brush_.Get());
                if (!message.user) {
                    target_->FillRectangle(D2D1::RectF(bubble.left, bubble.top + 8, bubble.left + 3, bubble.bottom - 8), cyan_brush_.Get());
                }
                draw_text(message.user ? L"YOU" : L"SPIRAL ETHER AI", micro_format_.Get(),
                          message.user ? cyan_brush_.Get() : amber_brush_.Get(),
                          D2D1::RectF(bubble.left + 15, bubble.top + 9, bubble.right - 12, bubble.top + 26));
                draw_text(message.text, body_format_.Get(), text_brush_.Get(),
                          D2D1::RectF(bubble.left + 15, bubble.top + 27, bubble.right - 15, bubble.bottom - 10));
            }
            y += bubble_height + 14.0F;
            content_height_ += bubble_height + 14.0F;
        }

        if (busy_.load()) {
            const float pulse = 0.42F + 0.38F * std::sin(animation_phase_);
            cyan_brush_->SetOpacity(pulse);
            const D2D1_RECT_F thinking = D2D1::RectF(clip.left + 16, y, clip.left + 236, y + 44);
            rounded_panel(thinking, 7, panel2_brush_.Get(), cyan_brush_.Get());
            draw_text(L"SPIRAL IS THINKING  · · ·", micro_format_.Get(), cyan_brush_.Get(), inset(thinking, 12));
            cyan_brush_->SetOpacity(1.0F);
            content_height_ += 58.0F;
        }

        target_->PopAxisAlignedClip();
    }

    void draw_composer(float width, float height, bool wide) {
        const float right_panel = wide ? kRightRailWidth + 18.0F : 18.0F;
        const D2D1_RECT_F outer = D2D1::RectF(kLeftRailWidth + 28.0F, height - kComposerHeight + 18.0F,
                                               width - right_panel, height - 18.0F);
        rounded_panel(outer, 9, panel_brush_.Get(), border_brush_.Get());
        target_->DrawLine(D2D1::Point2F(outer.left + 12, outer.top), D2D1::Point2F(outer.right - 12, outer.top), cyan_brush_.Get(), 1.0F);

        send_rect_ = D2D1::RectF(outer.right - 108, outer.top + 16, outer.right - 14, outer.bottom - 16);
        rounded_panel(send_rect_, 6, busy_.load() ? border_brush_.Get() : cyan_brush_.Get(), cyan_brush_.Get());
        draw_text(busy_.load() ? L"WAIT" : L"SEND", small_format_.Get(), busy_.load() ? muted_brush_.Get() : panel_brush_.Get(), inset(send_rect_, 20));

        draw_text(L"ENTER TO SEND   •   SHIFT+ENTER FOR NEW LINE   •   DROP A .BUNDLE TO LOAD LOCAL SPIRAL",
                  micro_format_.Get(), muted_brush_.Get(),
                  D2D1::RectF(outer.left + 14, outer.bottom - 18, send_rect_.left - 8, outer.bottom - 3));
    }

    static bool point_in(D2D1_RECT_F rect, float x, float y) {
        return x >= rect.left && x <= rect.right && y >= rect.top && y <= rect.bottom;
    }

    void on_click(int x_int, int y_int) {
        const float x = static_cast<float>(x_int);
        const float y = static_cast<float>(y_int);
        if (point_in(close_rect_, x, y)) { SendMessageW(hwnd_, WM_CLOSE, 0, 0); return; }
        if (point_in(min_rect_, x, y)) { ShowWindow(hwnd_, SW_MINIMIZE); return; }
        if (point_in(max_rect_, x, y)) { ShowWindow(hwnd_, IsZoomed(hwnd_) ? SW_RESTORE : SW_MAXIMIZE); return; }
        if (point_in(send_rect_, x, y)) { send_from_composer(); return; }
        if (point_in(new_chat_rect_, x, y)) {
            runtime_.clear();
            messages_.clear();
            messages_.push_back(UiMessage{false, L"New Spiral Ether AI session started."});
            cached_status_ = runtime_.status();
            scroll_offset_ = 0.0F;
            InvalidateRect(hwnd_, nullptr, FALSE);
            return;
        }
        if (point_in(backend_rect_, x, y) && !busy_.load()) {
            using spiral::genius::GptBackend;
            GptBackend next = GptBackend::Auto;
            if (cached_status_.shell.gpt_backend == GptBackend::Auto) next = GptBackend::OpenAI;
            else if (cached_status_.shell.gpt_backend == GptBackend::OpenAI) next = GptBackend::SpiralLocal;
            runtime_.set_backend(next);
            cached_status_ = runtime_.status();
            InvalidateRect(hwnd_, nullptr, FALSE);
            return;
        }
        if (busy_.load()) return;
        const spiral::ether_ai::HostDescriptor hosts[] = {
            spiral::ether_ai::standalone_host(),
            spiral::ether_ai::etherplay_host(),
            spiral::ether_ai::hakui_host(),
            spiral::ether_ai::etherbeat_host(),
        };
        for (std::size_t i = 0; i < 4; ++i) {
            if (point_in(host_rects_[i], x, y)) {
                runtime_.set_host(hosts[i]);
                cached_status_ = runtime_.status();
                messages_.push_back(UiMessage{false, L"Host context switched to " + utf8_to_wide(hosts[i].name) + L". The underlying Ether AI session runtime is unchanged."});
                scroll_to_bottom();
                InvalidateRect(hwnd_, nullptr, FALSE);
                return;
            }
        }
    }

    void send_from_composer() {
        if (busy_.load() || edit_ == nullptr) return;
        const int length = GetWindowTextLengthW(edit_);
        if (length <= 0 || length > 16000) return;
        std::wstring text(static_cast<std::size_t>(length), L'\0');
        GetWindowTextW(edit_, text.data(), length + 1);
        while (!text.empty() && (text.back() == L'\r' || text.back() == L'\n' || text.back() == L' ' || text.back() == L'\t')) text.pop_back();
        if (text.empty()) return;
        SetWindowTextW(edit_, L"");

        messages_.push_back(UiMessage{true, text});
        scroll_to_bottom();
        busy_.store(true);
        InvalidateRect(hwnd_, nullptr, FALSE);

        if (worker_.joinable()) worker_.join();
        const std::string input = wide_to_utf8(text);
        worker_ = std::thread([this, input] {
            std::string reply;
            if (!input.empty() && input.front() == '/') reply = runtime_.command(input);
            else reply = runtime_.send(input);
            auto* packet = new ResponsePacket{utf8_to_wide(reply), runtime_.status()};
            if (closing_.load() || !PostMessageW(hwnd_, WM_ETHER_RESPONSE, 0, reinterpret_cast<LPARAM>(packet))) delete packet;
        });
    }

    void on_response(ResponsePacket* packet) {
        std::unique_ptr<ResponsePacket> owned(packet);
        if (worker_.joinable()) worker_.join();
        busy_.store(false);
        if (owned != nullptr) {
            cached_status_ = owned->status;
            messages_.push_back(UiMessage{false, owned->reply.empty() ? L"[no response]" : owned->reply});
        }
        scroll_to_bottom();
        SetFocus(edit_);
        InvalidateRect(hwnd_, nullptr, FALSE);
    }

    void scroll_to_bottom() {
        scroll_offset_ = std::max(0.0F, content_height_ - chat_view_height_ + 90.0F);
    }

    void on_drop(HDROP drop) {
        wchar_t path[MAX_PATH]{};
        if (DragQueryFileW(drop, 0, path, MAX_PATH) > 0) {
            std::wstring dropped(path);
            const std::size_t dot = dropped.find_last_of(L'.');
            if (dot != std::wstring::npos) {
                std::wstring extension = dropped.substr(dot);
                std::transform(extension.begin(), extension.end(), extension.begin(), ::towlower);
                if (extension == L".bundle") {
                    std::string error;
                    if (runtime_.load_local_model(wide_to_utf8(dropped), &error)) {
                        messages_.push_back(UiMessage{false, L"Local Spiral model loaded: " + dropped});
                    } else {
                        messages_.push_back(UiMessage{false, L"Model load failed: " + utf8_to_wide(error)});
                    }
                    cached_status_ = runtime_.status();
                    scroll_to_bottom();
                    InvalidateRect(hwnd_, nullptr, FALSE);
                }
            }
        }
        DragFinish(drop);
    }

    HINSTANCE instance_ = nullptr;
    HWND hwnd_ = nullptr;
    HWND edit_ = nullptr;
    WNDPROC old_edit_proc_ = nullptr;
    HBRUSH edit_brush_ = nullptr;
    HFONT edit_font_ = nullptr;

    ComPtr<ID2D1Factory> d2d_factory_;
    ComPtr<IDWriteFactory> write_factory_;
    ComPtr<ID2D1HwndRenderTarget> target_;
    ComPtr<ID2D1LinearGradientBrush> background_gradient_;
    ComPtr<ID2D1SolidColorBrush> text_brush_;
    ComPtr<ID2D1SolidColorBrush> muted_brush_;
    ComPtr<ID2D1SolidColorBrush> cyan_brush_;
    ComPtr<ID2D1SolidColorBrush> amber_brush_;
    ComPtr<ID2D1SolidColorBrush> panel_brush_;
    ComPtr<ID2D1SolidColorBrush> panel2_brush_;
    ComPtr<ID2D1SolidColorBrush> border_brush_;
    ComPtr<ID2D1SolidColorBrush> user_brush_;
    ComPtr<ID2D1SolidColorBrush> assistant_brush_;
    ComPtr<ID2D1SolidColorBrush> danger_brush_;
    ComPtr<IDWriteTextFormat> body_format_;
    ComPtr<IDWriteTextFormat> small_format_;
    ComPtr<IDWriteTextFormat> title_format_;
    ComPtr<IDWriteTextFormat> micro_format_;

    spiral::ether_ai::Runtime runtime_;
    spiral::ether_ai::Status cached_status_;
    std::vector<UiMessage> messages_;
    std::thread worker_;
    std::atomic<bool> busy_{false};
    std::atomic<bool> closing_{false};

    D2D1_RECT_F min_rect_{};
    D2D1_RECT_F max_rect_{};
    D2D1_RECT_F close_rect_{};
    D2D1_RECT_F backend_rect_{};
    D2D1_RECT_F new_chat_rect_{};
    D2D1_RECT_F send_rect_{};
    D2D1_RECT_F host_rects_[4]{};
    float scroll_offset_ = 0.0F;
    float content_height_ = 0.0F;
    float chat_view_height_ = 1.0F;
    float animation_phase_ = 0.0F;
};

} // namespace

int WINAPI wWinMain(HINSTANCE instance, HINSTANCE, PWSTR, int show) {
    App app;
    return app.run(instance, show);
}

#else

int main() {
    return 0;
}

#endif
