#include "spiral/ether_ai.hpp"

#ifdef _WIN32

#define NOMINMAX
#include <windows.h>
#include <windowsx.h>
#include <d2d1.h>
#include <dwrite.h>
#include <dwmapi.h>
#include <shellapi.h>
#include <wrl/client.h>

#include <algorithm>
#include <array>
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
constexpr float kComposerHeight = 118.0F;
constexpr float kScrollbarWidth = 10.0F;

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
    return D2D1::ColorF(static_cast<float>((rgb >> 16U) & 0xFFU) / 255.0F,
                        static_cast<float>((rgb >> 8U) & 0xFFU) / 255.0F,
                        static_cast<float>(rgb & 0xFFU) / 255.0F,
                        alpha);
}

D2D1_RECT_F inset(D2D1_RECT_F rect, float amount) {
    return D2D1::RectF(rect.left + amount, rect.top + amount, rect.right - amount, rect.bottom - amount);
}

bool point_in(D2D1_RECT_F rect, float x, float y) {
    return x >= rect.left && x <= rect.right && y >= rect.top && y <= rect.bottom;
}

struct UiMessage { bool user = false; std::wstring text; };
struct ResponsePacket { std::wstring reply; spiral::ether_ai::Status status; };

class App final {
public:
    App() : runtime_(spiral::ether_ai::standalone_host()) {
        cached_status_ = runtime_.status();
        messages_.push_back(UiMessage{false,
            L"Spiral Ether AI online. Native organic cognition is active. The same C++ runtime can inhabit Windows, EtherPlay, Hakui, EtherBeat, or another Spiral host."});
        auto_scroll_pending_ = true;
    }

    ~App() {
        closing_.store(true);
        if (worker_.joinable()) worker_.join();
        if (edit_ != nullptr) RemovePropW(edit_, L"SPIRAL_ETHER_APP");
        if (edit_brush_ != nullptr) DeleteObject(edit_brush_);
        if (edit_font_ != nullptr) DeleteObject(edit_font_);
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
        return self != nullptr ? self->handle_message(msg, wparam, lparam) : DefWindowProcW(hwnd, msg, wparam, lparam);
    }

    static LRESULT CALLBACK edit_proc(HWND hwnd, UINT msg, WPARAM wparam, LPARAM lparam) {
        auto* self = static_cast<App*>(GetPropW(hwnd, L"SPIRAL_ETHER_APP"));
        if (self != nullptr && msg == WM_KEYDOWN && wparam == VK_RETURN && (GetKeyState(VK_SHIFT) & 0x8000) == 0) {
            self->send_from_composer();
            return 0;
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
        if (FAILED(write_factory_->CreateTextFormat(family, nullptr, DWRITE_FONT_WEIGHT_NORMAL, DWRITE_FONT_STYLE_NORMAL, DWRITE_FONT_STRETCH_NORMAL, 15.0F, L"en-us", body_format_.GetAddressOf()))) return false;
        if (FAILED(write_factory_->CreateTextFormat(family, nullptr, DWRITE_FONT_WEIGHT_SEMI_BOLD, DWRITE_FONT_STYLE_NORMAL, DWRITE_FONT_STRETCH_NORMAL, 13.0F, L"en-us", small_format_.GetAddressOf()))) return false;
        if (FAILED(write_factory_->CreateTextFormat(family, nullptr, DWRITE_FONT_WEIGHT_SEMI_BOLD, DWRITE_FONT_STYLE_NORMAL, DWRITE_FONT_STRETCH_NORMAL, 21.0F, L"en-us", title_format_.GetAddressOf()))) return false;
        if (FAILED(write_factory_->CreateTextFormat(family, nullptr, DWRITE_FONT_WEIGHT_BOLD, DWRITE_FONT_STYLE_NORMAL, DWRITE_FONT_STRETCH_NORMAL, 11.0F, L"en-us", micro_format_.GetAddressOf()))) return false;
        body_format_->SetWordWrapping(DWRITE_WORD_WRAPPING_WRAP);
        small_format_->SetWordWrapping(DWRITE_WORD_WRAPPING_WRAP);
        return true;
    }

    bool create_window(int show) {
        WNDCLASSEXW wc{};
        wc.cbSize = sizeof(wc);
        wc.style = CS_HREDRAW | CS_VREDRAW;
        wc.hInstance = instance_;
        wc.lpfnWndProc = &App::window_proc;
        wc.lpszClassName = L"SpiralEtherAIWindow";
        wc.hCursor = LoadCursorW(nullptr, IDC_ARROW);
        wc.hIcon = LoadIconW(nullptr, IDI_APPLICATION);
        if (RegisterClassExW(&wc) == 0 && GetLastError() != ERROR_CLASS_ALREADY_EXISTS) return false;

        hwnd_ = CreateWindowExW(WS_EX_APPWINDOW | WS_EX_ACCEPTFILES, wc.lpszClassName, L"Spiral Ether AI",
            WS_POPUP | WS_THICKFRAME | WS_MINIMIZEBOX | WS_MAXIMIZEBOX | WS_CLIPCHILDREN,
            CW_USEDEFAULT, CW_USEDEFAULT, 1320, 820, nullptr, nullptr, instance_, this);
        if (hwnd_ == nullptr) return false;

        const MARGINS margins{1,1,1,1};
        DwmExtendFrameIntoClientArea(hwnd_, &margins);
        SetWindowPos(hwnd_, nullptr, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_FRAMECHANGED);
        DragAcceptFiles(hwnd_, TRUE);

        edit_brush_ = CreateSolidBrush(RGB(12, 24, 31));
        edit_font_ = CreateFontW(-17, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE, DEFAULT_CHARSET,
            OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY, DEFAULT_PITCH | FF_DONTCARE, L"Segoe UI");

        edit_ = CreateWindowExW(0, L"EDIT", L"",
            WS_CHILD | WS_TABSTOP | WS_CLIPSIBLINGS | ES_LEFT | ES_MULTILINE | ES_AUTOVSCROLL | ES_WANTRETURN,
            0, 0, 1, 1, hwnd_, reinterpret_cast<HMENU>(1001), instance_, nullptr);
        if (edit_ == nullptr) return false;
        SendMessageW(edit_, WM_SETFONT, reinterpret_cast<WPARAM>(edit_font_), TRUE);
        SendMessageW(edit_, EM_SETMARGINS, EC_LEFTMARGIN | EC_RIGHTMARGIN, MAKELPARAM(10,10));
        SetPropW(edit_, L"SPIRAL_ETHER_APP", this);
        old_edit_proc_ = reinterpret_cast<WNDPROC>(SetWindowLongPtrW(edit_, GWLP_WNDPROC, reinterpret_cast<LONG_PTR>(&App::edit_proc)));

        layout_edit();
        ShowWindow(hwnd_, show);
        ShowWindow(edit_, SW_SHOW);
        SetWindowPos(edit_, HWND_TOP, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_SHOWWINDOW);
        RedrawWindow(edit_, nullptr, nullptr, RDW_INVALIDATE | RDW_ERASE | RDW_FRAME | RDW_UPDATENOW);
        SetTimer(hwnd_, kAnimationTimer, 50, nullptr);
        UpdateWindow(hwnd_);
        SetFocus(edit_);
        return true;
    }

    LRESULT handle_message(UINT msg, WPARAM wparam, LPARAM lparam) {
        switch (msg) {
            case WM_NCCALCSIZE: if (wparam != 0) return 0; break;
            case WM_NCHITTEST: return hit_test_frame(GET_X_LPARAM(lparam), GET_Y_LPARAM(lparam));
            case WM_SIZE: resize_target(LOWORD(lparam), HIWORD(lparam)); layout_edit(); clamp_scroll(); InvalidateRect(hwnd_, nullptr, FALSE); return 0;
            case WM_GETMINMAXINFO: { auto* info = reinterpret_cast<MINMAXINFO*>(lparam); info->ptMinTrackSize.x = 980; info->ptMinTrackSize.y = 680; return 0; }
            case WM_PAINT: paint(); return 0;
            case WM_ERASEBKGND: return 1;
            case WM_TIMER: animation_phase_ += 0.08F; if (animation_phase_ > 1000.0F) animation_phase_ = 0.0F; if (busy_.load()) InvalidateRect(hwnd_, nullptr, FALSE); return 0;
            case WM_MOUSEWHEEL: { const float notches = static_cast<float>(GET_WHEEL_DELTA_WPARAM(wparam)) / static_cast<float>(WHEEL_DELTA); scroll_by(-notches * 72.0F); return 0; }
            case WM_LBUTTONDOWN: on_left_button_down(GET_X_LPARAM(lparam), GET_Y_LPARAM(lparam)); return 0;
            case WM_MOUSEMOVE: if (dragging_scrollbar_) { drag_scrollbar(GET_Y_LPARAM(lparam)); return 0; } break;
            case WM_LBUTTONUP: on_left_button_up(GET_X_LPARAM(lparam), GET_Y_LPARAM(lparam)); return 0;
            case WM_CAPTURECHANGED: dragging_scrollbar_ = false; return 0;
            case WM_COMMAND:
                if (LOWORD(wparam) == 1001 && (HIWORD(wparam) == EN_CHANGE || HIWORD(wparam) == EN_UPDATE))
                    RedrawWindow(edit_, nullptr, nullptr, RDW_INVALIDATE | RDW_ERASE | RDW_UPDATENOW);
                return 0;
            case WM_CTLCOLOREDIT: { const HDC dc = reinterpret_cast<HDC>(wparam); SetTextColor(dc, RGB(226,240,244)); SetBkColor(dc, RGB(12,24,31)); return reinterpret_cast<LRESULT>(edit_brush_); }
            case WM_DROPFILES: on_drop(reinterpret_cast<HDROP>(wparam)); return 0;
            case WM_ETHER_RESPONSE: on_response(reinterpret_cast<ResponsePacket*>(lparam)); return 0;
            case WM_CLOSE: closing_.store(true); DestroyWindow(hwnd_); return 0;
            case WM_DESTROY: KillTimer(hwnd_, kAnimationTimer); PostQuitMessage(0); return 0;
        }
        return DefWindowProcW(hwnd_, msg, wparam, lparam);
    }

    LRESULT hit_test_frame(int sx, int sy) const {
        POINT p{sx,sy}; ScreenToClient(hwnd_, &p); RECT c{}; GetClientRect(hwnd_, &c); constexpr int e=8;
        const bool l=p.x<e,r=p.x>=c.right-e,t=p.y<e,b=p.y>=c.bottom-e;
        if(t&&l)return HTTOPLEFT; if(t&&r)return HTTOPRIGHT; if(b&&l)return HTBOTTOMLEFT; if(b&&r)return HTBOTTOMRIGHT;
        if(l)return HTLEFT; if(r)return HTRIGHT; if(t)return HTTOP; if(b)return HTBOTTOM;
        if(p.y<static_cast<int>(kHeaderHeight)&&p.x<c.right-170)return HTCAPTION; return HTCLIENT;
    }

    void ensure_render_target() {
        if (target_ != nullptr) return;
        RECT r{}; GetClientRect(hwnd_, &r); const D2D1_SIZE_U size=D2D1::SizeU(std::max(1L,r.right),std::max(1L,r.bottom));
        if (FAILED(d2d_factory_->CreateHwndRenderTarget(D2D1::RenderTargetProperties(D2D1_RENDER_TARGET_TYPE_DEFAULT), D2D1::HwndRenderTargetProperties(hwnd_,size), target_.GetAddressOf()))) return;
        target_->CreateSolidColorBrush(color(0xE3F5F7), text_brush_.GetAddressOf());
        target_->CreateSolidColorBrush(color(0x91B7BD), muted_brush_.GetAddressOf());
        target_->CreateSolidColorBrush(color(0x65DCE5), cyan_brush_.GetAddressOf());
        target_->CreateSolidColorBrush(color(0xE4B86A), amber_brush_.GetAddressOf());
        target_->CreateSolidColorBrush(color(0x0B151C,0.97F), panel_brush_.GetAddressOf());
        target_->CreateSolidColorBrush(color(0x142630,0.97F), panel2_brush_.GetAddressOf());
        target_->CreateSolidColorBrush(color(0x223B47,0.96F), border_brush_.GetAddressOf());
        target_->CreateSolidColorBrush(color(0x193944,0.95F), user_brush_.GetAddressOf());
        target_->CreateSolidColorBrush(color(0x0C1A22,0.97F), assistant_brush_.GetAddressOf());
        target_->CreateSolidColorBrush(color(0xD75A5A), danger_brush_.GetAddressOf());
        D2D1_GRADIENT_STOP stops[3]={{0.0F,color(0x071015)},{0.55F,color(0x0A171E)},{1.0F,color(0x10232B)}};
        ComPtr<ID2D1GradientStopCollection> col;
        if(SUCCEEDED(target_->CreateGradientStopCollection(stops,3,col.GetAddressOf()))) target_->CreateLinearGradientBrush(D2D1::LinearGradientBrushProperties(D2D1::Point2F(0,0),D2D1::Point2F(1400,820)),col.Get(),background_gradient_.GetAddressOf());
    }

    void discard_target() { background_gradient_.Reset(); danger_brush_.Reset(); assistant_brush_.Reset(); user_brush_.Reset(); border_brush_.Reset(); panel2_brush_.Reset(); panel_brush_.Reset(); amber_brush_.Reset(); cyan_brush_.Reset(); muted_brush_.Reset(); text_brush_.Reset(); target_.Reset(); }
    void resize_target(UINT w, UINT h) { if(target_!=nullptr&&w>0&&h>0&&FAILED(target_->Resize(D2D1::SizeU(w,h)))) discard_target(); }

    D2D1_RECT_F composer_rect(float w,float h) const { const bool wide=w>1120.0F; const float rp=wide?kRightRailWidth+18.0F:18.0F; return D2D1::RectF(kLeftRailWidth+28.0F,h-kComposerHeight+18.0F,w-rp,h-18.0F); }
    void layout_edit() {
        if(edit_==nullptr||hwnd_==nullptr)return; RECT r{}; GetClientRect(hwnd_,&r); const auto o=composer_rect(static_cast<float>(r.right),static_cast<float>(r.bottom));
        const int left=static_cast<int>(o.left+14.0F), top=static_cast<int>(o.top+15.0F), right=static_cast<int>(o.right-124.0F), bottom=static_cast<int>(o.bottom-30.0F);
        SetWindowPos(edit_,HWND_TOP,left,top,std::max(180,right-left),std::max(36,bottom-top),SWP_NOACTIVATE|SWP_SHOWWINDOW); InvalidateRect(edit_,nullptr,TRUE);
    }

    float measure_text(const std::wstring& text,IDWriteTextFormat* format,float width) const { ComPtr<IDWriteTextLayout> l; if(FAILED(write_factory_->CreateTextLayout(text.c_str(),static_cast<UINT32>(text.size()),format,std::max(1.0F,width),4000.0F,l.GetAddressOf())))return 24.0F; DWRITE_TEXT_METRICS m{}; if(FAILED(l->GetMetrics(&m)))return 24.0F; return m.height; }
    void draw_text(const std::wstring& t,IDWriteTextFormat* f,ID2D1Brush* b,D2D1_RECT_F r){ target_->DrawTextW(t.c_str(),static_cast<UINT32>(t.size()),f,r,b,D2D1_DRAW_TEXT_OPTIONS_CLIP,DWRITE_MEASURING_MODE_NATURAL); }
    void rounded_panel(D2D1_RECT_F r,float radius,ID2D1Brush* fill,ID2D1Brush* stroke=nullptr,float sw=1.0F){ const D2D1_ROUNDED_RECT rr{r,radius,radius}; if(fill)target_->FillRoundedRectangle(rr,fill); if(stroke)target_->DrawRoundedRectangle(rr,stroke,sw); }

    float max_scroll() const noexcept { return std::max(0.0F,content_height_-chat_view_height_); }
    void clamp_scroll(){ scroll_offset_=std::clamp(scroll_offset_,0.0F,max_scroll()); }
    void scroll_by(float d){ auto_scroll_pending_=false; scroll_offset_=std::clamp(scroll_offset_+d,0.0F,max_scroll()); InvalidateRect(hwnd_,nullptr,FALSE); }
    void request_scroll_to_bottom(){ auto_scroll_pending_=true; InvalidateRect(hwnd_,nullptr,FALSE); }
    void update_scrollbar_geometry(D2D1_RECT_F r){
        scroll_track_rect_=D2D1::RectF(r.right-18.0F,r.top+14.0F,r.right-8.0F,r.bottom-14.0F); const float th=std::max(1.0F,scroll_track_rect_.bottom-scroll_track_rect_.top);
        if(content_height_<=chat_view_height_+0.5F){scroll_thumb_rect_=scroll_track_rect_;return;}
        const float frac=std::clamp(chat_view_height_/content_height_,0.0F,1.0F), hh=std::clamp(th*frac,34.0F,th), travel=std::max(0.0F,th-hh), ratio=max_scroll()>0.0F?scroll_offset_/max_scroll():0.0F, top=scroll_track_rect_.top+travel*ratio;
        scroll_thumb_rect_=D2D1::RectF(scroll_track_rect_.left,top,scroll_track_rect_.right,top+hh);
    }

    void paint(){
        PAINTSTRUCT ps{}; BeginPaint(hwnd_,&ps); ensure_render_target(); if(!target_){EndPaint(hwnd_,&ps);return;} RECT c{}; GetClientRect(hwnd_,&c); const float w=static_cast<float>(c.right),h=static_cast<float>(c.bottom); const bool wide=w>1120.0F; const float rw=wide?kRightRailWidth:0.0F;
        target_->BeginDraw(); target_->SetTransform(D2D1::Matrix3x2F::Identity()); if(background_gradient_)target_->FillRectangle(D2D1::RectF(0,0,w,h),background_gradient_.Get());else target_->Clear(color(0x071015));
        draw_atmosphere(w,h); draw_header(w); draw_left_rail(h); if(wide)draw_right_rail(w,h); chat_rect_=D2D1::RectF(kLeftRailWidth+18.0F,kHeaderHeight+14.0F,w-rw-18.0F,h-kComposerHeight); draw_chat(chat_rect_); draw_composer(w,h);
        const HRESULT end=target_->EndDraw(); if(end==D2DERR_RECREATE_TARGET)discard_target(); EndPaint(hwnd_,&ps);
    }

    void draw_atmosphere(float w,float h){ border_brush_->SetOpacity(0.08F); for(float x=kLeftRailWidth;x<w;x+=72.0F)target_->DrawLine(D2D1::Point2F(x,kHeaderHeight),D2D1::Point2F(x,h),border_brush_.Get(),1.0F); for(float y=kHeaderHeight;y<h;y+=72.0F)target_->DrawLine(D2D1::Point2F(kLeftRailWidth,y),D2D1::Point2F(w,y),border_brush_.Get(),1.0F); border_brush_->SetOpacity(1.0F); }

    void draw_header(float w){
        target_->FillRectangle(D2D1::RectF(0,0,w,kHeaderHeight),panel_brush_.Get()); target_->DrawLine(D2D1::Point2F(0,kHeaderHeight-1),D2D1::Point2F(w,kHeaderHeight-1),border_brush_.Get(),1.0F);
        draw_text(L"SPIRAL // ETHER AI",title_format_.Get(),text_brush_.Get(),D2D1::RectF(28,15,320,45)); draw_text(L"NATIVE GENIUS HOLOGRAM",micro_format_.Get(),cyan_brush_.Get(),D2D1::RectF(29,44,310,63));
        const std::wstring backend=utf8_to_wide(spiral::genius::gpt_backend_name(cached_status_.shell.gpt_backend)); backend_rect_=D2D1::RectF(w-418,18,w-238,53); rounded_panel(backend_rect_,5,panel2_brush_.Get(),cyan_brush_.Get()); draw_text(L"BRAIN  "+backend,micro_format_.Get(),text_brush_.Get(),inset(backend_rect_,10));
        min_rect_=D2D1::RectF(w-154,14,w-108,58); max_rect_=D2D1::RectF(w-104,14,w-58,58); close_rect_=D2D1::RectF(w-54,14,w-8,58); rounded_panel(min_rect_,4,panel2_brush_.Get(),border_brush_.Get()); rounded_panel(max_rect_,4,panel2_brush_.Get(),border_brush_.Get()); rounded_panel(close_rect_,4,panel2_brush_.Get(),border_brush_.Get());
        draw_text(L"—",small_format_.Get(),muted_brush_.Get(),D2D1::RectF(min_rect_.left+15,min_rect_.top+9,min_rect_.right,min_rect_.bottom)); draw_text(IsZoomed(hwnd_)?L"◱":L"□",small_format_.Get(),muted_brush_.Get(),D2D1::RectF(max_rect_.left+15,max_rect_.top+9,max_rect_.right,max_rect_.bottom)); draw_text(L"×",title_format_.Get(),danger_brush_.Get(),D2D1::RectF(close_rect_.left+13,close_rect_.top+4,close_rect_.right,close_rect_.bottom));
    }

    void draw_left_rail(float h){
        target_->FillRectangle(D2D1::RectF(0,kHeaderHeight,kLeftRailWidth,h),panel_brush_.Get()); target_->DrawLine(D2D1::Point2F(kLeftRailWidth-1,kHeaderHeight),D2D1::Point2F(kLeftRailWidth-1,h),border_brush_.Get(),1.0F);
        new_chat_rect_=D2D1::RectF(18,kHeaderHeight+20,kLeftRailWidth-18,kHeaderHeight+62); rounded_panel(new_chat_rect_,5,panel2_brush_.Get(),cyan_brush_.Get()); draw_text(L"+  NEW SESSION",small_format_.Get(),text_brush_.Get(),inset(new_chat_rect_,12));
        draw_text(L"HOST CONTEXT",micro_format_.Get(),muted_brush_.Get(),D2D1::RectF(20,kHeaderHeight+92,kLeftRailWidth-12,kHeaderHeight+112));
        const std::pair<spiral::ether_ai::HostKind,const wchar_t*> hosts[]={{spiral::ether_ai::HostKind::StandaloneWindows,L"WINDOWS"},{spiral::ether_ai::HostKind::EtherPlay,L"ETHERPLAY"},{spiral::ether_ai::HostKind::Hakui,L"HAKUI"},{spiral::ether_ai::HostKind::EtherBeat,L"ETHERBEAT"}}; const float start=kHeaderHeight+122.0F;
        for(std::size_t i=0;i<4;++i){host_rects_[i]=D2D1::RectF(14,start+static_cast<float>(i)*46.0F,kLeftRailWidth-14,start+38.0F+static_cast<float>(i)*46.0F);const bool active=cached_status_.host.kind==hosts[i].first;if(active){rounded_panel(host_rects_[i],4,panel2_brush_.Get(),cyan_brush_.Get());target_->FillRectangle(D2D1::RectF(host_rects_[i].left,host_rects_[i].top,host_rects_[i].left+3,host_rects_[i].bottom),cyan_brush_.Get());}else target_->DrawRoundedRectangle(D2D1::RoundedRect(host_rects_[i],4,4),border_brush_.Get(),1.0F);draw_text(hosts[i].second,small_format_.Get(),active?text_brush_.Get():muted_brush_.Get(),inset(host_rects_[i],12));}
        draw_text(L"PORTABLE CORE",micro_format_.Get(),muted_brush_.Get(),D2D1::RectF(20,h-142,kLeftRailWidth-12,h-120)); draw_text(L"One native C++ mind\nshared across Spiral hosts.",small_format_.Get(),muted_brush_.Get(),D2D1::RectF(20,h-116,kLeftRailWidth-18,h-72)); draw_text(L"SPIRAL ETHER TECH",micro_format_.Get(),cyan_brush_.Get(),D2D1::RectF(20,h-38,kLeftRailWidth-12,h-16));
    }

    void draw_right_rail(float w,float h){
        const float left=w-kRightRailWidth; target_->FillRectangle(D2D1::RectF(left,kHeaderHeight,w,h),panel_brush_.Get()); target_->DrawLine(D2D1::Point2F(left,kHeaderHeight),D2D1::Point2F(left,h),border_brush_.Get(),1.0F);
        draw_text(L"SYSTEM STATE",micro_format_.Get(),muted_brush_.Get(),D2D1::RectF(left+20,kHeaderHeight+22,w-20,kHeaderHeight+42)); const D2D1_RECT_F card=D2D1::RectF(left+16,kHeaderHeight+52,w-16,kHeaderHeight+238); rounded_panel(card,6,panel2_brush_.Get(),border_brush_.Get()); const auto& s=cached_status_.shell;
        const std::wstring backend=utf8_to_wide(spiral::genius::gpt_backend_name(s.gpt_backend)),host=utf8_to_wide(spiral::ether_ai::host_kind_name(cached_status_.host.kind)); std::wstring gpu=s.gpu_available?utf8_to_wide(s.gpu_adapter):L"OFFLINE"; if(gpu.size()>27)gpu=gpu.substr(0,27)+L"…";
        const std::wstring state=L"HOST       "+host+L"\nBRAIN      "+backend+L"\nGPU        "+gpu+L"\nMEMORIES   "+std::to_wstring(s.organic_memories)+L"\nFOCUS      "+std::to_wstring(static_cast<int>(std::lround(s.organic_focus*100.0F)))+L"%\nCOHERENCE  "+std::to_wstring(static_cast<int>(std::lround(s.organic_coherence*100.0F)))+L"%"; draw_text(state,small_format_.Get(),text_brush_.Get(),inset(card,14));
        draw_text(L"COGNITION",micro_format_.Get(),muted_brush_.Get(),D2D1::RectF(left+20,kHeaderHeight+266,w-20,kHeaderHeight+286)); const D2D1_RECT_F cogcard=D2D1::RectF(left+16,kHeaderHeight+296,w-16,kHeaderHeight+438); rounded_panel(cogcard,6,panel2_brush_.Get(),border_brush_.Get());
        const std::wstring cog=L"LIRATEL    "+utf8_to_wide(s.cognition_liratel.empty()?std::string("IDLE"):s.cognition_liratel)+L"\nMIND       "+utf8_to_wide(s.cognition_mind.empty()?std::string("IDLE"):s.cognition_mind)+L"\nCODE       "+utf8_to_wide(s.cognition_code.empty()?std::string("IDLE"):s.cognition_code)+L"\nAUM        "+utf8_to_wide(s.cognition_aum.empty()?std::string("PRESERVE"):s.cognition_aum)+L"\nMALT       "+utf8_to_wide(s.cognition_malt.empty()?std::string("PASS"):s.cognition_malt)+L"\nLAMBDA     "+std::to_wstring(static_cast<int>(std::lround(s.cognition_lambda*100.0F)))+L"%"; draw_text(cog,small_format_.Get(),muted_brush_.Get(),inset(cogcard,14));
        draw_text(busy_.load()?L"● THINKING":L"● READY",micro_format_.Get(),cyan_brush_.Get(),D2D1::RectF(left+20,h-58,w-18,h-32));
    }

    void draw_chat(D2D1_RECT_F rect){
        rounded_panel(rect,8,assistant_brush_.Get(),border_brush_.Get()); D2D1_RECT_F clip=inset(rect,12.0F); clip.right-=kScrollbarWidth+12.0F; const float aw=clip.right-clip.left,bw=std::max(180.0F,std::min(720.0F,aw-40.0F)),tw=std::max(120.0F,bw-30.0F); std::vector<float> heights; heights.reserve(messages_.size()); content_height_=12.0F;
        for(const auto& m:messages_){const float bh=measure_text(m.text,body_format_.Get(),tw)+32.0F;heights.push_back(bh);content_height_+=bh+14.0F;} if(busy_.load())content_height_+=58.0F; chat_view_height_=std::max(1.0F,(clip.bottom-clip.top)-24.0F); if(auto_scroll_pending_){scroll_offset_=max_scroll();auto_scroll_pending_=false;}else clamp_scroll(); update_scrollbar_geometry(rect);
        target_->PushAxisAlignedClip(clip,D2D1_ANTIALIAS_MODE_PER_PRIMITIVE); float y=clip.top+12.0F-scroll_offset_;
        for(std::size_t i=0;i<messages_.size();++i){const auto&m=messages_[i];const float bh=heights[i],x=m.user?clip.right-bw-12.0F:clip.left+12.0F;const auto bubble=D2D1::RectF(x,y,x+bw,y+bh);if(bubble.bottom>=clip.top&&bubble.top<=clip.bottom){rounded_panel(bubble,8,m.user?user_brush_.Get():panel2_brush_.Get(),m.user?cyan_brush_.Get():border_brush_.Get());if(!m.user)target_->FillRectangle(D2D1::RectF(bubble.left,bubble.top+8,bubble.left+3,bubble.bottom-8),cyan_brush_.Get());draw_text(m.user?L"YOU":L"SPIRAL ETHER AI",micro_format_.Get(),m.user?cyan_brush_.Get():amber_brush_.Get(),D2D1::RectF(bubble.left+15,bubble.top+9,bubble.right-12,bubble.top+26));draw_text(m.text,body_format_.Get(),text_brush_.Get(),D2D1::RectF(bubble.left+15,bubble.top+27,bubble.right-15,bubble.bottom-10));}y+=bh+14.0F;}
        if(busy_.load()){const float pulse=0.42F+0.38F*std::sin(animation_phase_);cyan_brush_->SetOpacity(pulse);const auto thinking=D2D1::RectF(clip.left+12.0F,y,clip.left+242.0F,y+44.0F);rounded_panel(thinking,7,panel2_brush_.Get(),cyan_brush_.Get());draw_text(L"SPIRAL IS THINKING  · · ·",micro_format_.Get(),cyan_brush_.Get(),inset(thinking,12));cyan_brush_->SetOpacity(1.0F);} target_->PopAxisAlignedClip();
        border_brush_->SetOpacity(0.45F);rounded_panel(scroll_track_rect_,5,panel_brush_.Get(),border_brush_.Get());border_brush_->SetOpacity(1.0F);cyan_brush_->SetOpacity(max_scroll()>0.5F?0.85F:0.22F);rounded_panel(scroll_thumb_rect_,5,cyan_brush_.Get());cyan_brush_->SetOpacity(1.0F);
    }

    void draw_composer(float w,float h){const auto o=composer_rect(w,h);rounded_panel(o,9,panel_brush_.Get(),border_brush_.Get());target_->DrawLine(D2D1::Point2F(o.left+12,o.top),D2D1::Point2F(o.right-12,o.top),cyan_brush_.Get(),1.0F);send_rect_=D2D1::RectF(o.right-108,o.top+16,o.right-14,o.bottom-22);rounded_panel(send_rect_,6,busy_.load()?border_brush_.Get():cyan_brush_.Get(),cyan_brush_.Get());draw_text(busy_.load()?L"WAIT":L"SEND",small_format_.Get(),busy_.load()?muted_brush_.Get():panel_brush_.Get(),inset(send_rect_,20));draw_text(L"ENTER SEND   •   SHIFT+ENTER NEW LINE   •   MOUSE WHEEL / DRAG BAR TO SCROLL",micro_format_.Get(),muted_brush_.Get(),D2D1::RectF(o.left+14,o.bottom-18,send_rect_.left-8,o.bottom-3));}

    void on_left_button_down(int xi,int yi){const float x=static_cast<float>(xi),y=static_cast<float>(yi);if(max_scroll()>0.5F&&point_in(scroll_thumb_rect_,x,y)){dragging_scrollbar_=true;scrollbar_grab_offset_=y-scroll_thumb_rect_.top;SetCapture(hwnd_);return;}if(max_scroll()>0.5F&&point_in(scroll_track_rect_,x,y)){scroll_by((y<scroll_thumb_rect_.top?-1.0F:1.0F)*chat_view_height_*0.82F);return;}click_armed_=true;click_down_x_=x;click_down_y_=y;}
    void drag_scrollbar(int yi){if(!dragging_scrollbar_)return;const float y=static_cast<float>(yi),th=scroll_track_rect_.bottom-scroll_track_rect_.top,hh=scroll_thumb_rect_.bottom-scroll_thumb_rect_.top,travel=std::max(1.0F,th-hh),top=std::clamp(y-scrollbar_grab_offset_,scroll_track_rect_.top,scroll_track_rect_.bottom-hh),ratio=(top-scroll_track_rect_.top)/travel;auto_scroll_pending_=false;scroll_offset_=ratio*max_scroll();InvalidateRect(hwnd_,nullptr,FALSE);}
    void on_left_button_up(int xi,int yi){if(dragging_scrollbar_){dragging_scrollbar_=false;if(GetCapture()==hwnd_)ReleaseCapture();return;}if(!click_armed_)return;click_armed_=false;const float x=static_cast<float>(xi),y=static_cast<float>(yi);if(std::fabs(x-click_down_x_)<=5.0F&&std::fabs(y-click_down_y_)<=5.0F)on_click(x,y);}

    void on_click(float x,float y){
        if(point_in(close_rect_,x,y)){SendMessageW(hwnd_,WM_CLOSE,0,0);return;}if(point_in(min_rect_,x,y)){ShowWindow(hwnd_,SW_MINIMIZE);return;}if(point_in(max_rect_,x,y)){ShowWindow(hwnd_,IsZoomed(hwnd_)?SW_RESTORE:SW_MAXIMIZE);return;}if(point_in(send_rect_,x,y)){send_from_composer();return;}
        if(point_in(new_chat_rect_,x,y)){runtime_.clear();messages_.clear();messages_.push_back(UiMessage{false,L"New Spiral Ether AI session started."});cached_status_=runtime_.status();scroll_offset_=0.0F;auto_scroll_pending_=true;InvalidateRect(hwnd_,nullptr,FALSE);return;}
        if(point_in(backend_rect_,x,y)&&!busy_.load()){using spiral::genius::GptBackend;GptBackend next=GptBackend::Auto;if(cached_status_.shell.gpt_backend==GptBackend::Auto)next=GptBackend::OpenAI;else if(cached_status_.shell.gpt_backend==GptBackend::OpenAI)next=GptBackend::SpiralLocal;runtime_.set_backend(next);cached_status_=runtime_.status();InvalidateRect(hwnd_,nullptr,FALSE);return;}
        if(busy_.load())return;const spiral::ether_ai::HostDescriptor hosts[]={spiral::ether_ai::standalone_host(),spiral::ether_ai::etherplay_host(),spiral::ether_ai::hakui_host(),spiral::ether_ai::etherbeat_host()};for(std::size_t i=0;i<4;++i)if(point_in(host_rects_[i],x,y)){runtime_.set_host(hosts[i]);cached_status_=runtime_.status();messages_.push_back(UiMessage{false,L"Host context switched to "+utf8_to_wide(hosts[i].name)+L". The same organic session remains active."});request_scroll_to_bottom();return;}
    }

    void send_from_composer(){if(busy_.load()||edit_==nullptr)return;const int length=GetWindowTextLengthW(edit_);if(length<=0||length>16000)return;std::wstring buffer(static_cast<std::size_t>(length)+1U,L'\0');const int copied=GetWindowTextW(edit_,buffer.data(),length+1);if(copied<=0)return;buffer.resize(static_cast<std::size_t>(copied));while(!buffer.empty()&&(buffer.back()==L'\r'||buffer.back()==L'\n'||buffer.back()==L' '||buffer.back()==L'\t'))buffer.pop_back();if(buffer.empty())return;SetWindowTextW(edit_,L"");messages_.push_back(UiMessage{true,buffer});busy_.store(true);request_scroll_to_bottom();if(worker_.joinable())worker_.join();const std::string input=wide_to_utf8(buffer);worker_=std::thread([this,input]{std::string reply;if(!input.empty()&&input.front()=='/')reply=runtime_.command(input);else reply=runtime_.send(input);auto* packet=new ResponsePacket{utf8_to_wide(reply),runtime_.status()};if(closing_.load()||!PostMessageW(hwnd_,WM_ETHER_RESPONSE,0,reinterpret_cast<LPARAM>(packet)))delete packet;});}
    void on_response(ResponsePacket* packet){std::unique_ptr<ResponsePacket> owned(packet);if(worker_.joinable())worker_.join();busy_.store(false);if(owned){cached_status_=owned->status;messages_.push_back(UiMessage{false,owned->reply.empty()?L"[no response]":owned->reply});}request_scroll_to_bottom();SetFocus(edit_);RedrawWindow(edit_,nullptr,nullptr,RDW_INVALIDATE|RDW_ERASE|RDW_UPDATENOW);}
    void on_drop(HDROP drop){wchar_t path[MAX_PATH]{};if(DragQueryFileW(drop,0,path,MAX_PATH)>0){std::wstring dropped(path);const std::size_t dot=dropped.find_last_of(L'.');if(dot!=std::wstring::npos){std::wstring ext=dropped.substr(dot);std::transform(ext.begin(),ext.end(),ext.begin(),::towlower);if(ext==L".bundle"){std::string error;if(runtime_.load_local_model(wide_to_utf8(dropped),&error))messages_.push_back(UiMessage{false,L"Local Spiral model loaded: "+dropped});else messages_.push_back(UiMessage{false,L"Model load failed: "+utf8_to_wide(error)});cached_status_=runtime_.status();request_scroll_to_bottom();}}}DragFinish(drop);}

    HINSTANCE instance_=nullptr; HWND hwnd_=nullptr; HWND edit_=nullptr; WNDPROC old_edit_proc_=nullptr; HBRUSH edit_brush_=nullptr; HFONT edit_font_=nullptr;
    ComPtr<ID2D1Factory>d2d_factory_;ComPtr<IDWriteFactory>write_factory_;ComPtr<ID2D1HwndRenderTarget>target_;ComPtr<ID2D1LinearGradientBrush>background_gradient_;ComPtr<ID2D1SolidColorBrush>text_brush_,muted_brush_,cyan_brush_,amber_brush_,panel_brush_,panel2_brush_,border_brush_,user_brush_,assistant_brush_,danger_brush_;ComPtr<IDWriteTextFormat>body_format_,small_format_,title_format_,micro_format_;
    spiral::ether_ai::Runtime runtime_;spiral::ether_ai::Status cached_status_;std::vector<UiMessage>messages_;std::thread worker_;std::atomic<bool>busy_{false},closing_{false};
    D2D1_RECT_F min_rect_{},max_rect_{},close_rect_{},backend_rect_{},new_chat_rect_{},send_rect_{};std::array<D2D1_RECT_F,4>host_rects_{};D2D1_RECT_F chat_rect_{},scroll_track_rect_{},scroll_thumb_rect_{};
    float scroll_offset_=0.0F,content_height_=0.0F,chat_view_height_=1.0F,animation_phase_=0.0F,scrollbar_grab_offset_=0.0F,click_down_x_=0.0F,click_down_y_=0.0F;bool auto_scroll_pending_=true,dragging_scrollbar_=false,click_armed_=false;
};

} // namespace

int WINAPI wWinMain(HINSTANCE instance,HINSTANCE,PWSTR,int show){App app;return app.run(instance,show);}

#else
int main(){return 0;}
#endif
