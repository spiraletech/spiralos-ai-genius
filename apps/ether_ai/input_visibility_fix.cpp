#ifdef _WIN32

#define NOMINMAX
#include <windows.h>

#include <iterator>

namespace {

constexpr int kComposerId = 1001;
thread_local bool g_adjusting = false;

bool class_is(HWND hwnd, const wchar_t* expected) {
    if (hwnd == nullptr) return false;
    wchar_t class_name[96]{};
    if (GetClassNameW(hwnd, class_name, static_cast<int>(std::size(class_name))) <= 0) return false;
    return lstrcmpiW(class_name, expected) == 0;
}

bool is_ether_window(HWND hwnd) {
    return class_is(hwnd, L"SpiralEtherAIWindow");
}

bool is_composer(HWND hwnd) {
    if (!class_is(hwnd, L"Edit")) return false;
    if (GetDlgCtrlID(hwnd) != kComposerId) return false;
    return is_ether_window(GetParent(hwnd));
}

void harden_composer(HWND edit) {
    if (edit == nullptr || g_adjusting) return;
    g_adjusting = true;

    HWND parent = GetParent(edit);
    if (parent != nullptr) {
        LONG_PTR parent_style = GetWindowLongPtrW(parent, GWL_STYLE);
        const LONG_PTR wanted_parent = parent_style | WS_CLIPCHILDREN | WS_CLIPSIBLINGS;
        if (wanted_parent != parent_style) {
            SetWindowLongPtrW(parent, GWL_STYLE, wanted_parent);
            SetWindowPos(parent, nullptr, 0, 0, 0, 0,
                         SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE | SWP_FRAMECHANGED);
        }
    }

    LONG_PTR edit_style = GetWindowLongPtrW(edit, GWL_STYLE);
    const LONG_PTR wanted_edit = edit_style | WS_CLIPSIBLINGS | WS_TABSTOP;
    if (wanted_edit != edit_style) SetWindowLongPtrW(edit, GWL_STYLE, wanted_edit);

    // Keep the native input control above the Direct2D parent surface. This is
    // deliberately a z-order-only operation; the app's normal layout owns size/position.
    SetWindowPos(edit, HWND_TOP, 0, 0, 0, 0,
                 SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_SHOWWINDOW);

    g_adjusting = false;
}

void repaint_composer(HWND edit) {
    if (!is_composer(edit)) return;
    harden_composer(edit);
    RedrawWindow(edit, nullptr, nullptr,
                 RDW_INVALIDATE | RDW_ERASE | RDW_FRAME | RDW_UPDATENOW);
}

bool should_repaint_after(UINT message) {
    switch (message) {
        case WM_CHAR:
        case WM_KEYUP:
        case WM_SETTEXT:
        case WM_PASTE:
        case WM_CUT:
        case WM_UNDO:
        case EM_REPLACESEL:
        case WM_SETFOCUS:
        case WM_MOUSEACTIVATE:
            return true;
        default:
            return false;
    }
}

LRESULT CALLBACK before_message_hook(int code, WPARAM wparam, LPARAM lparam) {
    if (code >= 0) {
        const auto* message = reinterpret_cast<const CWPSTRUCT*>(lparam);
        if (message != nullptr && is_composer(message->hwnd)) harden_composer(message->hwnd);
    }
    return CallNextHookEx(nullptr, code, wparam, lparam);
}

LRESULT CALLBACK after_message_hook(int code, WPARAM wparam, LPARAM lparam) {
    if (code >= 0) {
        const auto* message = reinterpret_cast<const CWPRETSTRUCT*>(lparam);
        if (message != nullptr) {
            if (is_composer(message->hwnd) && should_repaint_after(message->message)) {
                repaint_composer(message->hwnd);
            } else if (is_ether_window(message->hwnd) && message->message == WM_COMMAND &&
                       LOWORD(message->wParam) == kComposerId &&
                       (HIWORD(message->wParam) == EN_UPDATE || HIWORD(message->wParam) == EN_CHANGE)) {
                repaint_composer(GetDlgItem(message->hwnd, kComposerId));
            }
        }
    }
    return CallNextHookEx(nullptr, code, wparam, lparam);
}

class InputVisibilityFix final {
public:
    InputVisibilityFix() {
        const DWORD thread_id = GetCurrentThreadId();
        before_ = SetWindowsHookExW(WH_CALLWNDPROC, before_message_hook, nullptr, thread_id);
        after_ = SetWindowsHookExW(WH_CALLWNDPROCRET, after_message_hook, nullptr, thread_id);
    }

    ~InputVisibilityFix() {
        if (after_ != nullptr) UnhookWindowsHookEx(after_);
        if (before_ != nullptr) UnhookWindowsHookEx(before_);
    }

private:
    HHOOK before_ = nullptr;
    HHOOK after_ = nullptr;
};

// Directly linked into SpiralEtherAI.exe, so this installs on the UI thread
// before wWinMain creates the composer control.
InputVisibilityFix g_input_visibility_fix;

} // namespace

#endif
