#!/usr/bin/env python3
"""
Windows input logger for the overslam SLAM data-capture pipeline.

Captures raw mouse deltas (Win32 Raw Input API) and keyboard events
(WH_KEYBOARD_LL low-level hook) and writes them as JSONL. Every event
gets two timestamps:

  t_qpc_ns  -- QueryPerformanceCounter converted to nanoseconds.
               Monotonic, sub-microsecond resolution. Use for relative
               timing between events.
  t_wall_ns -- GetSystemTimePreciseAsFileTime in nanoseconds since the
               UNIX epoch. Use to align with the video recording when
               ffmpeg is invoked with -use_wallclock_as_timestamps 1.

Must run on the Windows host (python.exe), not WSL's python.
"""
import argparse
import ctypes
import json
import os
import queue
import sys
import threading
from ctypes import wintypes

if not sys.platform.startswith("win"):
    sys.stderr.write(
        "input-logger.py must run on the Windows host (use python.exe, "
        "not the WSL python).\n"
    )
    sys.exit(1)

# Pointer-sized integer type. wintypes.WPARAM/LPARAM are pointer-sized
# in modern Python but we redefine LRESULT explicitly to be safe.
if ctypes.sizeof(ctypes.c_void_p) == 8:
    LRESULT = ctypes.c_int64
    ULONG_PTR = ctypes.c_uint64
else:
    LRESULT = ctypes.c_long
    ULONG_PTR = ctypes.c_ulong

# ---------- Win32 constants ----------
HWND_MESSAGE = wintypes.HWND(-3)
WM_INPUT = 0x00FF
WM_QUIT = 0x0012
WM_DESTROY = 0x0002
WM_KEYDOWN = 0x0100
WM_KEYUP = 0x0101
WM_SYSKEYDOWN = 0x0104
WM_SYSKEYUP = 0x0105

WH_KEYBOARD_LL = 13
HC_ACTION = 0

RIDEV_INPUTSINK = 0x00000100
RID_INPUT = 0x10000003
RIM_TYPEMOUSE = 0

MOUSE_MOVE_ABSOLUTE = 0x01

# (bitmask, button name, edge) for RAWMOUSE.usButtonFlags
BTN_FLAGS = [
    (0x0001, "left",   "down"),
    (0x0002, "left",   "up"),
    (0x0004, "right",  "down"),
    (0x0008, "right",  "up"),
    (0x0010, "middle", "down"),
    (0x0020, "middle", "up"),
    (0x0040, "x1",     "down"),
    (0x0080, "x1",     "up"),
    (0x0100, "x2",     "down"),
    (0x0200, "x2",     "up"),
]
RI_MOUSE_WHEEL  = 0x0400
RI_MOUSE_HWHEEL = 0x0800

CTRL_C_EVENT = 0
CTRL_BREAK_EVENT = 1
CTRL_CLOSE_EVENT = 2

# ---------- Win32 structures ----------
class RAWINPUTDEVICE(ctypes.Structure):
    _fields_ = [
        ("usUsagePage", wintypes.USHORT),
        ("usUsage",     wintypes.USHORT),
        ("dwFlags",     wintypes.DWORD),
        ("hwndTarget",  wintypes.HWND),
    ]

class RAWINPUTHEADER(ctypes.Structure):
    _fields_ = [
        ("dwType",  wintypes.DWORD),
        ("dwSize",  wintypes.DWORD),
        ("hDevice", wintypes.HANDLE),
        ("wParam",  wintypes.WPARAM),
    ]

class _BTN_S(ctypes.Structure):
    _fields_ = [
        ("usButtonFlags", wintypes.USHORT),
        ("usButtonData",  wintypes.USHORT),
    ]

class _BTN_U(ctypes.Union):
    _fields_ = [
        ("ulButtons", wintypes.ULONG),
        ("s",         _BTN_S),
    ]

class RAWMOUSE(ctypes.Structure):
    _fields_ = [
        ("usFlags",            wintypes.USHORT),
        ("u",                  _BTN_U),
        ("ulRawButtons",       wintypes.ULONG),
        ("lLastX",             wintypes.LONG),
        ("lLastY",             wintypes.LONG),
        ("ulExtraInformation", wintypes.ULONG),
    ]

class RAWINPUT_MOUSE(ctypes.Structure):
    _fields_ = [
        ("header", RAWINPUTHEADER),
        ("mouse",  RAWMOUSE),
    ]

class KBDLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = [
        ("vkCode",      wintypes.DWORD),
        ("scanCode",    wintypes.DWORD),
        ("flags",       wintypes.DWORD),
        ("time",        wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]

WNDPROC = ctypes.WINFUNCTYPE(
    LRESULT, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM,
)

class WNDCLASSEXW(ctypes.Structure):
    _fields_ = [
        ("cbSize",        wintypes.UINT),
        ("style",         wintypes.UINT),
        ("lpfnWndProc",   WNDPROC),
        ("cbClsExtra",    ctypes.c_int),
        ("cbWndExtra",    ctypes.c_int),
        ("hInstance",     wintypes.HINSTANCE),
        ("hIcon",         wintypes.HICON),
        ("hCursor",       wintypes.HANDLE),
        ("hbrBackground", wintypes.HBRUSH),
        ("lpszMenuName",  wintypes.LPCWSTR),
        ("lpszClassName", wintypes.LPCWSTR),
        ("hIconSm",       wintypes.HICON),
    ]

HOOKPROC = ctypes.WINFUNCTYPE(
    LRESULT, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM,
)
PHANDLER_ROUTINE = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.DWORD)

# ---------- DLL bindings ----------
user32 = ctypes.WinDLL("user32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

user32.RegisterClassExW.argtypes = [ctypes.POINTER(WNDCLASSEXW)]
user32.RegisterClassExW.restype  = wintypes.ATOM
user32.CreateWindowExW.argtypes = [
    wintypes.DWORD, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    wintypes.HWND, wintypes.HMENU, wintypes.HINSTANCE, wintypes.LPVOID,
]
user32.CreateWindowExW.restype = wintypes.HWND
user32.DefWindowProcW.argtypes = [
    wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM,
]
user32.DefWindowProcW.restype = LRESULT
user32.RegisterRawInputDevices.argtypes = [
    ctypes.POINTER(RAWINPUTDEVICE), wintypes.UINT, wintypes.UINT,
]
user32.RegisterRawInputDevices.restype = wintypes.BOOL
user32.GetRawInputData.argtypes = [
    wintypes.HANDLE, wintypes.UINT, wintypes.LPVOID,
    ctypes.POINTER(wintypes.UINT), wintypes.UINT,
]
user32.GetRawInputData.restype = wintypes.UINT
user32.GetMessageW.argtypes = [
    wintypes.LPMSG, wintypes.HWND, wintypes.UINT, wintypes.UINT,
]
user32.GetMessageW.restype = ctypes.c_int
user32.TranslateMessage.argtypes = [wintypes.LPMSG]
user32.TranslateMessage.restype  = wintypes.BOOL
user32.DispatchMessageW.argtypes = [wintypes.LPMSG]
user32.DispatchMessageW.restype  = LRESULT
user32.PostThreadMessageW.argtypes = [
    wintypes.DWORD, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM,
]
user32.PostThreadMessageW.restype = wintypes.BOOL
user32.SetWindowsHookExW.argtypes = [
    ctypes.c_int, HOOKPROC, wintypes.HINSTANCE, wintypes.DWORD,
]
user32.SetWindowsHookExW.restype = wintypes.HHOOK
user32.CallNextHookEx.argtypes = [
    wintypes.HHOOK, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM,
]
user32.CallNextHookEx.restype = LRESULT
user32.UnhookWindowsHookEx.argtypes = [wintypes.HHOOK]
user32.UnhookWindowsHookEx.restype  = wintypes.BOOL

kernel32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
kernel32.GetModuleHandleW.restype  = wintypes.HMODULE
kernel32.GetCurrentThreadId.argtypes = []
kernel32.GetCurrentThreadId.restype  = wintypes.DWORD
kernel32.QueryPerformanceCounter.argtypes   = [ctypes.POINTER(ctypes.c_int64)]
kernel32.QueryPerformanceCounter.restype    = wintypes.BOOL
kernel32.QueryPerformanceFrequency.argtypes = [ctypes.POINTER(ctypes.c_int64)]
kernel32.QueryPerformanceFrequency.restype  = wintypes.BOOL
kernel32.GetSystemTimePreciseAsFileTime.argtypes = [ctypes.POINTER(wintypes.FILETIME)]
kernel32.GetSystemTimePreciseAsFileTime.restype  = None
kernel32.SetConsoleCtrlHandler.argtypes = [PHANDLER_ROUTINE, wintypes.BOOL]
kernel32.SetConsoleCtrlHandler.restype  = wintypes.BOOL

# ---------- Timestamps ----------
_qpc_freq = ctypes.c_int64(0)
kernel32.QueryPerformanceFrequency(ctypes.byref(_qpc_freq))
_QPC_FREQ = _qpc_freq.value
# FILETIME epoch (1601-01-01) to UNIX epoch, in 100ns units
_FT_TO_UNIX_100NS = 116444736000000000

def qpc_ns():
    v = ctypes.c_int64(0)
    kernel32.QueryPerformanceCounter(ctypes.byref(v))
    return (v.value * 1_000_000_000) // _QPC_FREQ

def wall_ns():
    ft = wintypes.FILETIME()
    kernel32.GetSystemTimePreciseAsFileTime(ctypes.byref(ft))
    ticks = (ft.dwHighDateTime << 32) | ft.dwLowDateTime
    return (ticks - _FT_TO_UNIX_100NS) * 100

# ---------- VK name table ----------
VK_NAMES = {
    0x08: "Back", 0x09: "Tab", 0x0D: "Return", 0x10: "Shift", 0x11: "Control",
    0x12: "Alt", 0x14: "CapsLock", 0x1B: "Escape", 0x20: "Space",
    0x21: "PgUp", 0x22: "PgDn", 0x23: "End", 0x24: "Home",
    0x25: "Left", 0x26: "Up", 0x27: "Right", 0x28: "Down",
    0x2D: "Insert", 0x2E: "Delete",
    0xA0: "LShift", 0xA1: "RShift", 0xA2: "LControl", 0xA3: "RControl",
    0xA4: "LAlt", 0xA5: "RAlt", 0x5B: "LWin", 0x5C: "RWin",
}
for _i in range(0x30, 0x3A):
    VK_NAMES[_i] = chr(_i)
for _i in range(0x41, 0x5B):
    VK_NAMES[_i] = chr(_i)
for _i in range(0x70, 0x88):
    VK_NAMES[_i] = f"F{_i - 0x6F}"

def vk_name(vk):
    return VK_NAMES.get(vk, f"VK_{vk}")

# ---------- Event queue ----------
_event_q = queue.Queue(maxsize=65536)
_dropped = 0

def emit(evt):
    """Called from WNDPROC and the LL keyboard hook. Must be cheap."""
    global _dropped
    try:
        _event_q.put_nowait(evt)
    except queue.Full:
        _dropped += 1

def writer_loop(out_path, stop_evt):
    with open(out_path, "w", encoding="utf-8", buffering=1) as f:
        f.write(json.dumps({
            "type": "header", "version": 1,
            "qpc_freq_hz": _QPC_FREQ,
            "started_qpc_ns":  qpc_ns(),
            "started_wall_ns": wall_ns(),
            "pid": os.getpid(),
        }) + "\n")
        while True:
            try:
                evt = _event_q.get(timeout=0.25)
            except queue.Empty:
                if stop_evt.is_set() and _event_q.empty():
                    break
                continue
            f.write(json.dumps(evt, separators=(",", ":")) + "\n")
        f.write(json.dumps({
            "type": "footer",
            "ended_qpc_ns":  qpc_ns(),
            "ended_wall_ns": wall_ns(),
            "dropped_events": _dropped,
        }) + "\n")

# ---------- WNDPROC: handles WM_INPUT (raw mouse) ----------
_ri_buf = (ctypes.c_ubyte * 256)()

def wnd_proc(hwnd, msg, wparam, lparam):
    if msg == WM_INPUT:
        # Capture timestamps before touching the API.
        t_qpc  = qpc_ns()
        t_wall = wall_ns()
        size = wintypes.UINT(ctypes.sizeof(_ri_buf))
        n = user32.GetRawInputData(
            wintypes.HANDLE(lparam), RID_INPUT,
            ctypes.byref(_ri_buf), ctypes.byref(size),
            ctypes.sizeof(RAWINPUTHEADER),
        )
        if n != 0xFFFFFFFF and n >= ctypes.sizeof(RAWINPUTHEADER):
            header = ctypes.cast(_ri_buf, ctypes.POINTER(RAWINPUTHEADER)).contents
            if header.dwType == RIM_TYPEMOUSE:
                m = ctypes.cast(_ri_buf, ctypes.POINTER(RAWINPUT_MOUSE)).contents.mouse
                if m.lLastX != 0 or m.lLastY != 0:
                    emit({
                        "t_qpc_ns": t_qpc, "t_wall_ns": t_wall,
                        "type": "mouse_move",
                        "dx": m.lLastX, "dy": m.lLastY,
                        "absolute": bool(m.usFlags & MOUSE_MOVE_ABSOLUTE),
                    })
                flags = m.u.s.usButtonFlags
                if flags:
                    for bit, name, edge in BTN_FLAGS:
                        if flags & bit:
                            emit({
                                "t_qpc_ns": t_qpc, "t_wall_ns": t_wall,
                                "type": "mouse_button",
                                "button": name, "event": edge,
                            })
                    if flags & RI_MOUSE_WHEEL:
                        delta = ctypes.c_int16(m.u.s.usButtonData).value
                        emit({
                            "t_qpc_ns": t_qpc, "t_wall_ns": t_wall,
                            "type": "mouse_wheel", "axis": "v", "delta": delta,
                        })
                    if flags & RI_MOUSE_HWHEEL:
                        delta = ctypes.c_int16(m.u.s.usButtonData).value
                        emit({
                            "t_qpc_ns": t_qpc, "t_wall_ns": t_wall,
                            "type": "mouse_wheel", "axis": "h", "delta": delta,
                        })
    return user32.DefWindowProcW(hwnd, msg, wparam, lparam)

# ---------- WH_KEYBOARD_LL hook ----------
_pressed = set()  # for auto-repeat detection; mutated only from the hook thread

def hook_proc(nCode, wParam, lParam):
    if nCode == HC_ACTION:
        t_qpc  = qpc_ns()
        t_wall = wall_ns()
        kb = ctypes.cast(lParam, ctypes.POINTER(KBDLLHOOKSTRUCT)).contents
        is_down = wParam in (WM_KEYDOWN, WM_SYSKEYDOWN)
        is_up   = wParam in (WM_KEYUP,   WM_SYSKEYUP)
        if is_down or is_up:
            vk = kb.vkCode
            repeat = False
            if is_down:
                repeat = vk in _pressed
                _pressed.add(vk)
            else:
                _pressed.discard(vk)
            emit({
                "t_qpc_ns": t_qpc, "t_wall_ns": t_wall,
                "type": "key",
                "event": "down" if is_down else "up",
                "vk": vk, "name": vk_name(vk),
                "scan": kb.scanCode,
                "extended": bool(kb.flags & 0x01),
                "repeat": repeat,
            })
    return user32.CallNextHookEx(None, nCode, wParam, lParam)

# Hold refs to keep callbacks alive
_wnd_proc_cb  = WNDPROC(wnd_proc)
_hook_proc_cb = HOOKPROC(hook_proc)

# ---------- Ctrl-C handling ----------
_main_tid = 0

def ctrl_handler(ctrl_type):
    if ctrl_type in (CTRL_C_EVENT, CTRL_BREAK_EVENT, CTRL_CLOSE_EVENT):
        user32.PostThreadMessageW(_main_tid, WM_QUIT, 0, 0)
        return True
    return False

_ctrl_handler_cb = PHANDLER_ROUTINE(ctrl_handler)

# ---------- Main ----------
def main():
    global _main_tid
    ap = argparse.ArgumentParser(
        description="Win32 input logger for the overslam SLAM pipeline",
    )
    ap.add_argument("-o", "--output", required=True, help="Output JSONL path")
    ap.add_argument("--no-keyboard", action="store_true",
                    help="Skip the WH_KEYBOARD_LL hook")
    ap.add_argument("--no-mouse", action="store_true",
                    help="Skip raw mouse input")
    args = ap.parse_args()

    if args.no_keyboard and args.no_mouse:
        sys.stderr.write("Nothing to do: both --no-keyboard and --no-mouse set.\n")
        sys.exit(2)

    _main_tid = kernel32.GetCurrentThreadId()
    kernel32.SetConsoleCtrlHandler(_ctrl_handler_cb, True)

    hinst = kernel32.GetModuleHandleW(None)

    class_name = "OverslamInputLoggerCls"
    wc = WNDCLASSEXW()
    wc.cbSize = ctypes.sizeof(WNDCLASSEXW)
    wc.lpfnWndProc = _wnd_proc_cb
    wc.hInstance = hinst
    wc.lpszClassName = class_name
    if not user32.RegisterClassExW(ctypes.byref(wc)):
        raise ctypes.WinError(ctypes.get_last_error())

    hwnd = user32.CreateWindowExW(
        0, class_name, "overslam-input-logger",
        0, 0, 0, 0, 0,
        HWND_MESSAGE, None, hinst, None,
    )
    if not hwnd:
        raise ctypes.WinError(ctypes.get_last_error())

    if not args.no_mouse:
        rid = RAWINPUTDEVICE()
        rid.usUsagePage = 0x01   # Generic Desktop
        rid.usUsage     = 0x02   # Mouse
        rid.dwFlags     = RIDEV_INPUTSINK
        rid.hwndTarget  = hwnd
        if not user32.RegisterRawInputDevices(
            ctypes.byref(rid), 1, ctypes.sizeof(RAWINPUTDEVICE)
        ):
            raise ctypes.WinError(ctypes.get_last_error())

    hook = None
    if not args.no_keyboard:
        hook = user32.SetWindowsHookExW(WH_KEYBOARD_LL, _hook_proc_cb, hinst, 0)
        if not hook:
            raise ctypes.WinError(ctypes.get_last_error())

    stop_evt = threading.Event()
    writer = threading.Thread(
        target=writer_loop, args=(args.output, stop_evt), daemon=False,
    )
    writer.start()

    sys.stdout.write(f"Logging to {args.output}. Press Ctrl+C to stop.\n")
    sys.stdout.flush()

    msg = wintypes.MSG()
    while True:
        rv = user32.GetMessageW(ctypes.byref(msg), None, 0, 0)
        if rv <= 0:
            break
        user32.TranslateMessage(ctypes.byref(msg))
        user32.DispatchMessageW(ctypes.byref(msg))

    if hook:
        user32.UnhookWindowsHookEx(hook)
    stop_evt.set()
    writer.join()
    sys.stdout.write(f"Stopped. Dropped events: {_dropped}\n")

if __name__ == "__main__":
    main()
