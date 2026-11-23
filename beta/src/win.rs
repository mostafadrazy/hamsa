//! Small Windows helpers: window handles and styles, monitors, the cursor, dark mode, local time, single instance.

use std::ffi::c_void;

use slint::winit_030::winit::raw_window_handle::{HasWindowHandle, RawWindowHandle};
use slint::winit_030::WinitWindowAccessor;
use windows_sys::Win32::Foundation::{CloseHandle, GetLastError, ERROR_ALREADY_EXISTS, FILETIME, HWND, POINT, RECT, SYSTEMTIME};
use windows_sys::Win32::Graphics::Gdi::{
    CreateRectRgn, EnumDisplayMonitors, GetMonitorInfoW, MonitorFromPoint, HDC, HMONITOR, MONITORINFO, MONITORINFOEXW,
    MONITOR_DEFAULTTONULL, MONITOR_DEFAULTTOPRIMARY, SetWindowRgn,
};
use windows_sys::Win32::System::Registry::{RegGetValueW, HKEY_CURRENT_USER, RRF_RT_REG_DWORD};
use windows_sys::Win32::System::Threading::{CreateEventW, CreateMutexW, OpenEventW, SetEvent, WaitForSingleObject, EVENT_MODIFY_STATE, INFINITE};
use windows_sys::Win32::System::Time::{FileTimeToSystemTime, SystemTimeToTzSpecificLocalTime};
use windows_sys::Win32::UI::HiDpi::{GetDpiForMonitor, MDT_EFFECTIVE_DPI};
use windows_sys::Win32::UI::Input::KeyboardAndMouse::{GetAsyncKeyState, VK_LBUTTON, VK_RBUTTON};
use windows_sys::Win32::UI::WindowsAndMessaging::{
    GetCursorPos, GetForegroundWindow, GetWindowLongPtrW, GetWindowRect, SetForegroundWindow, SetWindowLongPtrW, SetWindowPos, ShowWindow,
    SystemParametersInfoW, SPI_GETCLIENTAREAANIMATION,
    GWL_EXSTYLE, HWND_TOPMOST, SWP_NOACTIVATE, SWP_NOSIZE, SWP_NOZORDER, SW_HIDE, SW_SHOWNOACTIVATE, WS_EX_APPWINDOW, WS_EX_NOACTIVATE,
    WS_EX_TOOLWINDOW,
};

/// The native handle of a Slint window (once it has been shown).
pub fn hwnd(window: &slint::Window) -> Option<HWND> {
    window
        .with_winit_window(|w| match w.window_handle().ok()?.as_raw() {
            RawWindowHandle::Win32(h) => Some(h.hwnd.get() as HWND),
            _ => None,
        })
        .flatten()
}

/// A tool window that never takes focus: no taskbar button, and clicking it leaves focus in the app you're typing in.
pub fn make_overlay(hwnd: HWND) {
    unsafe {
        let style = GetWindowLongPtrW(hwnd, GWL_EXSTYLE);
        let style = (style | WS_EX_NOACTIVATE as isize | WS_EX_TOOLWINDOW as isize) & !(WS_EX_APPWINDOW as isize);
        SetWindowLongPtrW(hwnd, GWL_EXSTYLE, style);
    }
}

pub fn show_no_activate(hwnd: HWND, show: bool) {
    unsafe {
        ShowWindow(hwnd, if show { SW_SHOWNOACTIVATE } else { SW_HIDE });
    }
}

/// Whether this window is the one the user is typing into.
pub fn is_foreground(hwnd: HWND) -> bool {
    unsafe { GetForegroundWindow() == hwnd }
}

pub fn bring_to_front(hwnd: HWND) {
    unsafe {
        SetForegroundWindow(hwnd);
    }
}

/// Move and size a window in physical px, keeping it topmost and without activating it.
pub fn set_rect(hwnd: HWND, x: i32, y: i32, w: i32, h: i32) {
    unsafe {
        SetWindowPos(hwnd, HWND_TOPMOST, x, y, w, h, SWP_NOACTIVATE);
    }
}

/// Move a window without resizing it (same flags as `set_rect`).
pub fn move_to(hwnd: HWND, x: i32, y: i32) {
    unsafe {
        SetWindowPos(hwnd, HWND_TOPMOST, x, y, 0, 0, SWP_NOACTIVATE | SWP_NOSIZE);
    }
}

/// Limit the part of the window that is drawn and receives the mouse (window-relative px); the rest is click-through.
pub fn set_region(hwnd: HWND, left: i32, top: i32, right: i32, bottom: i32) {
    unsafe {
        // The system owns the region after this call, so it is not deleted here.
        let rgn = CreateRectRgn(left, top, right, bottom);
        SetWindowRgn(hwnd, rgn, 1);
    }
}

/// Centre a normal window on the work area of the screen the mouse is on (without changing its size or stacking).
pub fn center(hwnd: HWND) {
    let (cx, cy) = cursor();
    let (Some(m), Some(r)) = (monitor_at(cx, cy).or_else(primary_monitor), window_rect(hwnd)) else { return };
    let (w, h) = (r.right - r.left, r.bottom - r.top);
    let x = m.work.left + ((m.work.right - m.work.left - w) / 2).max(0);
    let y = m.work.top + ((m.work.bottom - m.work.top - h) / 2).max(0);
    unsafe {
        SetWindowPos(hwnd, std::ptr::null_mut(), x, y, 0, 0, SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE);
    }
}

pub fn window_rect(hwnd: HWND) -> Option<RECT> {
    let mut r = RECT { left: 0, top: 0, right: 0, bottom: 0 };
    (unsafe { GetWindowRect(hwnd, &mut r) } != 0).then_some(r)
}

pub fn cursor() -> (i32, i32) {
    let mut p = POINT { x: 0, y: 0 };
    unsafe { GetCursorPos(&mut p) };
    (p.x, p.y)
}

pub fn left_button_down() -> bool {
    unsafe { (GetAsyncKeyState(VK_LBUTTON as i32) as u16 & 0x8000) != 0 }
}

pub fn any_button_down() -> bool {
    left_button_down() || unsafe { (GetAsyncKeyState(VK_RBUTTON as i32) as u16 & 0x8000) != 0 }
}

pub fn contains(r: &RECT, (x, y): (i32, i32)) -> bool {
    x >= r.left && x < r.right && y >= r.top && y < r.bottom
}

/// A display: the area without the taskbar, scale (1.0 = 96 dpi) and device name.
#[derive(Clone)]
pub struct Monitor {
    pub work: RECT,
    pub scale: f64,
    pub name: String,
}

fn monitor_info(h: HMONITOR) -> Option<Monitor> {
    unsafe {
        let mut info: MONITORINFOEXW = std::mem::zeroed();
        info.monitorInfo.cbSize = std::mem::size_of::<MONITORINFOEXW>() as u32;
        if GetMonitorInfoW(h, &mut info as *mut _ as *mut MONITORINFO) == 0 {
            return None;
        }
        let (mut dx, mut dy) = (96u32, 96u32);
        GetDpiForMonitor(h, MDT_EFFECTIVE_DPI, &mut dx, &mut dy);
        let len = info.szDevice.iter().position(|&c| c == 0).unwrap_or(info.szDevice.len());
        Some(Monitor {
            work: info.monitorInfo.rcWork,
            scale: dx as f64 / 96.0,
            name: String::from_utf16_lossy(&info.szDevice[..len]),
        })
    }
}

pub fn primary_monitor() -> Option<Monitor> {
    monitor_info(unsafe { MonitorFromPoint(POINT { x: 0, y: 0 }, MONITOR_DEFAULTTOPRIMARY) })
}

/// The monitor containing a point, if any.
pub fn monitor_at(x: i32, y: i32) -> Option<Monitor> {
    let h = unsafe { MonitorFromPoint(POINT { x, y }, MONITOR_DEFAULTTONULL) };
    if h.is_null() {
        None
    } else {
        monitor_info(h)
    }
}

pub fn monitor_named(name: &str) -> Option<Monitor> {
    unsafe extern "system" fn collect(h: HMONITOR, _: HDC, _: *mut RECT, data: isize) -> i32 {
        let list = &mut *(data as *mut Vec<HMONITOR>);
        list.push(h);
        1
    }
    let mut list: Vec<HMONITOR> = Vec::new();
    unsafe { EnumDisplayMonitors(std::ptr::null_mut(), std::ptr::null(), Some(collect), &mut list as *mut _ as isize) };
    list.into_iter().filter_map(monitor_info).find(|m| m.name == name)
}

/// Windows "apps use dark mode" setting.
pub fn system_dark() -> bool {
    let key: Vec<u16> = "Software\\Microsoft\\Windows\\CurrentVersion\\Themes\\Personalize\0".encode_utf16().collect();
    let value: Vec<u16> = "AppsUseLightTheme\0".encode_utf16().collect();
    let mut data: u32 = 1;
    let mut size = std::mem::size_of::<u32>() as u32;
    let ok = unsafe {
        RegGetValueW(
            HKEY_CURRENT_USER,
            key.as_ptr(),
            value.as_ptr(),
            RRF_RT_REG_DWORD,
            std::ptr::null_mut(),
            &mut data as *mut u32 as *mut c_void,
            &mut size,
        )
    };
    ok == 0 && data == 0
}

/// Local date and time for a Unix time in milliseconds: (year, month, day, weekday 0 = Sunday, hour, minute).
pub fn local_time(unix_ms: i64) -> (u16, u16, u16, u16, u16, u16) {
    // FILETIME counts 100 ns since 1601-01-01.
    let ft = (unix_ms + 11_644_473_600_000) as u64 * 10_000;
    let ft = FILETIME { dwLowDateTime: ft as u32, dwHighDateTime: (ft >> 32) as u32 };
    unsafe {
        let mut utc: SYSTEMTIME = std::mem::zeroed();
        let mut local: SYSTEMTIME = std::mem::zeroed();
        FileTimeToSystemTime(&ft, &mut utc);
        SystemTimeToTzSpecificLocalTime(std::ptr::null(), &utc, &mut local);
        (local.wYear, local.wMonth, local.wDay, local.wDayOfWeek, local.wHour, local.wMinute)
    }
}

const SHOW_EVENT: &str = "Local\\HamsaBeta-show-window\0";

/// True if this is the first Hamsa Beta; otherwise asks the running one to show its window.
pub fn claim_single_instance() -> bool {
    let name: Vec<u16> = "Local\\HamsaBeta-single-instance\0".encode_utf16().collect();
    unsafe {
        let m = CreateMutexW(std::ptr::null(), 1, name.as_ptr());
        if !m.is_null() && GetLastError() == ERROR_ALREADY_EXISTS {
            let ev: Vec<u16> = SHOW_EVENT.encode_utf16().collect();
            let e = OpenEventW(EVENT_MODIFY_STATE, 0, ev.as_ptr());
            if !e.is_null() {
                SetEvent(e);
                CloseHandle(e);
            }
            return false;
        }
        // The mutex handle stays open for the life of the process (it's never closed).
        let _ = m;
    }
    true
}

/// Call `on_show` whenever a second launch asks this instance to show itself.
pub fn listen_for_second_launch(on_show: impl Fn() + Send + 'static) {
    let ev: Vec<u16> = SHOW_EVENT.encode_utf16().collect();
    let e = unsafe { CreateEventW(std::ptr::null(), 0, 0, ev.as_ptr()) } as usize;
    if e == 0 {
        return;
    }
    std::thread::spawn(move || loop {
        unsafe { WaitForSingleObject(e as _, INFINITE) };
        on_show();
    });
}

/// Windows "Animation effects" (Settings > Accessibility > Visual effects). True when animations may play.
pub fn animations_enabled() -> bool {
    let mut on: i32 = 1;
    let ok = unsafe { SystemParametersInfoW(SPI_GETCLIENTAREAANIMATION, 0, &mut on as *mut i32 as *mut c_void, 0) };
    ok == 0 || on != 0
}
