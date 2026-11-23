//! Hold-to-talk shortcut via a Windows low-level keyboard hook. Any shortcut of up to 4 keys.
//!
//! Key ids are Windows virtual-key codes. Generic modifiers (Ctrl 0x11, Shift 0x10, Alt 0x12, Win 0x5B) match either
//! side; side-specific codes (Right Ctrl 0xA3, Right Alt 0xA5, …) match that key only.
//!
//! - All shortcut keys down          → `HotkeyDown` (start recording)
//! - Any shortcut key released       → `HotkeyUp`   (stop and process)
//! - Any other key while held        → `Cancel`     (it was a different shortcut, e.g. Ctrl+Win+D)
//!
//! While hands-free recording is on (double-tap, see `handsfree`), Esc is swallowed and cancels it.
//!
//! Non-modifier keys of the shortcut (F8 in Ctrl+F8, or F8 alone) are swallowed so they don't reach the app. When the
//! shortcut includes Win or Alt, an unassigned "mask" key (0xE8) is injected so releasing it doesn't open the Start
//! menu or an app's menu bar.
//!
//! Capture mode records the next shortcut the user presses (all keys swallowed) and reports it on release.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::SeqCst};
use std::sync::mpsc::Sender;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use windows_sys::Win32::System::LibraryLoader::GetModuleHandleW;
use windows_sys::Win32::UI::Input::KeyboardAndMouse::{
    SendInput, INPUT, INPUT_0, INPUT_KEYBOARD, KEYBDINPUT, KEYEVENTF_KEYUP,
};
use windows_sys::Win32::UI::WindowsAndMessaging::{
    CallNextHookEx, DispatchMessageW, GetMessageW, SetWindowsHookExW, TranslateMessage,
    KBDLLHOOKSTRUCT, LLKHF_INJECTED, MSG, WH_KEYBOARD_LL, WM_KEYDOWN, WM_KEYUP, WM_SYSKEYDOWN,
    WM_SYSKEYUP,
};

use crate::dictation::Control;

const CTRL: u16 = 0x11;
const SHIFT: u16 = 0x10;
const ALT: u16 = 0x12;
const WIN: u16 = 0x5B;
const ESC: u32 = 0x1B;
/// Unassigned virtual key used only to mask Win / Alt releases.
const VK_MASK: u16 = 0xE8;

static SENDER: OnceLock<Sender<Control>> = OnceLock::new();
static ON_CAPTURE: OnceLock<Box<dyn Fn(Vec<u16>) + Send + Sync>> = OnceLock::new();
static RAW_DOWN: [AtomicBool; 256] = [const { AtomicBool::new(false) }; 256];
static SWALLOWED: [AtomicBool; 256] = [const { AtomicBool::new(false) }; 256];
/// Up to four key ids packed 16 bits each; 0 = empty slot.
static COMBO: AtomicU64 = AtomicU64::new(CTRL as u64 | (WIN as u64) << 16);
static ACTIVE: AtomicBool = AtomicBool::new(false);
/// Hands-free recording is on: Esc cancels it (and is swallowed). Set by the dictation thread.
static LOCKED: AtomicBool = AtomicBool::new(false);
static CAPTURING: AtomicBool = AtomicBool::new(false);
static CAPTURED: Mutex<Vec<u16>> = Mutex::new(Vec::new());

pub fn set_locked(on: bool) {
    LOCKED.store(on, SeqCst);
}

fn send(c: Control) {
    if let Some(tx) = SENDER.get() {
        let _ = tx.send(c);
    }
}

fn combo() -> ([u16; 4], usize) {
    let packed = COMBO.load(SeqCst);
    let mut keys = [0u16; 4];
    let mut n = 0;
    for i in 0..4 {
        let k = (packed >> (16 * i)) as u16;
        if k != 0 {
            keys[n] = k;
            n += 1;
        }
    }
    (keys, n)
}

pub fn set_combo(keys: &[u16]) {
    let packed = keys.iter().take(4).enumerate().fold(0u64, |acc, (i, &k)| acc | (k as u64) << (16 * i));
    COMBO.store(packed, SeqCst);
    ACTIVE.store(false, SeqCst);
}

/// The raw keys a key id stands for.
fn raws(id: u16) -> &'static [u16] {
    match id {
        CTRL => &[0xA2, 0xA3, 0x11],
        SHIFT => &[0xA0, 0xA1, 0x10],
        ALT => &[0xA4, 0xA5, 0x12],
        WIN => &[0x5B, 0x5C],
        _ => &[],
    }
}

fn matches(id: u16, raw: u32) -> bool {
    let r = raws(id);
    if r.is_empty() {
        id as u32 == raw
    } else {
        r.iter().any(|&x| x as u32 == raw)
    }
}

fn id_down(id: u16) -> bool {
    let r = raws(id);
    if r.is_empty() {
        RAW_DOWN[id as usize & 0xFF].load(SeqCst)
    } else {
        r.iter().any(|&x| RAW_DOWN[x as usize].load(SeqCst))
    }
}

fn is_modifier(raw: u32) -> bool {
    matches!(raw, 0x10 | 0x11 | 0x12 | 0xA0..=0xA5 | 0x5B | 0x5C)
}

fn masks_menu(id: u16) -> bool {
    matches!(id, ALT | WIN | 0xA4 | 0xA5 | 0x5C)
}

unsafe fn send_mask_key() {
    let key = |flags: u32| INPUT {
        r#type: INPUT_KEYBOARD,
        Anonymous: INPUT_0 { ki: KEYBDINPUT { wVk: VK_MASK, wScan: 0, dwFlags: flags, time: 0, dwExtraInfo: 0 } },
    };
    let inputs = [key(0), key(KEYEVENTF_KEYUP)];
    SendInput(inputs.len() as u32, inputs.as_ptr(), std::mem::size_of::<INPUT>() as i32);
}

/// Returns true when the event should be swallowed.
unsafe fn handle(vk: u32, down: bool, up: bool) -> bool {
    if CAPTURING.load(SeqCst) {
        let mut captured = CAPTURED.lock().unwrap_or_else(|e| e.into_inner());
        if down {
            if vk == ESC && captured.is_empty() {
                CAPTURING.store(false, SeqCst);
                drop(captured);
                report_capture(Vec::new());
                return true;
            }
            RAW_DOWN[vk as usize].store(true, SeqCst);
            if !captured.contains(&(vk as u16)) && captured.len() < 4 {
                captured.push(vk as u16);
            }
        } else if up {
            RAW_DOWN[vk as usize].store(false, SeqCst);
            let all_up = captured.iter().all(|&k| !RAW_DOWN[k as usize].load(SeqCst));
            if all_up && !captured.is_empty() {
                CAPTURING.store(false, SeqCst);
                let keys = std::mem::take(&mut *captured);
                drop(captured);
                report_capture(keys);
            }
        }
        return true;
    }

    if down || up {
        RAW_DOWN[vk as usize].store(down, SeqCst);
    }
    let (keys, n) = combo();
    let keys = &keys[..n];
    if n == 0 {
        return false;
    }
    let in_combo = keys.iter().any(|&id| matches(id, vk));

    // Hands-free recording: Esc cancels it. Only then is Esc taken; otherwise it goes to the app untouched.
    if vk == ESC && !in_combo {
        if down && (LOCKED.load(SeqCst) || SWALLOWED[ESC as usize].load(SeqCst)) {
            // Key repeat keeps sending "down"; only the first one cancels.
            if !SWALLOWED[ESC as usize].swap(true, SeqCst) {
                send(Control::Esc);
            }
            return true;
        }
        if up && SWALLOWED[ESC as usize].swap(false, SeqCst) {
            return true;
        }
    }

    if down {
        if !in_combo {
            if ACTIVE.swap(false, SeqCst) {
                send(Control::Cancel);
            }
        } else if keys.iter().all(|&id| id_down(id)) {
            // Key repeat keeps sending "down"; only the first one starts.
            if !ACTIVE.swap(true, SeqCst) {
                if keys.iter().any(|&id| masks_menu(id)) {
                    send_mask_key();
                }
                send(Control::HotkeyDown(Instant::now()));
            }
            if !is_modifier(vk) {
                SWALLOWED[vk as usize].store(true, SeqCst);
                return true;
            }
        }
    } else if up {
        if in_combo && ACTIVE.swap(false, SeqCst) {
            send(Control::HotkeyUp(Instant::now()));
        }
        if SWALLOWED[vk as usize].swap(false, SeqCst) {
            return true;
        }
    }
    false
}

fn report_capture(keys: Vec<u16>) {
    if let Some(cb) = ON_CAPTURE.get() {
        // Off the hook thread: the hook must return quickly.
        std::thread::spawn(move || cb(keys));
    }
}

unsafe extern "system" fn keyboard_proc(code: i32, wparam: usize, lparam: isize) -> isize {
    if code >= 0 {
        let kb = &*(lparam as *const KBDLLHOOKSTRUCT);
        // Ignore keys we (or other tools) injected.
        if kb.flags & LLKHF_INJECTED == 0 {
            let msg = wparam as u32;
            let down = msg == WM_KEYDOWN || msg == WM_SYSKEYDOWN;
            let up = msg == WM_KEYUP || msg == WM_SYSKEYUP;
            if handle(kb.vkCode & 0xFF, down, up) {
                return 1;
            }
        }
    }
    CallNextHookEx(std::ptr::null_mut(), code, wparam, lparam)
}

/// True while any key of the shortcut is still held (pasting with Ctrl+V must wait until they're up).
pub fn shortcut_held() -> bool {
    let (keys, n) = combo();
    keys[..n].iter().any(|&id| id_down(id))
}

/// Record the next shortcut the user presses; `on_capture` receives the raw keys (empty when Esc cancels).
pub fn begin_capture() {
    if let Ok(mut c) = CAPTURED.lock() {
        c.clear();
    }
    if ACTIVE.swap(false, SeqCst) || LOCKED.load(SeqCst) {
        send(Control::Cancel);
    }
    CAPTURING.store(true, SeqCst);
}

pub fn cancel_capture() {
    CAPTURING.store(false, SeqCst);
}

/// Turn captured raw keys into a shortcut: a single key keeps its side (Right Alt, Right Ctrl); in a combination
/// left/right modifiers become generic and modifiers come first.
pub fn normalize(raw: &[u16]) -> Vec<u16> {
    if raw.len() == 1 {
        return vec![if raw[0] == 0x5C { WIN } else { raw[0] }];
    }
    let generic = |k: u16| match k {
        0xA2 | 0xA3 => CTRL,
        0xA0 | 0xA1 => SHIFT,
        0xA4 | 0xA5 => ALT,
        0x5C => WIN,
        k => k,
    };
    let mut out: Vec<u16> = Vec::new();
    for &k in raw {
        let g = generic(k);
        if !out.contains(&g) {
            out.push(g);
        }
    }
    let rank = |k: &u16| match *k {
        CTRL => 0,
        SHIFT => 1,
        ALT => 2,
        WIN => 3,
        _ => 4,
    };
    out.sort_by_key(rank);
    out
}

fn is_typing_key(k: u16) -> bool {
    matches!(k, 0x08 | 0x09 | 0x0D | 0x20 | 0x30..=0x39 | 0x41..=0x5A | 0x60..=0x6F | 0xBA..=0xC0 | 0xDB..=0xDF)
}

/// Why a shortcut can't be used, if it can't.
pub fn validate(keys: &[u16]) -> Result<(), String> {
    match keys {
        [] => Err("No keys were pressed.".into()),
        [k] => {
            let ok = matches!(*k, 0x70..=0x87 | 0xA3 | 0xA5 | 0x14 | 0x91 | 0x13 | 0x2D);
            if ok {
                Ok(())
            } else {
                Err("A single key must be one you don't type with, like F8, Right Alt or Right Ctrl. Or hold a modifier with it.".into())
            }
        }
        _ => {
            let has_main_mod = keys.iter().any(|&k| matches!(k, CTRL | ALT | WIN));
            if !has_main_mod && keys.iter().any(|&k| is_typing_key(k)) {
                Err("Add Ctrl, Alt or Win so the shortcut doesn't type a character.".into())
            } else {
                Ok(())
            }
        }
    }
}

pub fn label(id: u16) -> String {
    let s = match id {
        CTRL => "Ctrl",
        0xA2 => "Left Ctrl",
        0xA3 => "Right Ctrl",
        SHIFT => "Shift",
        0xA0 => "Left Shift",
        0xA1 => "Right Shift",
        ALT => "Alt",
        0xA4 => "Left Alt",
        0xA5 => "Right Alt",
        WIN | 0x5C => "Win",
        0x08 => "Backspace",
        0x09 => "Tab",
        0x0D => "Enter",
        0x13 => "Pause",
        0x14 => "Caps Lock",
        0x20 => "Space",
        0x21 => "Page Up",
        0x22 => "Page Down",
        0x23 => "End",
        0x24 => "Home",
        0x25 => "Left",
        0x26 => "Up",
        0x27 => "Right",
        0x28 => "Down",
        0x2D => "Insert",
        0x2E => "Delete",
        0x91 => "Scroll Lock",
        0xBA => ";",
        0xBB => "=",
        0xBC => ",",
        0xBD => "-",
        0xBE => ".",
        0xBF => "/",
        0xC0 => "`",
        0xDB => "[",
        0xDC => "\\",
        0xDD => "]",
        0xDE => "'",
        0x70..=0x87 => return format!("F{}", id - 0x6F),
        0x30..=0x39 | 0x41..=0x5A => return char::from(id as u8).to_string(),
        0x60..=0x69 => return format!("Num {}", id - 0x60),
        _ => return format!("Key {id:#04X}"),
    };
    s.to_string()
}

/// Install the hook on its own thread (low-level hooks need a message loop).
pub fn start(tx: Sender<Control>, shortcut: &[u16], on_capture: impl Fn(Vec<u16>) + Send + Sync + 'static) {
    set_combo(shortcut);
    if SENDER.set(tx).is_err() {
        return;
    }
    let _ = ON_CAPTURE.set(Box::new(on_capture));
    std::thread::Builder::new()
        .name("hamsa-hotkey".into())
        .spawn(|| unsafe {
            let module = GetModuleHandleW(std::ptr::null());
            let hook = SetWindowsHookExW(WH_KEYBOARD_LL, Some(keyboard_proc), module, 0);
            if hook.is_null() {
                eprintln!("hamsa: could not install the keyboard hook");
                return;
            }
            let mut msg: MSG = std::mem::zeroed();
            while GetMessageW(&mut msg, std::ptr::null_mut(), 0, 0) > 0 {
                TranslateMessage(&msg);
                DispatchMessageW(&msg);
            }
        })
        .expect("spawn hotkey thread");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_keys_keep_their_side() {
        assert_eq!(normalize(&[0xA5]), vec![0xA5]); // Right Alt
        assert_eq!(normalize(&[0x77]), vec![0x77]); // F8
        assert_eq!(normalize(&[0x5C]), vec![WIN]); // Right Win → Win
    }

    #[test]
    fn combinations_use_generic_modifiers_first() {
        // Pressed Right Win, then Left Ctrl → Ctrl + Win
        assert_eq!(normalize(&[0x5C, 0xA2]), vec![CTRL, WIN]);
        // Space, Right Shift, Left Alt → Shift + Alt + Space
        assert_eq!(normalize(&[0x20, 0xA1, 0xA4]), vec![SHIFT, ALT, 0x20]);
        // Both Ctrl keys collapse into one
        assert_eq!(normalize(&[0xA2, 0xA3, 0x77]), vec![CTRL, 0x77]);
    }

    #[test]
    fn rejects_shortcuts_that_would_type() {
        assert!(validate(&[]).is_err());
        assert!(validate(&[0x41]).is_err()); // A
        assert!(validate(&[0x20]).is_err()); // Space
        assert!(validate(&[0xA2]).is_err()); // Left Ctrl alone
        assert!(validate(&[SHIFT, 0x41]).is_err()); // Shift + A types "A"
        assert!(validate(&[0x77]).is_ok()); // F8
        assert!(validate(&[0xA5]).is_ok()); // Right Alt
        assert!(validate(&[CTRL, WIN]).is_ok());
        assert!(validate(&[CTRL, SHIFT]).is_ok());
        assert!(validate(&[CTRL, 0x20]).is_ok()); // Ctrl + Space
        assert!(validate(&[ALT, SHIFT, 0x44]).is_ok());
    }

    #[test]
    fn labels() {
        assert_eq!(label(CTRL), "Ctrl");
        assert_eq!(label(0xA5), "Right Alt");
        assert_eq!(label(0x77), "F8");
        assert_eq!(label(0x44), "D");
        assert_eq!(label(0x20), "Space");
    }

    #[test]
    fn combo_round_trip_and_matching() {
        set_combo(&[CTRL, SHIFT, 0x20]);
        let (keys, n) = combo();
        assert_eq!(&keys[..n], &[CTRL, SHIFT, 0x20]);
        assert!(matches(CTRL, 0xA3) && matches(CTRL, 0xA2) && !matches(CTRL, 0xA0));
        assert!(matches(0xA5, 0xA5) && !matches(0xA5, 0xA4));
        set_combo(&[CTRL, WIN]);
    }
}
