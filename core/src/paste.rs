//! Put text into whatever app has focus: clipboard + Ctrl+V, then restore the previous clipboard text.

use std::time::Duration;

pub fn paste_text(text: &str) -> Result<(), String> {
    let mut clipboard = arboard::Clipboard::new().map_err(|e| format!("clipboard unavailable: {e}"))?;
    let previous = clipboard.get_text().ok();
    clipboard.set_text(text.to_string()).map_err(|e| format!("couldn't set the clipboard: {e}"))?;
    std::thread::sleep(Duration::from_millis(30));
    send_ctrl_v();
    // Give the target app time to read the clipboard before restoring it.
    std::thread::sleep(Duration::from_millis(250));
    if let Some(previous) = previous {
        let _ = clipboard.set_text(previous);
    }
    Ok(())
}

#[cfg(windows)]
fn send_ctrl_v() {
    use windows_sys::Win32::UI::Input::KeyboardAndMouse::{
        SendInput, INPUT, INPUT_0, INPUT_KEYBOARD, KEYBDINPUT, KEYEVENTF_KEYUP, VK_CONTROL,
    };
    const VK_V: u16 = 0x56;
    let key = |vk: u16, flags: u32| INPUT {
        r#type: INPUT_KEYBOARD,
        Anonymous: INPUT_0 { ki: KEYBDINPUT { wVk: vk, wScan: 0, dwFlags: flags, time: 0, dwExtraInfo: 0 } },
    };
    let inputs = [key(VK_CONTROL, 0), key(VK_V, 0), key(VK_V, KEYEVENTF_KEYUP), key(VK_CONTROL, KEYEVENTF_KEYUP)];
    unsafe {
        SendInput(inputs.len() as u32, inputs.as_ptr(), std::mem::size_of::<INPUT>() as i32);
    }
}

#[cfg(not(windows))]
fn send_ctrl_v() {}
