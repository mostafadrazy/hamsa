//! "Start Hamsa when Windows starts": a value in the per-user Run key. The registry is the source of truth;
//! nothing is written until the user switches it on.

use std::path::Path;

use windows_sys::Win32::Foundation::ERROR_SUCCESS;
use windows_sys::Win32::System::Registry::{
    RegCloseKey, RegCreateKeyExW, RegDeleteKeyValueW, RegGetValueW, RegSetValueExW, HKEY, HKEY_CURRENT_USER, KEY_SET_VALUE,
    REG_OPTION_NON_VOLATILE, REG_SZ, RRF_RT_REG_SZ,
};

const RUN_KEY: &str = r"Software\Microsoft\Windows\CurrentVersion\Run";
const VALUE_NAME: &str = "HamsaBeta";

fn wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(std::iter::once(0)).collect()
}

/// The command Windows runs at sign-in: the quoted exe path, then `--background` (tray and bar only, no window).
pub fn run_value(exe: &Path) -> String {
    format!("\"{}\" --background", exe.display())
}

fn read_value(name: &str) -> Option<String> {
    let (key, name) = (wide(RUN_KEY), wide(name));
    let mut size: u32 = 0;
    let flags = RRF_RT_REG_SZ;
    let ok = unsafe { RegGetValueW(HKEY_CURRENT_USER, key.as_ptr(), name.as_ptr(), flags, std::ptr::null_mut(), std::ptr::null_mut(), &mut size) };
    if ok != ERROR_SUCCESS || size == 0 {
        return None;
    }
    let mut buf = vec![0u16; (size as usize).div_ceil(2) + 1];
    let mut size = (buf.len() * 2) as u32;
    let ok = unsafe {
        RegGetValueW(HKEY_CURRENT_USER, key.as_ptr(), name.as_ptr(), flags, std::ptr::null_mut(), buf.as_mut_ptr() as *mut _, &mut size)
    };
    if ok != ERROR_SUCCESS {
        return None;
    }
    let len = buf.iter().position(|&c| c == 0).unwrap_or(buf.len());
    Some(String::from_utf16_lossy(&buf[..len]))
}

fn write_value(name: &str, value: &str) -> bool {
    let (key, name, data) = (wide(RUN_KEY), wide(name), wide(value));
    unsafe {
        let mut h: HKEY = std::ptr::null_mut();
        let rc = RegCreateKeyExW(
            HKEY_CURRENT_USER,
            key.as_ptr(),
            0,
            std::ptr::null(),
            REG_OPTION_NON_VOLATILE,
            KEY_SET_VALUE,
            std::ptr::null(),
            &mut h,
            std::ptr::null_mut(),
        );
        if rc != ERROR_SUCCESS {
            return false;
        }
        let rc = RegSetValueExW(h, name.as_ptr(), 0, REG_SZ, data.as_ptr() as *const u8, (data.len() * 2) as u32);
        RegCloseKey(h);
        rc == ERROR_SUCCESS
    }
}

fn delete_value(name: &str) -> bool {
    let (key, name) = (wide(RUN_KEY), wide(name));
    // A value that is not there counts as deleted (ERROR_FILE_NOT_FOUND).
    let rc = unsafe { RegDeleteKeyValueW(HKEY_CURRENT_USER, key.as_ptr(), name.as_ptr()) };
    rc == ERROR_SUCCESS || rc == 2
}

fn current_value() -> Option<String> {
    std::env::current_exe().ok().map(|p| run_value(&p))
}

/// Whether Hamsa Beta is set to start with Windows (read from the registry).
pub fn is_enabled() -> bool {
    read_value(VALUE_NAME).is_some()
}

/// Switch it on (write the value) or off (delete it). Returns whether the registry now matches `on`.
pub fn set_enabled(on: bool) -> bool {
    if on {
        current_value().is_some_and(|v| write_value(VALUE_NAME, &v))
    } else {
        delete_value(VALUE_NAME)
    }
}

/// When it is on but points at another exe path (the app moved), rewrite it. Never creates the value.
pub fn refresh_path() {
    if let (Some(have), Some(want)) = (read_value(VALUE_NAME), current_value()) {
        if have != want {
            write_value(VALUE_NAME, &want);
        }
    }
}

/// True when the process was started by the Run key (or with --background): no main window.
pub fn background_launch() -> bool {
    std::env::args().skip(1).any(|a| a == "--background")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn run_value_quotes_the_path_and_adds_background() {
        let p = PathBuf::from(r"C:\Users\Some One\AppData\Local\Programs\Hamsa Beta\hamsa-beta.exe");
        assert_eq!(
            run_value(&p),
            r#""C:\Users\Some One\AppData\Local\Programs\Hamsa Beta\hamsa-beta.exe" --background"#
        );
        assert_eq!(run_value(Path::new(r"C:\a\b.exe")), r#""C:\a\b.exe" --background"#);
    }

    #[test]
    fn registry_write_read_delete_on_a_scratch_name() {
        let name = "HamsaBetaTest";
        let value = run_value(Path::new(r"C:\Program Files\Hamsa Test\hamsa-beta.exe"));
        assert!(write_value(name, &value));
        assert_eq!(read_value(name).as_deref(), Some(value.as_str()));
        assert!(delete_value(name));
        assert_eq!(read_value(name), None);
        assert!(delete_value(name), "deleting a missing value is fine");
    }
}
