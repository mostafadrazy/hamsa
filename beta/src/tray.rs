//! System tray icon: Hamsa Beta keeps running here when its window is closed.

use tray_icon::menu::{Menu, MenuEvent, MenuItem, PredefinedMenuItem};
use tray_icon::{MouseButton, MouseButtonState, TrayIcon, TrayIconBuilder, TrayIconEvent};

pub enum TrayAction {
    Open,
    TestMicrophone,
    Quit,
}

/// Create the tray icon. `on_action` runs on the UI thread.
pub fn create(on_action: impl Fn(TrayAction) + Send + Sync + Clone + 'static) -> Option<TrayIcon> {
    let open = MenuItem::new("Open Hamsa Beta", true, None);
    let test = MenuItem::new("Test microphone", true, None);
    let quit = MenuItem::new("Quit Hamsa Beta", true, None);
    let menu = Menu::new();
    menu.append_items(&[&open, &test, &PredefinedMenuItem::separator(), &quit]).ok()?;
    let (open_id, test_id, quit_id) = (open.id().clone(), test.id().clone(), quit.id().clone());

    let act = on_action.clone();
    MenuEvent::set_event_handler(Some(move |e: MenuEvent| {
        let action = if e.id == open_id {
            TrayAction::Open
        } else if e.id == test_id {
            TrayAction::TestMicrophone
        } else if e.id == quit_id {
            TrayAction::Quit
        } else {
            return;
        };
        let act = act.clone();
        let _ = slint::invoke_from_event_loop(move || act(action));
    }));
    let act = on_action;
    TrayIconEvent::set_event_handler(Some(move |e: TrayIconEvent| {
        if let TrayIconEvent::Click { button: MouseButton::Left, button_state: MouseButtonState::Up, .. } = e {
            let act = act.clone();
            let _ = slint::invoke_from_event_loop(move || act(TrayAction::Open));
        }
    }));

    // The icon is the one built into the .exe (resource 1, see build.rs).
    let icon = tray_icon::Icon::from_resource(1, Some((32, 32))).ok();
    let mut builder = TrayIconBuilder::new().with_menu(Box::new(menu)).with_tooltip("Hamsa Beta");
    if let Some(icon) = icon {
        builder = builder.with_icon(icon);
    }
    builder.build().ok()
}
