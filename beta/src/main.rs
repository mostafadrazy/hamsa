//! Hamsa Beta: the native (Slint) version of Hamsa. Same features and data as the Tauri build, no web view.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod app;
mod autostart;
mod bar;
mod flow;
mod store;
mod tray;
mod win;

slint::include_modules!();

use slint::winit_030::winit::platform::windows::WindowAttributesExtWindows;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Developer hook: run smart cleanup on one text, print what happened, exit (no windows, no other instance needed).
    if let Ok(text) = std::env::var("HAMSA_BETA_CLEANUP_TEST") {
        let size = std::env::var("HAMSA_BETA_CLEANUP_SIZE")
            .unwrap_or_else(|_| store::load_settings(&hamsa_core::data_dir()).cleanup_size);
        hamsa_core::llm::cleanup::dev_test(&text, hamsa_core::CleanupSize::from_name(&size));
        return Ok(());
    }

    // Developer hook: transcribe a WAV with a Darija model (HAMSA_BETA_ASR_MODEL=0.6b|1.7b), print, exit.
    if let Some(wav) = std::env::var_os("HAMSA_BETA_ASR_TEST") {
        let model = match std::env::var("HAMSA_BETA_ASR_MODEL").as_deref() {
            Ok("1.7b") => hamsa_core::AsrModel::Accurate,
            _ => hamsa_core::AsrModel::Fast,
        };
        hamsa_core::llm::asr::dev_test(std::path::Path::new(&wav), model);
        return Ok(());
    }

    // A second launch just brings the running one forward.
    if !win::claim_single_instance() {
        return Ok(());
    }

    // Software rendering keeps memory low (no GPU driver in the UI); the bar window is created inactive and
    // without a taskbar button so it never steals focus from the app you're dictating into.
    slint::BackendSelector::new()
        .backend_name("winit".into())
        .renderer_name("software".into())
        .with_winit_window_attributes_hook(|attrs| {
            if attrs.title == "Hamsa bar" {
                attrs.with_active(false).with_skip_taskbar(true)
            } else {
                attrs
            }
        })
        .select()?;

    let app = app::App::start()?;
    // Started at sign-in (--background): tray icon and bar only, never the setup. A later launch still brings the window
    // forward. A first run (no setup done and no speech model) opens the setup instead of the main window.
    // HAMSA_BETA_SETUP=1..5 opens the setup at that step whatever the saved state (for checking screens).
    let background = autostart::background_launch();
    let dev_step = std::env::var("HAMSA_BETA_SETUP").ok().and_then(|v| v.trim().parse::<i32>().ok()).filter(|s| (1..=5).contains(s));
    if let Some(step) = dev_step {
        app.show_setup(step);
    } else if !app.startup_setup(background) && !background {
        app.show_main();
    }
    app.apply_dev_options();

    let _tray = tray::create(|action| {
        app::APP.with(|a| {
            if let Some(app) = a.borrow().as_ref() {
                match action {
                    tray::TrayAction::Open => app.show_main(),
                    tray::TrayAction::TestMicrophone => app.test_microphone(),
                    tray::TrayAction::Quit => {
                        let _ = slint::quit_event_loop();
                    }
                }
            }
        });
    });

    win::listen_for_second_launch(|| {
        let _ = slint::invoke_from_event_loop(|| {
            app::APP.with(|a| {
                if let Some(app) = a.borrow().as_ref() {
                    app.show_main();
                }
            });
        });
    });

    // Keep running when the main window is closed (it hides to the tray).
    slint::run_event_loop_until_quit()?;
    app::APP.with(|a| {
        if let Some(app) = a.borrow().as_ref() {
            app.save_all();
        }
    });
    app::APP.with(|a| a.borrow_mut().take());
    Ok(())
}
