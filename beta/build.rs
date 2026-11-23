fn main() {
    slint_build::compile("ui/app.slint").expect("compile Slint UI");
    // App icon (resource 1): used for the .exe, the window and the tray.
    println!("cargo:rerun-if-changed=assets/app.rc");
    println!("cargo:rerun-if-changed=assets/app-icon.ico");
    embed_resource::compile("assets/app.rc", embed_resource::NONE).manifest_optional().unwrap();
}
