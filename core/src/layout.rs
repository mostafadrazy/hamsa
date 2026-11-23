//! Where the dictation bar sits and how it behaves. Saved in preferences.json as
//! `bar: { visible, preview, placement: { mode, x, y }, monitor }` (same shape as the other Hamsa builds).

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Default, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "lowercase")]
pub enum Placement {
    /// Bottom centre, above the taskbar.
    #[default]
    Bottom,
    /// Top centre.
    Top,
    /// Where the user dragged it: top-left of the small window, in physical screen px.
    Custom { x: i32, y: i32 },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct Layout {
    /// Show the bar while idle. When false it appears only during a dictation.
    pub visible: bool,
    /// Show the words while talking.
    pub preview: bool,
    pub placement: Placement,
    /// Monitor name for Bottom / Top.
    pub monitor: Option<String>,
}

impl Default for Layout {
    fn default() -> Self {
        Self { visible: true, preview: true, placement: Placement::Bottom, monitor: None }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn placement_matches_the_shared_preferences_shape() {
        let l: Layout = serde_json::from_str(
            r#"{"visible":false,"preview":true,"placement":{"mode":"custom","x":-1200,"y":40},"monitor":"\\\\.\\DISPLAY2"}"#,
        )
        .unwrap();
        assert_eq!(l.placement, Placement::Custom { x: -1200, y: 40 });
        assert!(!l.visible);
        let v = serde_json::to_value(Layout::default()).unwrap();
        assert_eq!(v["placement"]["mode"], "bottom");
    }
}
