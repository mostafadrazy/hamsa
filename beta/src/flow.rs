//! Wrapping layout for the dictionary's word chips (Slint has no flow layout), measured with the interface font.

use std::sync::OnceLock;

const FONT: &[u8] = include_bytes!("../assets/fonts/ReadexPro-Latin.ttf");

/// Chip: 16 px left padding, label, 4 px gap, 28 px remove button, 6 px right padding, 1 px border each side.
const CHIP_EXTRA: f32 = 16.0 + 4.0 + 28.0 + 6.0 + 2.0;
pub const CHIP_HEIGHT: f32 = 38.0;
pub const GAP: f32 = 8.0;

fn face() -> Option<&'static ttf_parser::Face<'static>> {
    static FACE: OnceLock<Option<ttf_parser::Face<'static>>> = OnceLock::new();
    FACE.get_or_init(|| ttf_parser::Face::parse(FONT, 0).ok()).as_ref()
}

/// Width of `text` at `size` px. A little generous (the medium weight is wider than the font's default), never short.
pub fn text_width(text: &str, size: f32) -> f32 {
    let Some(face) = face() else { return text.chars().count() as f32 * size * 0.6 };
    let units = face.units_per_em() as f32;
    let em: f32 = text
        .chars()
        .map(|c| face.glyph_index(c).and_then(|g| face.glyph_hor_advance(g)).map_or(0.6, |a| a as f32 / units))
        .sum();
    em * size * 1.05
}

pub fn chip_width(text: &str) -> f32 {
    text_width(text, 14.0) + CHIP_EXTRA
}

/// Top-left corner of each box when laid out left to right, wrapping at `avail`; and the total height.
pub fn flow(widths: &[f32], avail: f32, height: f32, gap: f32) -> (Vec<(f32, f32)>, f32) {
    let mut out = Vec::with_capacity(widths.len());
    let (mut x, mut y) = (0.0_f32, 0.0_f32);
    for &w in widths {
        if x > 0.0 && x + w > avail {
            x = 0.0;
            y += height + gap;
        }
        out.push((x, y));
        x += w + gap;
    }
    (out, if widths.is_empty() { 0.0 } else { y + height })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wraps_when_a_row_is_full() {
        let (pos, h) = flow(&[100.0, 100.0, 100.0], 220.0, 38.0, 8.0);
        assert_eq!(pos, vec![(0.0, 0.0), (108.0, 0.0), (0.0, 46.0)]);
        assert_eq!(h, 84.0);
        assert_eq!(flow(&[], 220.0, 38.0, 8.0).1, 0.0);
        // A box wider than the row still gets its own row.
        assert_eq!(flow(&[300.0, 50.0], 220.0, 38.0, 8.0).0, vec![(0.0, 0.0), (0.0, 46.0)]);
    }

    #[test]
    fn longer_text_is_wider() {
        assert!(chip_width("Kubernetes") > chip_width("Go"));
        assert!(chip_width("Go") > CHIP_EXTRA);
    }
}
