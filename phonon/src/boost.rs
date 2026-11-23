//! Contextual biasing: nudge greedy decoding toward words and phrases from the user's dictionary.
//!
//! Each term is split into SentencePiece pieces (longest match against the model's vocabulary) and stored in a
//! prefix tree. While decoding, a token that starts a term gets a small bonus and a token that continues a term
//! already under way gets a larger one, so "Phonon" wins over "Fonon" when the audio is close, without forcing
//! the term where it wasn't said.

use std::collections::HashMap;

/// Bonus (in logits) for the first piece of a term. Tuned on test clips: strong enough to fix rare words,
/// and unrelated speech came out unchanged even at 4.5 / 6.
const START_BONUS: f32 = 3.5;
/// Bonus for each following piece once a term is under way.
const CONTINUE_BONUS: f32 = 5.0;
/// Longest piece in the vocabulary we try to match, in characters.
const MAX_PIECE_CHARS: usize = 24;

/// PHONON_BOOST="start,continue" overrides the bonuses (for tuning).
fn bonuses() -> (f32, f32) {
    static B: std::sync::OnceLock<(f32, f32)> = std::sync::OnceLock::new();
    *B.get_or_init(|| {
        std::env::var("PHONON_BOOST")
            .ok()
            .and_then(|v| {
                let (a, b) = v.split_once(',')?;
                Some((a.trim().parse().ok()?, b.trim().parse().ok()?))
            })
            .unwrap_or((START_BONUS, CONTINUE_BONUS))
    })
}

#[derive(Default)]
struct Node {
    children: HashMap<u32, usize>,
}

/// A prefix tree of term token sequences.
pub struct Boost {
    nodes: Vec<Node>,
}

impl Boost {
    pub(crate) fn build(terms: &[String], piece_ids: &HashMap<&str, u32>) -> Self {
        let mut boost = Self { nodes: vec![Node::default()] };
        let mut seen = std::collections::HashSet::new();
        for term in terms {
            let term = term.split_whitespace().collect::<Vec<_>>().join(" ");
            if term.is_empty() {
                continue;
            }
            // As written, plus capitalised (sentence start) and lower-case variants.
            let mut cap = term.clone();
            if let Some(first) = cap.chars().next() {
                cap = first.to_uppercase().collect::<String>() + &cap[first.len_utf8()..];
            }
            for variant in [term.clone(), cap, term.to_lowercase()] {
                if !seen.insert(variant.clone()) {
                    continue;
                }
                if let Some(ids) = tokenize(&variant, piece_ids) {
                    boost.insert(&ids);
                }
            }
        }
        boost
    }

    fn insert(&mut self, ids: &[u32]) {
        let mut node = 0;
        for &id in ids {
            node = match self.nodes[node].children.get(&id) {
                Some(&n) => n,
                None => {
                    self.nodes.push(Node::default());
                    let n = self.nodes.len() - 1;
                    self.nodes[node].children.insert(id, n);
                    n
                }
            };
        }
    }

    pub fn is_empty(&self) -> bool {
        self.nodes[0].children.is_empty()
    }

    /// Add this step's bonuses to `logits` given the terms under way (`active` tree nodes).
    pub(crate) fn apply(&self, active: &[usize], logits: &mut [f32]) {
        let (start, cont) = bonuses();
        for (&id, _) in &self.nodes[0].children {
            logits[id as usize] += start;
        }
        for &n in active {
            for (&id, _) in &self.nodes[n].children {
                logits[id as usize] += cont;
            }
        }
    }

    /// The terms still under way after emitting `token`.
    pub(crate) fn advance(&self, active: &[usize], token: u32) -> Vec<usize> {
        let mut next: Vec<usize> = active.iter().filter_map(|&n| self.nodes[n].children.get(&token).copied()).collect();
        if let Some(&n) = self.nodes[0].children.get(&token) {
            next.push(n);
        }
        next.retain(|&n| !self.nodes[n].children.is_empty());
        next.sort_unstable();
        next.dedup();
        next
    }
}

/// "New York" → pieces for "▁New▁York", longest match first. None if some character isn't in the vocabulary.
fn tokenize(term: &str, piece_ids: &HashMap<&str, u32>) -> Option<Vec<u32>> {
    let text: String = term.split(' ').map(|w| format!("\u{2581}{w}")).collect();
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let mut ids = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let mut found = None;
        let max = (chars.len() - i).min(MAX_PIECE_CHARS);
        for len in (1..=max).rev() {
            let start = chars[i].0;
            let end = chars.get(i + len).map_or(text.len(), |c| c.0);
            if let Some(&id) = piece_ids.get(&text[start..end]) {
                found = Some((id, len));
                break;
            }
        }
        let (id, len) = found?;
        ids.push(id);
        i += len;
    }
    Some(ids)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vocab() -> HashMap<&'static str, u32> {
        [("\u{2581}Ph", 1), ("on", 2), ("\u{2581}ph", 3), ("\u{2581}", 4), ("P", 5), ("h", 6), ("o", 7), ("n", 8), ("\u{2581}New", 9), ("\u{2581}York", 10)]
            .into_iter()
            .collect()
    }

    #[test]
    fn tokenizes_longest_match_and_phrases() {
        let v = vocab();
        assert_eq!(tokenize("Phonon", &v), Some(vec![1, 2, 2]));
        assert_eq!(tokenize("New York", &v), Some(vec![9, 10]));
        assert_eq!(tokenize("Ωmega", &v), None);
    }

    #[test]
    fn continuing_a_term_gets_the_big_bonus() {
        let v = vocab();
        let b = Boost::build(&["Phonon".into()], &v);
        let mut logits = vec![0.0f32; 12];
        b.apply(&[], &mut logits);
        assert_eq!(logits[1], START_BONUS);
        assert_eq!(logits[2], 0.0);
        let active = b.advance(&[], 1);
        let mut logits = vec![0.0f32; 12];
        b.apply(&active, &mut logits);
        assert_eq!(logits[2], CONTINUE_BONUS);
        // The full term ends the match.
        let active = b.advance(&b.advance(&active, 2), 2);
        assert!(active.is_empty());
    }
}
