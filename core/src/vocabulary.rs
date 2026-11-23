//! Dictionary and snippets, applied to every transcript.
//!
//! Both are lists of rules: when the `spoken` phrase is heard, `text` is written instead.
//! - Dictionary: names and terms. A plain word ("Phonon") fixes its spelling and capitalisation; a pair
//!   ("jason" → "JSON") replaces what Hamsa hears. Dictionary terms also steer the speech model (see phonon's boost).
//! - Snippets: say a cue ("my address") anywhere in a dictation and the saved text is typed in its place.
//!
//! Stored in preferences.json as `vocabulary: { dictionary: [...], snippets: [...] }`.

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rule {
    pub spoken: String,
    pub text: String,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Vocabulary {
    pub dictionary: Vec<Rule>,
    pub snippets: Vec<Rule>,
}

impl Rule {
    /// A plain word: written the way it is said (capitals aside), as opposed to a replacement.
    pub fn is_word(&self) -> bool {
        self.spoken.trim().to_lowercase() == self.text.trim().to_lowercase()
    }

    /// Whether `query` appears, ignoring case, in what is said or in what is written. An empty query matches everything.
    pub fn matches(&self, query: &str) -> bool {
        let q = query.trim().to_lowercase();
        q.is_empty() || self.spoken.to_lowercase().contains(&q) || self.text.to_lowercase().contains(&q)
    }
}

/// Comparison form of one word: lower case, without apostrophes, hyphens and dots inside it.
fn norm(word: &str) -> String {
    word.chars().filter(|c| !matches!(c, '\'' | '’' | '-' | '.')).flat_map(char::to_lowercase).collect()
}

/// A phrase as comparison words ("Hello,  World" → ["hello", "world"]).
fn phrase_words(phrase: &str) -> Vec<String> {
    phrase.split_whitespace().map(|w| norm(trim_punct(w).1)).filter(|w| !w.is_empty()).collect()
}

/// Split a whitespace chunk into (leading punctuation, core, trailing punctuation). "C++" and "C#" keep their signs.
fn trim_punct(chunk: &str) -> (&str, &str, &str) {
    let start = chunk.find(char::is_alphanumeric).unwrap_or(chunk.len());
    let mut end = chunk.rfind(char::is_alphanumeric).map_or(start, |i| i + chunk[i..].chars().next().map_or(1, char::len_utf8));
    end += chunk[end..].chars().take_while(|c| matches!(c, '+' | '#')).map(char::len_utf8).sum::<usize>();
    (&chunk[..start], &chunk[start..end], &chunk[end..])
}

/// Check a list before saving it: both fields filled in, no cue used twice.
pub fn validate(rules: &[Rule], what: &str) -> Result<(), String> {
    let mut seen = std::collections::HashSet::new();
    for r in rules {
        if phrase_words(&r.spoken).is_empty() || r.text.trim().is_empty() {
            return Err(format!("Each {what} needs both what you say and what to write."));
        }
        if !seen.insert(phrase_words(&r.spoken)) {
            return Err(format!("“{}” is already in the list.", r.spoken.trim()));
        }
    }
    Ok(())
}

struct Chunk<'a> {
    lead: &'a str,
    core: &'a str,
    trail: &'a str,
    /// Byte offset of `core` in the transcript.
    at: usize,
}

impl Vocabulary {
    /// Terms the speech model should favour.
    pub fn terms(&self) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        for r in &self.dictionary {
            for t in [r.text.trim(), r.spoken.trim()] {
                // Multi-line or very long texts aren't words to listen for.
                if !t.is_empty() && t.len() <= 48 && !t.contains('\n') && !out.iter().any(|o| o == t) {
                    out.push(t.to_string());
                }
            }
        }
        out
    }

    /// Apply snippets and the dictionary to a transcript.
    pub fn apply(&self, transcript: &str) -> String {
        self.apply_detailed(transcript).0
    }

    /// Like `apply`, also telling whether the whole utterance was a single snippet cue (then the text is
    /// meant to be pasted exactly as saved).
    pub fn apply_detailed(&self, transcript: &str) -> (String, bool) {
        if self.dictionary.is_empty() && self.snippets.is_empty() {
            return (transcript.to_string(), false);
        }
        let mut chunks = Vec::new();
        let mut pos = 0;
        for piece in transcript.split_whitespace() {
            let at = pos + transcript[pos..].find(piece).unwrap_or(0);
            pos = at + piece.len();
            let (lead, core, trail) = trim_punct(piece);
            chunks.push(Chunk { lead, core, trail, at: at + lead.len() });
        }
        let words: Vec<String> = chunks.iter().map(|c| norm(c.core)).collect();

        // Snippets first, then dictionary; longer phrases win.
        let mut rules: Vec<(Vec<String>, &str, bool)> = self
            .snippets
            .iter()
            .map(|r| (phrase_words(&r.spoken), r.text.as_str(), true))
            .chain(self.dictionary.iter().map(|r| (phrase_words(&r.spoken), r.text.trim(), false)))
            .filter(|(w, t, _)| !w.is_empty() && !t.trim().is_empty())
            .collect();
        rules.sort_by(|a, b| b.2.cmp(&a.2).then(b.0.len().cmp(&a.0.len())));

        let mut out = String::with_capacity(transcript.len());
        let mut copied = 0; // bytes of `transcript` already handled
        let mut only_snippet = None;
        let mut i = 0;
        while i < chunks.len() {
            let hit = rules.iter().find(|(w, _, _)| {
                let n = w.len();
                i + n <= chunks.len()
                    && (0..n).all(|k| {
                        words[i + k] == w[k]
                            && (k + 1 == n || chunks[i + k].trail.is_empty())
                            && (k == 0 || chunks[i + k].lead.is_empty())
                    })
            });
            match hit {
                Some((w, text, snippet)) => {
                    let first = &chunks[i];
                    let last = &chunks[i + w.len() - 1];
                    out.push_str(&transcript[copied..first.at]);
                    out.push_str(text);
                    copied = last.at + last.core.len();
                    if *snippet && i == 0 && w.len() == chunks.len() {
                        only_snippet = Some(text.to_string());
                    }
                    i += w.len();
                }
                None => i += 1,
            }
        }
        out.push_str(&transcript[copied..]);

        // "My address." on its own becomes just the address, without the full stop the model added.
        if let Some(text) = only_snippet {
            let rest = out.strip_prefix(&text).unwrap_or("");
            if rest.chars().all(|c| matches!(c, '.' | '!' | '?' | ',' | ' ')) {
                return (text, true);
            }
        }
        (out, false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rule(s: &str, t: &str) -> Rule {
        Rule { spoken: s.into(), text: t.into() }
    }

    fn vocab() -> Vocabulary {
        Vocabulary {
            dictionary: vec![rule("Phonon", "Phonon"), rule("jason", "JSON"), rule("new york", "New York"), rule("kubectl", "kubectl")],
            snippets: vec![rule("my address", "1 Example Street\nSpringfield"), rule("sign off", "Best,\nSam")],
        }
    }

    #[test]
    fn fixes_spelling_and_replacements() {
        let v = vocab();
        assert_eq!(v.apply("Send the phonon build."), "Send the Phonon build.");
        assert_eq!(v.apply("Parse the Jason, then go."), "Parse the JSON, then go.");
        assert_eq!(v.apply("I love new york!"), "I love New York!");
        assert_eq!(v.apply("New, York"), "New, York");
        assert_eq!(v.apply("Phonons are fine"), "Phonons are fine");
    }

    #[test]
    fn snippets_expand_in_place_or_alone() {
        let v = vocab();
        assert_eq!(v.apply("My address."), "1 Example Street\nSpringfield");
        assert_eq!(v.apply("Ship it to my address, please."), "Ship it to 1 Example Street\nSpringfield, please.");
        assert_eq!(v.apply("Thanks. Sign off."), "Thanks. Best,\nSam.");
    }

    #[test]
    fn detailed_flags_a_lone_snippet() {
        let v = vocab();
        assert_eq!(v.apply_detailed("My address."), ("1 Example Street
Springfield".to_string(), true));
        assert!(!v.apply_detailed("Ship it to my address, please.").1);
        assert!(!v.apply_detailed("Hello there").1);
    }

    #[test]
    fn empty_vocabulary_changes_nothing() {
        assert_eq!(Vocabulary::default().apply("Hello there."), "Hello there.");
    }

    #[test]
    fn validation_catches_blanks_and_duplicates() {
        assert!(validate(&[rule("a", "b")], "word").is_ok());
        assert!(validate(&[rule(" ", "b")], "word").is_err());
        assert!(validate(&[rule("My cue", "x"), rule("my  cue", "y")], "snippet").is_err());
    }

    #[test]
    fn terms_skip_long_texts() {
        let t = vocab().terms();
        assert!(t.contains(&"JSON".to_string()) && t.contains(&"jason".to_string()));
        assert!(!t.iter().any(|x| x.contains('\n')));
    }

    #[test]
    fn words_replacements_and_search() {
        assert!(rule("Phonon", "phonon").is_word());
        assert!(!rule("jason", "JSON").is_word());
        assert!(rule("jason", "JSON").matches("  JSO "));
        assert!(rule("jason", "JSON").matches("ASO"));
        assert!(!rule("jason", "JSON").matches("xyz"));
        assert!(rule("a", "b").matches(""));
    }

    #[test]
    fn reads_the_shared_preferences_shape() {
        let v: Vocabulary = serde_json::from_str(r#"{"dictionary":[{"spoken":"a","text":"A"}],"snippets":[]}"#).unwrap();
        assert_eq!(v.dictionary[0].text, "A");
    }
}
