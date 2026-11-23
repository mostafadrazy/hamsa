//! Rule-based text cleanup applied to a transcript before it is pasted.
//!
//! Independent switches (the Style page in Hamsa Beta): drop filler words, list formatting, smart quotes and dashes,
//! capital letters at the start of sentences, and a full stop at the end.

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct Formatting {
    /// Capital letter at the start of the text and of each sentence.
    pub capitalize: bool,
    /// Add a full stop when the text ends on a word.
    pub end_punctuation: bool,
    /// Remove "um", "uh", "erm".
    pub remove_fillers: bool,
    /// Curly quotes and apostrophes, em dashes.
    pub smart_punctuation: bool,
    /// "first ... second ..." and "shopping list: a, b, c" become numbered or bulleted lines.
    pub lists: bool,
    /// The mark in front of bulleted items (numbered lists always use "1.").
    pub bullet: Bullet,
}

/// The mark in front of each item of a bulleted list.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Bullet {
    /// "• Item"
    #[default]
    Dot,
    /// "- Item"
    Dash,
}

impl Bullet {
    pub fn mark(self) -> &'static str {
        match self {
            Bullet::Dot => "•",
            Bullet::Dash => "-",
        }
    }
}

impl Default for Formatting {
    /// What the Style page starts with.
    fn default() -> Self {
        Self {
            capitalize: true,
            end_punctuation: true,
            remove_fillers: true,
            smart_punctuation: false,
            lists: true,
            bullet: Bullet::Dot,
        }
    }
}

impl Formatting {
    /// Changes nothing.
    pub const OFF: Formatting = Formatting {
        capitalize: false,
        end_punctuation: false,
        remove_fillers: false,
        smart_punctuation: false,
        lists: false,
        bullet: Bullet::Dot,
    };
}

/// Clean up `text` according to the switches.
pub fn apply(text: &str, f: &Formatting) -> String {
    apply_detailed(text, f).0
}

/// As `apply`; the flag says the text was turned into a list (several lines), which later steps must not flatten.
pub fn apply_detailed(text: &str, f: &Formatting) -> (String, bool) {
    let mut out = text.to_string();
    if f.remove_fillers {
        out = remove_fillers(&out);
    }
    let mut listed = false;
    if f.lists {
        if let Some(l) = crate::lists::format(&out, f.bullet, f.capitalize) {
            out = l;
            listed = true;
        }
    }
    if f.smart_punctuation {
        out = smart_punctuation(&out);
    }
    if f.capitalize {
        out = capitalize(&out);
    }
    // List items carry no full stop.
    if f.end_punctuation && !listed {
        out = end_punctuation(&out);
    }
    (out, listed)
}

const FILLERS: [&str; 4] = ["um", "uh", "erm", "uhm"];

/// A standalone filler word, possibly with punctuation around it ("Um,", "uh."). Not "umbrella" or "um.com".
fn is_filler(token: &str) -> bool {
    let core = token.trim_matches(|c: char| !c.is_alphanumeric());
    let inner = token.trim_end_matches(['.', '!', '?', ',', ';', ':', '…']);
    FILLERS.iter().any(|f| core.eq_ignore_ascii_case(f)) && !inner.contains(['.', '@', '/'])
}

fn remove_fillers(text: &str) -> String {
    text.split('\n').map(remove_fillers_in_line).collect::<Vec<_>>().join("\n")
}

fn remove_fillers_in_line(line: &str) -> String {
    // Lines without a filler are left exactly as they are (spacing included).
    if !line.split_whitespace().any(is_filler) {
        return line.to_string();
    }
    let indent = &line[..line.len() - line.trim_start().len()];
    let mut out: Vec<String> = Vec::new();
    let mut cap_next = false;
    for token in line.split_whitespace() {
        if !is_filler(token) {
            let mut t = token.to_string();
            if cap_next {
                t = upper_first(&t);
                cap_next = false;
            }
            out.push(t);
            continue;
        }
        let starts_sentence = match out.last() {
            None => true,
            Some(prev) => prev.ends_with(['.', '!', '?']),
        };
        let was_capital =
            token.trim_start_matches(|c: char| !c.is_alphanumeric()).chars().next().is_some_and(char::is_uppercase);
        if starts_sentence && was_capital {
            cap_next = true;
        }
        // A closing mark carried by the filler ("we go, um.") moves onto the previous word.
        let mut trail: Vec<char> = token.chars().rev().take_while(|c| !c.is_alphanumeric()).collect();
        trail.reverse();
        if let Some(end) = trail.into_iter().find(|c| matches!(c, '.' | '!' | '?')) {
            if let Some(prev) = out.last_mut() {
                if prev.ends_with([',', ';']) {
                    prev.pop();
                }
                if !prev.ends_with(['.', '!', '?', ':']) {
                    prev.push(end);
                }
            }
        }
    }
    format!("{indent}{}", out.join(" "))
}

/// Upper-case the first letter of a word (skipping an opening quote or bracket).
pub(crate) fn upper_first(s: &str) -> String {
    let mut done = false;
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        if !done && c.is_alphabetic() {
            done = true;
            out.extend(c.to_uppercase());
        } else {
            if c.is_alphanumeric() {
                done = true;
            }
            out.push(c);
        }
    }
    out
}

fn capitalize(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut need = true; // the next letter starts a sentence
    let mut ended = false; // just saw . ! ? and no space yet
    for c in text.chars() {
        if c == '\n' {
            need = true;
            ended = false;
        } else if c.is_whitespace() {
            if ended {
                need = true;
                ended = false;
            }
        } else if matches!(c, '.' | '!' | '?') {
            ended = true;
        } else if ended && matches!(c, '"' | '\'' | '”' | '’' | ')' | ']' | '}') {
            // a closing quote after the full stop is still the end of the sentence
        } else if need && matches!(c, '"' | '\'' | '“' | '‘' | '(' | '[' | '{' | '¿' | '¡') {
            // an opening quote before the first letter
        } else {
            // "hamsa.app", "a@b.com", "3.5": no space after the dot, so no new sentence
            ended = false;
            if need {
                need = false;
                if c.is_lowercase() {
                    out.extend(c.to_uppercase());
                    continue;
                }
            }
        }
        out.push(c);
    }
    out
}

fn end_punctuation(text: &str) -> String {
    let body = text.trim_end();
    let rest = &text[body.len()..];
    let words = body.split_whitespace().count();
    if words >= 2 && body.chars().next_back().is_some_and(char::is_alphanumeric) {
        format!("{body}.{rest}")
    } else {
        text.to_string()
    }
}

fn smart_punctuation(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        let prev = out.chars().next_back();
        match c {
            '\'' => out.push('’'),
            '"' => {
                let opening = prev.is_none_or(|p| p.is_whitespace() || matches!(p, '(' | '[' | '{' | '—' | '‘'));
                out.push(if opening { '“' } else { '”' });
            }
            ' ' => {
                // " - " and " -- " between words become " — ".
                let dashes = chars[i + 1..].iter().take_while(|&&d| d == '-').count();
                let next = chars.get(i + 2 + dashes).copied();
                let joins_numbers = prev.is_some_and(|b| b.is_ascii_digit()) && next.is_some_and(|n| n.is_ascii_digit());
                if (dashes == 1 || dashes == 2)
                    && chars.get(i + 1 + dashes) == Some(&' ')
                    && prev.is_some_and(|b| !b.is_whitespace())
                    && next.is_some_and(|n| !n.is_whitespace())
                    && !joins_numbers
                {
                    out.push_str(" — ");
                    i += 1 + dashes;
                } else {
                    out.push(' ');
                }
            }
            _ => out.push(c),
        }
        i += 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn only(f: impl Fn(&mut Formatting)) -> Formatting {
        let mut x = Formatting::OFF;
        f(&mut x);
        x
    }
    fn fill(t: &str) -> String {
        apply(t, &only(|f| f.remove_fillers = true))
    }
    fn cap(t: &str) -> String {
        apply(t, &only(|f| f.capitalize = true))
    }
    fn end(t: &str) -> String {
        apply(t, &only(|f| f.end_punctuation = true))
    }
    fn smart(t: &str) -> String {
        apply(t, &only(|f| f.smart_punctuation = true))
    }

    #[test]
    fn fillers_are_removed_with_their_commas() {
        assert_eq!(fill("So, um, I think so"), "So, I think so");
        assert_eq!(fill("Um I think so"), "I think so");
        assert_eq!(fill("Um, I think so"), "I think so");
        assert_eq!(fill("I think, uh, we go"), "I think, we go");
        assert_eq!(fill("We go erm to town"), "We go to town");
        assert_eq!(fill("UM okay"), "Okay");
        assert_eq!(fill("Hello. Um, we go."), "Hello. We go.");
        assert_eq!(fill("We go, um."), "We go.");
        assert_eq!(fill("um um okay"), "okay");
    }

    #[test]
    fn fillers_leave_other_words_and_lines_alone() {
        assert_eq!(fill("Bring an umbrella to Uhura"), "Bring an umbrella to Uhura");
        assert_eq!(fill("Visit um.com now"), "Visit um.com now");
        assert_eq!(fill("a  b\num c"), "a  b\nc");
        assert_eq!(fill(""), "");
    }

    #[test]
    fn capitalizes_sentences() {
        assert_eq!(cap("hello there. how are you? fine! ok"), "Hello there. How are you? Fine! Ok");
        assert_eq!(cap("line one\nline two"), "Line one\nLine two");
        assert_eq!(cap("he said \"stop.\" then left"), "He said \"stop.\" Then left");
        assert_eq!(cap("\"hello\" she said"), "\"Hello\" she said");
        assert_eq!(cap(""), "");
    }

    #[test]
    fn capitalize_leaves_urls_emails_and_numbers() {
        assert_eq!(cap("go to hamsa.app now"), "Go to hamsa.app now");
        assert_eq!(cap("mail a@b.com today"), "Mail a@b.com today");
        assert_eq!(cap("it costs 3.5 euros"), "It costs 3.5 euros");
        assert_eq!(cap("Already Fine. Yes."), "Already Fine. Yes.");
    }

    #[test]
    fn adds_an_end_stop_only_when_needed() {
        assert_eq!(end("I think so"), "I think so.");
        assert_eq!(end("I think so  "), "I think so.  ");
        assert_eq!(end("Really?"), "Really?");
        assert_eq!(end("Done!"), "Done!");
        assert_eq!(end("Wait:"), "Wait:");
        assert_eq!(end("He said \"go.\""), "He said \"go.\"");
        assert_eq!(end("(see above)"), "(see above)");
        assert_eq!(end("Yes"), "Yes");
        assert_eq!(end(""), "");
        assert_eq!(end("Call 555"), "Call 555.");
    }

    #[test]
    fn smart_quotes_and_dashes() {
        assert_eq!(smart("don't go"), "don’t go");
        assert_eq!(smart("she said \"hi there\" today"), "she said “hi there” today");
        assert_eq!(smart("(\"hi\")"), "(“hi”)");
        assert_eq!(smart("wait -- what"), "wait — what");
        assert_eq!(smart("wait - what"), "wait — what");
        assert_eq!(smart("well-known 5 - 3 e-mail"), "well-known 5 - 3 e-mail");
        assert_eq!(smart("a\n - item"), "a\n - item");
        assert_eq!(smart(""), "");
    }

    fn lst(t: &str) -> String {
        apply(t, &Formatting::default())
    }
    fn plain(t: &str) -> String {
        apply(t, &Formatting { lists: false, ..Formatting::default() })
    }

    #[test]
    fn numbered_lists_in_english_french_and_darija() {
        assert_eq!(lst("first eggs second milk third bread"), "1. Eggs
2. Milk
3. Bread");
        assert_eq!(lst("number one call Sam number two send the report"), "1. Call Sam
2. Send the report");
        assert_eq!(lst("Here is the plan, first call Sam, second send the report."), "Here is the plan
1. Call Sam
2. Send the report");
        assert_eq!(lst("First, eggs and second, milk."), "1. Eggs
2. Milk");
        assert_eq!(lst("one, call Sam, two, send the report"), "1. Call Sam
2. Send the report");
        assert_eq!(lst("1. eggs 2. milk"), "1. Eggs
2. Milk");
        assert_eq!(lst("premièrement acheter du pain deuxièmement appeler Paul"), "1. Acheter du pain
2. Appeler Paul");
        assert_eq!(lst("الأول نشري الخبز الثاني نعيط لمحمد"), "1. نشري الخبز
2. نعيط لمحمد");
        assert_eq!(lst("واحد الخبز، جوج الحليب، تلاتة الزيت"), "1. الخبز
2. الحليب
3. الزيت");
    }

    #[test]
    fn bullet_lists_in_english_french_and_darija() {
        assert_eq!(lst("shopping list eggs, milk, bread and cheese"), "Shopping list:
• Eggs
• Milk
• Bread
• Cheese");
        assert_eq!(lst("I need to buy: apples, rice, olive oil and tea"), "I need to buy:
• Apples
• Rice
• Olive oil
• Tea");
        assert_eq!(lst("liste de courses: pain, lait, fromage et œufs"), "Liste de courses:
• Pain
• Lait
• Fromage
• Œufs");
        assert_eq!(lst("لائحة ديال الماكلة: الخبز، الحليب، الزيت و أتاي"), "لائحة ديال الماكلة:
• الخبز
• الحليب
• الزيت
• أتاي");
        assert_eq!(lst("shopping list colon eggs, milk, bread"), "Shopping list:
• Eggs
• Milk
• Bread");
        assert_eq!(lst("we need milk, eggs, and bread"), "We need:
• Milk
• Eggs
• Bread");
        assert_eq!(lst("Okay. Buy milk, eggs and bread."), "Okay. Buy:
• Milk
• Eggs
• Bread");
    }

    #[test]
    fn dash_bullets_and_switches() {
        let dash = Formatting { bullet: Bullet::Dash, ..Formatting::default() };
        assert_eq!(apply("shopping list eggs, milk, bread and cheese", &dash), "Shopping list:
- Eggs
- Milk
- Bread
- Cheese");
        let no_caps = Formatting { capitalize: false, ..Formatting::default() };
        assert_eq!(apply("shopping list eggs, milk, bread", &no_caps), "shopping list:
• eggs
• milk
• bread");
        let text = "first eggs second milk third bread";
        assert_eq!(apply(text, &Formatting { lists: false, ..Formatting::OFF }), text);
        assert_eq!(plain("shopping list eggs, milk, bread"), "Shopping list eggs, milk, bread.");
    }

    #[test]
    fn lists_work_with_fillers_capitals_and_end_stops() {
        assert_eq!(lst("um shopping list uh eggs, milk and bread"), "Shopping list:
• Eggs
• Milk
• Bread");
        assert_eq!(lst("uh first eggs um second milk."), "1. Eggs
2. Milk");
        let (out, listed) = apply_detailed("shopping list eggs, milk, bread", &Formatting::default());
        assert!(listed && !out.ends_with('.'));
        assert!(!apply_detailed("hello there", &Formatting::default()).1);
    }

    #[test]
    fn ordinary_sentences_are_not_lists() {
        for t in [
            "I have two cats and one dog",
            "I have one cat and two dogs",
            "the one I like",
            "I went to the shop, bought milk, and came home",
            "first of all, thanks for coming",
            "Yesterday, after work, I called Sam",
            "shopping list eggs and milk",
            "shopping list: eggs, milk",
            "the list is long, boring, and slow",
            "I made a list, then went home, and slept",
            "one, two, three",
            "I need to buy milk, and then I will go home, and sleep",
            "He came first and second place went to Bob",
        ] {
            assert_eq!(lst(t), plain(t), "{t}");
        }
        // Existing line breaks are left alone.
        let t = "shopping list
eggs, milk, bread";
        assert_eq!(lst(t), plain(t));
        let t = "first eggs
second milk";
        assert_eq!(lst(t), plain(t));
    }

    #[test]
    fn arabic_and_french_text_comes_through_unharmed() {
        // Darija: Arabic script with French words in Latin letters.
        let text = "التلفون ديالي طاح وتهرس ليه l'écran";
        assert_eq!(cap(text), text, "capitalize does nothing to Arabic");
        assert_eq!(fill(text), text);
        assert_eq!(end(text), format!("{text}."), "a full stop after the last letter");
        assert_eq!(end("واش نتا مزيان؟"), "واش نتا مزيان؟");
        assert_eq!(smart("ديالي l'écran"), "ديالي l’écran");
        // A French sentence start is still capitalised; the Arabic around it is left alone.
        assert_eq!(cap("سلام. bonjour tout le monde"), "سلام. Bonjour tout le monde");
        // No direction marks or other characters are added or lost.
        let all = Formatting { smart_punctuation: true, ..Formatting::default() };
        let out = apply("الtéléphone ديالي طاح وتهرس l'écran", &all);
        assert_eq!(out, "الtéléphone ديالي طاح وتهرس l’écran.");
        assert!(out.chars().all(|c| !matches!(c, '\u{200e}' | '\u{200f}' | '\u{202a}'..='\u{202e}' | '\u{2066}'..='\u{2069}')));
        assert_eq!(apply(&out, &all), out, "idempotent");
    }

    #[test]
    fn everything_together_and_idempotent() {
        let f = Formatting { smart_punctuation: true, ..Formatting::default() };
        let once = apply("so, um, it's fine -- really", &f);
        assert_eq!(once, "So, it’s fine — really.");
        assert_eq!(apply(&once, &f), once);
        assert_eq!(apply("Already fine.", &Formatting::default()), "Already fine.");
        assert_eq!(apply("anything at all", &Formatting::OFF), "anything at all");
    }
}
