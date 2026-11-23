//! Rule-based list formatting for dictated text (English, French, Darija in Arabic script).
//!
//! Two triggers, both conservative:
//! * counting ("first eggs second milk", "number one ... number two ...") gives a numbered list;
//! * a lead-in ("shopping list", "I need to buy:") followed by three or more short items gives a bulleted list.
//!
//! Text that already has line breaks is never touched. The result is plain text with `\n`.

use crate::format::{upper_first, Bullet};

/// Item separators: comma, semicolon and their Arabic forms.
const SEPARATORS: [char; 4] = [',', ';', '،', '؛'];
/// Marks that end a clause (a counting word after one of them starts a new item).
const BOUNDARY: [char; 9] = [',', ';', ':', '.', '!', '?', '،', '؛', '؟'];
const SENTENCE_END: [char; 4] = ['.', '!', '?', '؟'];

const ORDINALS: [&[&str]; 10] = [
    &["first", "firstly", "premièrement", "premier", "première", "الأول", "الاول", "الأولى"],
    &["second", "secondly", "deuxièmement", "deuxième", "seconde", "الثاني", "الثانية"],
    &["third", "thirdly", "troisièmement", "troisième", "الثالث", "الثالثة"],
    &["fourth", "fourthly", "quatrièmement", "quatrième", "الرابع", "الرابعة"],
    &["fifth", "fifthly", "cinquièmement", "cinquième", "الخامس", "الخامسة"],
    &["sixth", "sixthly", "sixièmement", "sixième", "السادس", "السادسة"],
    &["seventh", "seventhly", "septièmement", "septième", "السابع", "السابعة"],
    &["eighth", "eighthly", "huitièmement", "huitième", "الثامن", "الثامنة"],
    &["ninth", "ninthly", "neuvièmement", "neuvième", "التاسع", "التاسعة"],
    &["tenth", "tenthly", "dixièmement", "dixième", "العاشر", "العاشرة"],
];

const CARDINALS: [&[&str]; 10] = [
    &["one", "un", "une", "واحد"],
    &["two", "deux", "جوج"],
    &["three", "trois", "تلاتة", "ثلاثة", "تلاثة"],
    &["four", "quatre", "ربعة", "أربعة", "اربعة"],
    &["five", "cinq", "خمسة"],
    &["six", "ستة"],
    &["seven", "sept", "سبعة"],
    &["eight", "huit", "تمنية", "ثمانية"],
    &["nine", "neuf", "تسعود", "تسعة"],
    &["ten", "dix", "عشرة"],
];

/// A first word that shows the "item" is really a clause ("then I will go home"), not a thing in a list.
const NOT_AN_ITEM: [&str; 24] = [
    "i", "we", "you", "he", "she", "they", "it", "then", "so", "but", "because", "is", "are", "was", "were", "has",
    "have", "je", "nous", "il", "elle", "est", "mais", "puis",
];

/// Words that name a list.
const LIST_NOUNS: [&str; 8] = ["list", "liste", "todo", "لائحة", "لائحه", "ليستة", "ليست", "لايحة"];

/// Verb lead-ins, longest first.
const VERB_LEADS: [&[&str]; 7] = [
    &["i", "need", "to", "buy"],
    &["we", "need", "to", "buy"],
    &["remember", "to", "get"],
    &["we", "need"],
    &["pick", "up"],
    &["buy"],
    &["bring"],
];

/// Turn `text` into a list, or `None` when it isn't one.
pub fn format(text: &str, bullet: Bullet, capitalize: bool) -> Option<String> {
    if text.contains('\n') {
        return None;
    }
    let toks: Vec<&str> = text.split_whitespace().collect();
    if toks.len() < 4 {
        return None;
    }
    numbered(&toks, capitalize).or_else(|| bulleted(&toks, bullet, capitalize))
}

fn core(tok: &str) -> String {
    tok.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase()
}

fn ends_boundary(tok: &str) -> bool {
    tok.ends_with(BOUNDARY)
}

fn is_conj(tok: &str) -> bool {
    matches!(core(tok).as_str(), "and" | "et" | "و")
}

fn item_case(s: String, capitalize: bool) -> String {
    if capitalize {
        upper_first(&s)
    } else {
        s
    }
}

// ---- numbered lists --------------------------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Family {
    /// first, second, premièrement, الأول
    Ordinal,
    /// one, two, un, deux, واحد, a bare 1
    Cardinal,
    /// "number one", "numéro 2"
    Number,
    /// "1." or "2)"
    Digit,
}

fn lookup(table: &[&[&str]; 10], word: &str) -> Option<u8> {
    let found = |w: &str| table.iter().position(|names| names.contains(&w)).map(|i| i as u8 + 1);
    if let Some(v) = found(word) {
        return Some(v);
    }
    // Arabic "and" glued to the word: وجوج
    let rest = word.strip_prefix('و')?;
    if rest.chars().count() >= 3 {
        found(rest)
    } else {
        None
    }
}

/// A plain 1..=10 ("3" or "three") with the punctuation around it removed.
fn number_value(tok: &str) -> Option<u8> {
    let c = core(tok);
    if let Ok(n) = c.parse::<u8>() {
        return (1..=10).contains(&n).then_some(n);
    }
    lookup(&CARDINALS, &c)
}

/// Does a counting marker start at `toks[i]`? Gives its family, value and length in words.
fn marker_at(toks: &[&str], i: usize) -> Option<(Family, u8, usize)> {
    let tok = toks[i];
    let c = core(tok);
    let found = 'm: {
        let lead = tok.trim_start_matches(|ch: char| !ch.is_alphanumeric());
        let digits: String = lead.chars().take_while(char::is_ascii_digit).collect();
        if !digits.is_empty() {
            let n = digits.parse::<u32>().ok().filter(|n| (1..=10).contains(n))? as u8;
            let rest = &lead[digits.len()..];
            break 'm if rest == "." || rest == ")" {
                (Family::Digit, n, 1)
            } else if rest.is_empty() || rest == "," {
                (Family::Cardinal, n, 1)
            } else {
                return None;
            };
        }
        if matches!(c.as_str(), "number" | "numéro" | "رقم") {
            let n = number_value(toks.get(i + 1)?)?;
            break 'm (Family::Number, n, 2);
        }
        if let Some(n) = lookup(&ORDINALS, &c) {
            break 'm (Family::Ordinal, n, 1);
        }
        if let Some(n) = lookup(&CARDINALS, &c) {
            break 'm (Family::Cardinal, n, 1);
        }
        return None;
    };
    // "first of all", "one of them"
    if toks.get(i + found.2).is_some_and(|t| core(t) == "of") {
        return None;
    }
    Some(found)
}

/// May a list start at the counting word `toks[i]`? It must open a clause, so "the one I like" and
/// "I have two cats and one dog" are not lists.
fn can_start(toks: &[&str], i: usize, family: Family) -> bool {
    if i == 0 || ends_boundary(toks[i - 1]) {
        return true;
    }
    family != Family::Cardinal
        && matches!(
            core(toks[i - 1]).as_str(),
            "and" | "then" | "et" | "puis" | "ensuite" | "so" | "ok" | "okay" | "alors" | "donc" | "و" | "ثم"
        )
}

fn numbered(toks: &[&str], capitalize: bool) -> Option<String> {
    for i in 0..toks.len() {
        let Some((family, 1, len)) = marker_at(toks, i) else { continue };
        if !can_start(toks, i, family) {
            continue;
        }
        // The markers must follow on in counting order, all of the same kind.
        let mut chain = vec![(i, len)];
        let mut j = i + len;
        while j < toks.len() {
            match marker_at(toks, j) {
                Some((f, v, l)) if f == family && v as usize == chain.len() + 1 => {
                    chain.push((j, l));
                    j += l;
                }
                _ => j += 1,
            }
        }
        if chain.len() < 2 {
            continue;
        }
        // Bare "one, two" need a comma before each marker unless the count goes on to three or more.
        if family == Family::Cardinal && chain.len() < 3 && chain.iter().any(|&(k, _)| k > 0 && !ends_boundary(toks[k - 1]))
        {
            continue;
        }
        let mut items = Vec::new();
        for (n, &(start, l)) in chain.iter().enumerate() {
            let end = chain.get(n + 1).map_or(toks.len(), |&(next, _)| next);
            let item = clean_item(&toks[start + l..end]);
            if item.is_empty() {
                break;
            }
            items.push(item_case(item, capitalize));
        }
        if items.len() != chain.len() {
            continue;
        }
        let lead = toks[..i].join(" ");
        let lead = lead.trim_end_matches([',', ';', '،', '؛', ' ']);
        let mut out = String::new();
        if !lead.is_empty() {
            out.push_str(lead);
            out.push('\n');
        }
        let lines: Vec<String> = items.iter().enumerate().map(|(n, it)| format!("{}. {it}", n + 1)).collect();
        out.push_str(&lines.join("\n"));
        return Some(out);
    }
    None
}

/// An item's words without the separators and "and"/"et"/"و" that hang around it.
fn clean_item(toks: &[&str]) -> String {
    let mut s = toks.join(" ");
    loop {
        let t = s.trim_matches([',', ';', '،', '؛', '.', ' ']).to_string();
        let t = match t.rsplit_once(' ') {
            Some((rest, last)) if is_conj(last) => rest.to_string(),
            _ if is_conj(&t) => String::new(),
            _ => t,
        };
        if t == s {
            return t;
        }
        s = t;
    }
}

// ---- bulleted lists --------------------------------------------------------------------------------------------

/// Where the lead-in ends and the items begin, in the last sentence: (lead-in words shown, first item word).
fn find_lead(sent: &[&str]) -> Option<(usize, usize)> {
    // 1. A word ending in a colon.
    if let Some(c) = sent[..sent.len() - 1].iter().position(|t| t.ends_with(':') && t.trim_end_matches(':') != "") {
        return Some((c + 1, c + 1));
    }
    // 2. A list noun near the start: "shopping list", "liste de courses", "لائحة ديال الماكلة".
    for k in 0..sent.len().min(5) {
        let noun = core(sent[k]);
        if !LIST_NOUNS.contains(&noun.as_str()) {
            continue;
        }
        let mut e = k + 1;
        let word = |n: usize| sent.get(n).map(|t| core(t));
        if word(e).as_deref() == Some("de") && word(e + 1).as_deref() == Some("courses") {
            e += 2;
        } else if word(e).as_deref() == Some("ديال") {
            e += 2;
        }
        let shown = e;
        if word(e).as_deref() == Some("colon") {
            e += 1;
        }
        return (e < sent.len()).then_some((shown, e));
    }
    // 3. A short verb lead-in at the start of the sentence.
    for lead in VERB_LEADS {
        if sent.len() > lead.len() && lead.iter().enumerate().all(|(n, w)| core(sent[n]) == *w) {
            let mut e = lead.len();
            if core(sent[e]) == "colon" && e + 1 < sent.len() {
                e += 1;
            }
            return Some((lead.len(), e));
        }
    }
    None
}

fn bulleted(toks: &[&str], bullet: Bullet, capitalize: bool) -> Option<String> {
    // Only the last sentence can be the list; earlier ones stay as they are.
    let start = toks[..toks.len() - 1].iter().rposition(|t| t.ends_with(SENTENCE_END)).map_or(0, |p| p + 1);
    let sent = &toks[start..];
    let (shown, first_item) = find_lead(sent)?;

    let rest = sent[first_item..].join(" ");
    let rest = rest.trim_end_matches(['.', '!', '?', '؟', ' ']);
    let segments: Vec<&str> = rest.split(SEPARATORS).map(str::trim).collect();
    if segments.iter().any(|s| s.is_empty()) {
        return None;
    }
    let last = segments.len() - 1;
    let mut items: Vec<Vec<&str>> = Vec::new();
    for (k, seg) in segments.iter().enumerate() {
        let words: Vec<&str> = seg.split_whitespace().collect();
        if is_conj(words[0]) {
            // ", and bread"
            if words.len() == 1 {
                return None;
            }
            items.push(words[1..].to_vec());
        } else if k == last {
            // "bread and cheese": the final "and" ends the list
            match (1..words.len().saturating_sub(1)).rev().find(|&m| is_conj(words[m])) {
                Some(m) => {
                    items.push(words[..m].to_vec());
                    items.push(words[m + 1..].to_vec());
                }
                None => items.push(words),
            }
        } else {
            items.push(words);
        }
    }
    let short = items.iter().all(|w| {
        !w.is_empty()
            && w.len() <= 5
            && !w.iter().any(|t| t.ends_with(SENTENCE_END) || t.ends_with(':'))
            && !NOT_AN_ITEM.contains(&core(w[0]).as_str())
    });
    if items.len() < 3 || !short {
        return None;
    }

    let lead = sent[..shown].join(" ");
    let lead = lead.trim_end_matches([',', ';', '،', '؛', ':', ' ']);
    let mut out = String::new();
    if start > 0 {
        out.push_str(&toks[..start].join(" "));
        out.push(' ');
    }
    out.push_str(lead);
    out.push(':');
    for words in items {
        out.push('\n');
        out.push_str(bullet.mark());
        out.push(' ');
        out.push_str(&item_case(words.join(" "), capitalize));
    }
    Some(out)
}
