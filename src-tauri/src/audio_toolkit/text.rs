use once_cell::sync::Lazy;
use regex::Regex;
use strsim::levenshtein;

// ---------------------------------------------------------------------------
// Fuzzy correction of transcripts against the active vocabulary.
//
// Designed for German medical dictation, where the cost of a wrong replacement
// is far higher than the cost of a missed one: turning "Mitralklappenstenose"
// into "Mitralklappeninsuffizienz", or swallowing the "kein" in "kein
// Perikarderguss", changes a finding. Every rule below therefore errs towards
// leaving the transcript alone. An uncertain replacement is simply skipped.
//
// Deliberately *not* used: Soundex. It is an English algorithm that encodes
// only the first letter plus three consonants, so long German compounds that
// share a prefix ("Mitralklappen…") collide and received a large score boost —
// that is exactly how the diagnosis swaps above happened. Matching is plain
// normalised edit distance on an orthographically folded key.
// ---------------------------------------------------------------------------

/// German inflectional endings. A transcript word that differs from a
/// dictionary term only by one of these is a correctly recognised inflected
/// form ("Stenosen", "linksventrikulären") and must not be flattened to the
/// dictionary's base form.
const INFLECTION_SUFFIXES: &[&str] = &["e", "n", "en", "s", "es", "er", "em", "ern", "ens", "nen"];

/// Derivational endings. A transcript word that is the term's stem plus one of
/// these is a *different word* — usually another part of speech
/// ("echokardiographisch" beside "Echokardiographie") — and must not be turned
/// into the term.
const DERIVATIONAL_ENDINGS: &[&str] = &[
    "isch", "ische", "ischen", "ischer", "isches", "ischem", "lich", "liche", "lichen", "licher",
    "liches", "lichem",
];

/// Negations. Never replaced, never absorbed into a multi-word span: losing one
/// inverts the finding.
const NEGATIONS: &[&str] = &[
    "kein", "keine", "keinen", "keinem", "keiner", "keines", "nicht", "nichts", "ohne", "nie",
    "niemals", "weder",
];

/// Units that follow a dose. Protected like negations so "Ramipril 5 mg" can
/// never have its dose folded into a drug name.
const DOSE_UNITS: &[&str] = &[
    "mg", "g", "kg", "µg", "ug", "mcg", "ml", "l", "ie", "iu", "mmol", "mval", "meq", "mmhg", "h",
];

/// A correction is skipped when some *other* term is at most this many edits
/// further from the transcript than the best one. Counted in whole edits, not
/// normalised distance, because the danger is concrete: "Nitedipin" is one
/// edit from Nifedipin and two from Nitrendipin, and guessing between two drugs
/// is worse than leaving the word as heard.
const AMBIGUITY_EDIT_MARGIN: usize = 1;

/// When a span and a term split into a different number of words — the
/// transcript joined or split a compound — words cannot be compared one by one,
/// so the whole span may differ from the term by at most this many edits.
/// Without this cap, a long term tolerates enough edits to swap one of its
/// words entirely ("…mit erhaltener…" for "…mit reduzierter…").
const MISALIGNED_EDIT_CAP: usize = 2;

/// Spans are never longer than this many transcript words, whatever the
/// dictionary contains.
const MAX_SPAN_WORDS: usize = 6;

/// Lower-case a word and fold German orthography to its standard ASCII
/// transliteration (ä→ae, ö→oe, ü→ue, ß→ss) plus a small table of Latin
/// accents, then keep letters and digits only.
///
/// ä→ae is the orthographic equivalence German itself uses, so "Mehrgefässerkrankung"
/// and "Mehrgefäßerkrankung" share a key. It does *not* collapse ä into a: "a"
/// and "ae" stay distinct, so different words never merge through folding.
/// Non-Latin scripts are left as they are and are then rejected as unsupported.
fn fold_key(word: &str) -> String {
    let mut out = String::with_capacity(word.len());
    for c in word.chars().flat_map(char::to_lowercase) {
        match c {
            'ä' => out.push_str("ae"),
            'ö' => out.push_str("oe"),
            'ü' => out.push_str("ue"),
            'ß' => out.push_str("ss"),
            'à' | 'á' | 'â' | 'ã' | 'å' => out.push('a'),
            'ç' => out.push('c'),
            'è' | 'é' | 'ê' | 'ë' => out.push('e'),
            'ì' | 'í' | 'î' | 'ï' => out.push('i'),
            'ñ' => out.push('n'),
            'ò' | 'ó' | 'ô' | 'õ' | 'ø' => out.push('o'),
            'ù' | 'ú' | 'û' => out.push('u'),
            c if c.is_alphanumeric() => out.push(c),
            _ => {}
        }
    }
    out
}

/// A folded key is usable when it is non-empty and entirely ASCII after
/// folding. CJK and other scripts fall outside: whitespace tokenisation and
/// character edit distance do not fit them.
fn is_supported_key(key: &str) -> bool {
    !key.is_empty() && key.chars().all(|c| c.is_ascii_alphanumeric())
}

fn digits_of(key: &str) -> String {
    key.chars().filter(char::is_ascii_digit).collect()
}

fn is_negation(word: &str) -> bool {
    NEGATIONS.contains(&fold_key(word).as_str())
}

fn is_dose_unit(word: &str) -> bool {
    DOSE_UNITS.contains(&fold_key(word).as_str())
}

fn has_digit(word: &str) -> bool {
    word.chars().any(|c| c.is_ascii_digit())
}

/// Which words a correction must never touch or absorb: negations everywhere,
/// and dose units where they follow a number ("5 mg", "10 IE"). A unit is only
/// protected in that position — on its own, "G" in a spelled-out "Chat G P T"
/// is just a letter.
fn protected_words(words: &[&str]) -> Vec<bool> {
    words
        .iter()
        .enumerate()
        .map(|(i, w)| is_negation(w) || (is_dose_unit(w) && i > 0 && has_digit(words[i - 1])))
        .collect()
}

/// Whether the transcript word `spoken` is `term` in another grammatical form,
/// compared on folded keys, rather than a misspelling of it.
///
/// Accepted:
/// - `spoken` adds or swaps an ending: "Stenosen" / "Stenose",
///   "linksventrikulären" / "linksventrikulärer", "Ergusses" / "Erguss";
/// - `spoken` is the "-e" form of an adjective the dictionary lists as
///   "-er/-en/-es/-em": "linksventrikuläre" / "linksventrikulärer".
///
/// Rejected — these are truncations or typos, and should be corrected:
/// - `spoken` merely lacks the term's last letter where no ending was dropped:
///   "Ejektionsfraktio" / "Ejektionsfraktion";
/// - the difference is a doubled letter: "Perikardergus" / "Perikarderguss".
///
/// A shared stem of at least five letters is required, so short words cannot
/// pass as each other's inflections.
fn is_inflection_variant(spoken: &str, term: &str) -> bool {
    let s: Vec<char> = fold_key(spoken).chars().collect();
    let t: Vec<char> = fold_key(term).chars().collect();
    if s == t {
        return true;
    }
    let common = s.iter().zip(&t).take_while(|(x, y)| x == y).count();
    let is_ending = |rest: &[char]| {
        let rest: String = rest.iter().collect();
        INFLECTION_SUFFIXES.contains(&rest.as_str())
    };
    // The shared stem may have swallowed the first letter of both endings
    // ("-en" / "-er" share their "e"), so step back up to three letters.
    for p in (5..=common).rev().take(4) {
        let (rest_s, rest_t) = (&s[p..], &t[p..]);
        let stem_last = s[p - 1];
        let doubled = |rest: &[char]| rest.first() == Some(&stem_last);
        if doubled(rest_s) || doubled(rest_t) {
            continue;
        }
        if rest_s.is_empty() {
            // Spoken form is shorter: only the adjective "-e" form qualifies.
            if stem_last == 'e' && matches!(rest_t, ['r'] | ['n'] | ['s'] | ['m']) {
                return true;
            }
            continue;
        }
        if is_ending(rest_s) && (rest_t.is_empty() || is_ending(rest_t)) {
            return true;
        }
    }
    false
}

/// A term split into the words a transcript would show, and the separator
/// that followed each word in the dictionary spelling ("-", "/", " " or "").
fn term_parts(term: &str) -> (Vec<&str>, Vec<&str>) {
    let mut words = Vec::new();
    let mut seps = Vec::new();
    let mut start: Option<usize> = None;
    let mut sep_start = 0;
    for (index, c) in term.char_indices() {
        let is_sep = c.is_whitespace() || c == '-' || c == '/';
        match (is_sep, start) {
            (false, None) => {
                if !words.is_empty() {
                    seps.push(&term[sep_start..index]);
                }
                start = Some(index);
            }
            (true, Some(word_start)) => {
                words.push(&term[word_start..index]);
                start = None;
                sep_start = index;
            }
            _ => {}
        }
    }
    if let Some(word_start) = start {
        words.push(&term[word_start..]);
    }
    seps.push("");
    (words, seps)
}

/// Split a dictionary term into the words a transcript would show.
fn term_words(term: &str) -> Vec<&str> {
    term_parts(term).0
}

/// Whether `spoken` is a word *derived* from the term's stem — the stem plus a
/// derivational ending such as "-isch" — rather than the term misspelt.
fn is_derived_form(spoken: &str, term: &str) -> bool {
    let (s, t) = (fold_key(spoken), fold_key(term));
    DERIVATIONAL_ENDINGS.iter().any(|ending| {
        s.strip_suffix(ending).is_some_and(|stem| {
            stem.chars().count() >= 5 && t.starts_with(stem) && !t.ends_with(ending)
        })
    })
}

/// The class of a folded key's first sound. ASR confusions keep a word's onset
/// far more often than they change it, and the classes only merge spellings of
/// one sound: c/k/z ("Cyanose", "Kyanose", "Zyanose"), f/v/w and "ph", t/d, p/b.
/// "Funktion" (f) and "Punktion" (p) therefore differ — one edit apart, but a
/// normal word and a procedure.
fn onset_class(key: &str) -> Option<char> {
    let first = if key.starts_with("ph") {
        'f'
    } else {
        key.chars().next()?
    };
    Some(match first {
        'c' | 'k' | 'z' | 'q' => 'k',
        'f' | 'v' | 'w' => 'f',
        't' | 'd' => 't',
        'p' | 'b' => 'p',
        other => other,
    })
}

/// How close one transcript word must be to its counterpart in the term: same
/// onset class, and within the threshold on its own.
fn word_close(spoken: &str, term_word: &str, threshold: f64) -> bool {
    let (a, b) = (fold_key(spoken), fold_key(term_word));
    if onset_class(&a) != onset_class(&b) {
        return false;
    }
    let max_len = a.chars().count().max(b.chars().count()).max(1) as f64;
    (levenshtein(&a, &b) as f64 / max_len) < threshold
}

fn letters(word: &str) -> impl Iterator<Item = char> + '_ {
    word.chars().filter(|c| c.is_alphabetic())
}

/// An abbreviation written in capitals ("DES", "EKG", "INR"). Many coincide
/// with ordinary German words ("des"), so one may only replace a transcript
/// token that is itself written in capitals.
fn is_capitals_abbreviation(word: &str) -> bool {
    letters(word).count() >= 2 && letters(word).all(char::is_uppercase)
}

fn has_lowercase(word: &str) -> bool {
    letters(word).any(char::is_lowercase)
}

struct TermKey {
    term_index: usize,
    key: String,
    digits: String,
}

/// Precomputed comparison data for the active vocabulary.
struct Matcher<'a> {
    terms: &'a [String],
    keys: Vec<TermKey>,
    /// Longest span worth trying: the most words any term splits into.
    max_span: usize,
    /// Longest candidate worth scoring: past this, the 25 % length rule would
    /// reject every term anyway. Derived from the vocabulary rather than fixed,
    /// so the longest real terms stay reachable.
    max_candidate_len: usize,
    threshold: f64,
}

/// Outcome of scoring one candidate span.
#[derive(Debug, Clone, Copy)]
enum Verdict {
    NoMatch,
    /// Two different terms are about equally close — do not guess.
    Ambiguous,
    Match {
        term_index: usize,
        score: f64,
    },
}

impl<'a> Matcher<'a> {
    fn new(terms: &'a [String], threshold: f64) -> Self {
        let mut keys = Vec::new();
        let mut max_span = 3;
        let mut max_key_len = 0;
        for (term_index, term) in terms.iter().enumerate() {
            let key = fold_key(term);
            if is_supported_key(&key) {
                max_key_len = max_key_len.max(key.chars().count());
                max_span = max_span.max(term_words(term).len());
                // A capitals abbreviation can arrive spelled out letter by
                // letter ("N S T E M I"), one transcript word per letter.
                if is_capitals_abbreviation(term) {
                    max_span = max_span.max(letters(term).count());
                }
                keys.push(TermKey {
                    term_index,
                    digits: digits_of(&key),
                    key: key.clone(),
                });
            }
            if term.contains('&') {
                let expanded = fold_key(&term.replace('&', " and "));
                if is_supported_key(&expanded) && expanded != key {
                    max_key_len = max_key_len.max(expanded.chars().count());
                    keys.push(TermKey {
                        term_index,
                        digits: digits_of(&expanded),
                        key: expanded,
                    });
                }
            }
        }
        let slack = ((max_key_len as f64) * 0.25).ceil().max(2.0) as usize;
        Matcher {
            terms,
            keys,
            max_span: max_span.min(MAX_SPAN_WORDS),
            max_candidate_len: max_key_len + slack,
            threshold,
        }
    }

    /// Score a folded candidate against every term.
    fn score(&self, candidate: &str) -> Verdict {
        let candidate_len = candidate.chars().count();
        if !is_supported_key(candidate) || candidate_len > self.max_candidate_len {
            return Verdict::NoMatch;
        }
        let candidate_digits = digits_of(candidate);

        // Every term close enough in length to be comparable: (term, edits,
        // normalised score). Kept whole so ambiguity can be judged against
        // terms that miss the acceptance threshold too.
        let mut comparable: Vec<(usize, usize, f64)> = Vec::new();
        for term_key in &self.keys {
            // Numbers are never altered: a candidate may only match a term with
            // the identical digit sequence ("HbA 1c" → "HbA1c" yes,
            // "Ramipril 5" → "Ramipril" no, "T2 Mapping" → "T1-Mapping" no).
            if term_key.digits != candidate_digits {
                continue;
            }
            let term_len = term_key.key.chars().count();
            let max_len = candidate_len.max(term_len) as f64;
            let max_allowed_diff = (max_len * 0.25).max(2.0);
            if candidate_len.abs_diff(term_len) as f64 > max_allowed_diff {
                continue;
            }
            let edits = levenshtein(candidate, &term_key.key);
            comparable.push((term_key.term_index, edits, edits as f64 / max_len));
        }

        let Some(&(term_index, edits, score)) = comparable
            .iter()
            .filter(|(_, _, score)| *score < self.threshold)
            .min_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal))
        else {
            return Verdict::NoMatch;
        };
        // An exact key match is never ambiguous.
        if edits == 0 {
            return Verdict::Match { term_index, score };
        }
        let rival = comparable.iter().any(|&(other, other_edits, _)| {
            other != term_index && other_edits <= edits + AMBIGUITY_EDIT_MARGIN
        });
        if rival {
            Verdict::Ambiguous
        } else {
            Verdict::Match { term_index, score }
        }
    }

    /// Fewest edits between `candidate` and any key of the given term.
    fn edits_to(&self, term_index: usize, candidate: &str) -> usize {
        self.keys
            .iter()
            .filter(|k| k.term_index == term_index)
            .map(|k| levenshtein(candidate, &k.key))
            .min()
            .unwrap_or(usize::MAX)
    }

    fn score_span(&self, keys: &[String]) -> Verdict {
        self.score(&keys.concat())
    }
}

/// The best span found at one position: (words, term index, score), and
/// whether any span there was rejected as ambiguous.
#[derive(Debug, Clone, Copy, Default)]
struct SpanChoice {
    best: Option<(usize, usize, f64)>,
    ambiguous: bool,
}

fn verdict_score(verdict: Verdict) -> Option<f64> {
    match verdict {
        Verdict::Match { score, .. } => Some(score),
        _ => None,
    }
}

/// One replacement the corrector made, or chose not to make.
#[derive(Debug, Clone, PartialEq)]
pub struct CorrectionEvent {
    /// The transcript text the decision was about.
    pub original: String,
    /// The dictionary term involved.
    pub term: String,
    /// Normalised edit distance (0 = identical key).
    pub score: f64,
    pub kind: CorrectionKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CorrectionKind {
    /// The span was replaced by the term's canonical spelling.
    Replaced,
    /// The span already was a correctly inflected form of the term; kept.
    KeptInflected,
}

/// Text after correction plus an audit trail of what changed.
#[derive(Debug, Clone, Default)]
pub struct CorrectionReport {
    pub text: String,
    pub events: Vec<CorrectionEvent>,
    /// Spans left alone because two terms matched about equally well.
    pub ambiguous_spans: Vec<String>,
}

/// Correct `text` against the active vocabulary and report every decision.
///
/// A span of up to [`MAX_SPAN_WORDS`] words is replaced by the canonical
/// spelling of a dictionary term only when all of these hold:
///
/// - its folded key is within `threshold` normalised edit distance of the term,
///   and no other term is within [`AMBIGUITY_EDIT_MARGIN`] further edits;
/// - it contains no negation or dose unit, and its digits match the term's;
/// - it does not cross a punctuation boundary;
/// - every word in it contributes — dropping its first or last word must not
///   match as well, otherwise that edge word (an article, a "kein") would be
///   swallowed;
/// - it is not already a correctly inflected form of the term.
pub fn correct_with_vocabulary(
    text: &str,
    custom_words: &[String],
    threshold: f64,
) -> CorrectionReport {
    if custom_words.is_empty() {
        return CorrectionReport {
            text: text.to_string(),
            ..Default::default()
        };
    }
    let matcher = Matcher::new(custom_words, threshold);
    let words: Vec<&str> = text.split_whitespace().collect();
    // Per-word facts, computed once rather than for every span that
    // contains the word.
    let keys: Vec<String> = words.iter().map(|w| fold_key(w)).collect();
    let protected = protected_words(&words);
    let trailing_punct: Vec<bool> = words
        .iter()
        .map(|w| !extract_punctuation(w).1.is_empty())
        .collect();

    // Best acceptable span starting at `i`, memoised because the look-ahead
    // below asks for position i+1 before the loop gets there.
    let mut memo: Vec<Option<SpanChoice>> = vec![None; words.len()];
    let mut best_at = |i: usize| -> SpanChoice {
        if let Some(choice) = memo[i] {
            return choice;
        }
        let mut choice = SpanChoice::default();
        for n in (1..=matcher.max_span).rev() {
            if i + n > words.len() || protected[i..i + n].iter().any(|&p| p) {
                continue;
            }
            // Never consume across a punctuation boundary: only the last word
            // of a span may carry trailing punctuation.
            if trailing_punct[i..i + n - 1].iter().any(|&p| p) {
                continue;
            }
            let span_keys = &keys[i..i + n];
            let (term_index, score) = match matcher.score_span(span_keys) {
                Verdict::Match { term_index, score } => (term_index, score),
                Verdict::Ambiguous => {
                    choice.ambiguous = true;
                    continue;
                }
                Verdict::NoMatch => continue,
            };
            // Every word must pull its weight: if the span minus its first or
            // last word matches as well, that edge word (an article, a filler)
            // is not part of the term and must not be swallowed. On an exact
            // tie a single letter is kept — it is a fragment of a spelled-out
            // abbreviation ("N S T E M E"), not a word of its own.
            if n > 1 {
                let edge_is_extraneous = |sub: Option<f64>, edge_word: &str| {
                    sub.is_some_and(|s| {
                        s < score || (s == score && letters(edge_word).count() >= 2)
                    })
                };
                let without_first = verdict_score(matcher.score_span(&span_keys[1..]));
                let without_last = verdict_score(matcher.score_span(&span_keys[..n - 1]));
                if edge_is_extraneous(without_first, words[i])
                    || edge_is_extraneous(without_last, words[i + n - 1])
                {
                    continue;
                }
            }
            if choice.best.is_none_or(|(_, _, best)| score < best) {
                choice.best = Some((n, term_index, score));
            }
        }
        memo[i] = Some(choice);
        choice
    };

    let mut report = CorrectionReport::default();
    let mut out: Vec<String> = Vec::with_capacity(words.len());
    let mut i = 0;
    while i < words.len() {
        let here = best_at(i);
        let mut chosen = here.best;
        // Look-ahead: a multi-word span starting here must not beat a strictly
        // better span that starts on its second word — otherwise a leading word
        // ("è Charge" before "Charge B") gets absorbed into the wrong match.
        if let Some((n, _, score)) = chosen {
            if n > 1 && i + 1 < words.len() {
                if let Some((_, _, next_score)) = best_at(i + 1).best {
                    if next_score < score {
                        chosen = None;
                    }
                }
            }
        }

        let Some((n, term_index, score)) = chosen else {
            if here.ambiguous && here.best.is_none() {
                report.ambiguous_spans.push(words[i].to_string());
            }
            out.push(words[i].to_string());
            i += 1;
            continue;
        };

        let span = &words[i..i + n];
        let term = &matcher.terms[term_index];
        let original = span.join(" ");
        let (prefix, _) = extract_punctuation(span[0]);
        let (_, suffix) = extract_punctuation(span[n - 1]);

        let Some(resolution) = resolve_span(&matcher, span, &keys[i..i + n], term_index) else {
            // The best-scoring term does not survive the word-level checks:
            // leave this word as spoken and move on.
            out.push(words[i].to_string());
            i += 1;
            continue;
        };

        match resolution {
            Resolution::KeepAsSpoken => {
                report.events.push(CorrectionEvent {
                    original: original.clone(),
                    term: term.clone(),
                    score,
                    kind: CorrectionKind::KeptInflected,
                });
                out.extend(span.iter().map(|w| w.to_string()));
            }
            Resolution::Replace(core) => {
                let at_sentence_start = i == 0
                    || words[i - 1]
                        .chars()
                        .last()
                        .is_some_and(|c| matches!(c, '.' | '!' | '?' | ':'));
                let rendered = format!(
                    "{}{}{}",
                    prefix,
                    adapt_case(span, &core, at_sentence_start),
                    suffix
                );
                if rendered != original {
                    report.events.push(CorrectionEvent {
                        original,
                        term: term.clone(),
                        score,
                        kind: CorrectionKind::Replaced,
                    });
                }
                out.push(rendered);
            }
        }
        i += n;
    }

    report.text = out.join(" ");
    report
}

/// Correct `text` against the active vocabulary. See
/// [`correct_with_vocabulary`] for the rules; this drops the audit trail.
pub fn apply_custom_words(text: &str, custom_words: &[String], threshold: f64) -> String {
    correct_with_vocabulary(text, custom_words, threshold).text
}

enum Resolution {
    /// The span already is the term in another grammatical form.
    KeepAsSpoken,
    /// Replace the span with this text (before case adaptation).
    Replace(String),
}

/// Decide what a matched span becomes, checking it word by word.
///
/// When span and term have the same number of words, each word must either be
/// the term's word (up to spelling), an inflected form of it, or within the
/// threshold on its own. A single substituted word fails the whole match, even
/// when the span as a whole is close — that is what keeps "…mit erhaltener
/// Ejektionsfraktion" from becoming "…mit reduzierter Ejektionsfraktion".
/// Inflected words are kept as spoken; the rest take the dictionary spelling,
/// rejoined with the term's own hyphens, slashes and spaces.
///
/// When the word counts differ, the transcript split or joined a compound:
///
/// - more transcript words than term words — the model split the term
///   ("N S T E M I", "Mehrgefäß Erkrankung"): at most [`MISALIGNED_EDIT_CAP`]
///   edits, same onset, and not a span written entirely in lower case when the
///   term is a capitals abbreviation;
/// - fewer transcript words — the term has a word the speaker did not say
///   ("Koronarangiographie" vs "CT-Koronarangiographie"): only a pure joining,
///   with zero edits, is accepted. Anything else would add content.
fn resolve_span(
    matcher: &Matcher,
    span: &[&str],
    span_keys: &[String],
    term_index: usize,
) -> Option<Resolution> {
    let term = &matcher.terms[term_index];
    let (term_words, seps) = term_parts(term);

    if term_words.len() != span.len() {
        let joined = span_keys.concat();
        let edits = matcher.edits_to(term_index, &joined);
        let allowed = if span.len() > term_words.len() {
            edits <= MISALIGNED_EDIT_CAP
                && onset_class(&joined) == onset_class(&fold_key(term))
                && !(is_capitals_abbreviation(term)
                    && span.iter().all(|w| !letters(w).any(char::is_uppercase)))
        } else {
            edits == 0
        };
        return allowed.then(|| Resolution::Replace(term.clone()));
    }

    let mut pieces: Vec<String> = Vec::with_capacity(span.len());
    let mut kept_inflection = false;
    let mut corrected_spelling = false;
    for (word, term_word) in span.iter().zip(&term_words) {
        let core = {
            let (prefix, suffix) = extract_punctuation(word);
            &word[prefix.len()..word.len() - suffix.len()]
        };
        // A capitals abbreviation never replaces an ordinary lower-case word,
        // not even on an exact key match: "des" is an article, not "DES".
        if is_capitals_abbreviation(term_word) && has_lowercase(core) {
            return None;
        }
        if fold_key(core) == fold_key(term_word) {
            if core != *term_word {
                corrected_spelling = true;
            }
            pieces.push(term_word.to_string());
        } else if is_inflection_variant(core, term_word) || is_derived_form(core, term_word) {
            kept_inflection = true;
            pieces.push(core.to_string());
        } else if word_close(core, term_word, matcher.threshold) {
            corrected_spelling = true;
            pieces.push(term_word.to_string());
        } else {
            return None;
        }
    }

    // Hyphens, slashes and spaces the transcript lost are a spelling fix too.
    let rejoined: String = pieces
        .iter()
        .zip(&seps)
        .map(|(piece, sep)| format!("{}{}", piece, sep))
        .collect();
    let spoken_joined = span
        .iter()
        .map(|w| {
            let (prefix, suffix) = extract_punctuation(w);
            &w[prefix.len()..w.len() - suffix.len()]
        })
        .collect::<Vec<_>>()
        .join(" ");
    if rejoined != spoken_joined {
        corrected_spelling = true;
    }

    if kept_inflection && !corrected_spelling {
        Some(Resolution::KeepAsSpoken)
    } else {
        Some(Resolution::Replace(rejoined))
    }
}

/// The canonical dictionary spelling, adjusted only where the transcript's
/// casing carries meaning:
///
/// - a span written entirely in capitals keeps that ("CHARGE B" → "CHARGEBEE");
///   a leading abbreviation alone ("AV Knoten …") does not count;
/// - at the start of a sentence, a capital the transcript already had is kept
///   even if the dictionary spells the term in lower case ("Paroxysmale …").
///
/// Otherwise the dictionary's own spelling wins, including umlauts, hyphens and
/// slashes the transcript may have lost.
fn adapt_case(span: &[&str], canonical: &str, at_sentence_start: bool) -> String {
    let letters_upper = |w: &&str| {
        let letters: Vec<char> = w.chars().filter(|c| c.is_alphabetic()).collect();
        !letters.is_empty() && letters.iter().all(|c| c.is_uppercase())
    };
    let shouting = span.iter().all(letters_upper)
        && span
            .iter()
            .any(|w| w.chars().filter(|c| c.is_alphabetic()).count() >= 2);
    if shouting {
        return canonical.to_uppercase();
    }
    let original_capitalised = span
        .first()
        .and_then(|w| w.chars().find(|c| c.is_alphabetic()))
        .is_some_and(char::is_uppercase);
    if at_sentence_start && original_capitalised {
        let mut chars = canonical.chars();
        if let Some(first) = chars.next() {
            if first.is_lowercase() {
                return first.to_uppercase().chain(chars).collect();
            }
        }
    }
    canonical.to_string()
}

/// Extracts punctuation prefix and suffix from a word
fn extract_punctuation(word: &str) -> (&str, &str) {
    // String slices use byte offsets. Derive both boundaries from char_indices
    // so multibyte punctuation such as `。` and `「」` can never be split.
    let prefix_end = word
        .char_indices()
        .find(|(_, c)| c.is_alphanumeric())
        .map(|(index, _)| index)
        .unwrap_or(word.len());
    let suffix_start = word
        .char_indices()
        .rev()
        .find(|(_, c)| c.is_alphanumeric())
        .map(|(index, c)| index + c.len_utf8())
        .unwrap_or(0);

    let prefix = if prefix_end > 0 {
        &word[..prefix_end]
    } else {
        ""
    };

    let suffix = if suffix_start < word.len() {
        &word[suffix_start..]
    } else {
        ""
    };

    (prefix, suffix)
}

/// Evidence for the language of the text being cleaned.
///
/// This intentionally describes the transcription output, not Handy's UI
/// language. Unknown output languages fail closed: built-in filler removal is
/// skipped rather than applying a language profile speculatively.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OutputLanguageEvidence {
    UserSelected(String),
    ModelConstrained(String),
    /// The transcription model itself identified the language (audio-based
    /// LID, e.g. Whisper in auto mode).
    ModelDetected(String),
    /// Detected from the transcribed text with high confidence, constrained to
    /// the model's supported languages. Weakest accepted evidence.
    TextDetected(String),
    TranslatedToEnglish,
    Unknown,
}

impl OutputLanguageEvidence {
    fn language(&self) -> Option<&str> {
        match self {
            Self::UserSelected(language)
            | Self::ModelConstrained(language)
            | Self::ModelDetected(language)
            | Self::TextDetected(language) => Some(language),
            Self::TranslatedToEnglish => Some("en"),
            Self::Unknown => None,
        }
    }
}

/// Filler tokens that are not lexical words in any language Handy's models can
/// output, so removing them cannot corrupt text regardless of the (possibly
/// unknown) output language. Kept deliberately conservative: anything that is a
/// real word somewhere ("um" pt/de, "ha" es, "ah"/"eh" interjections, "mm"
/// millimetres) belongs in the language-gated lists instead.
const UNIVERSAL_FILLER_WORDS: &[&str] = &[
    "uh", "uhm", "umm", "uhh", "uhhh", "ehh", "ehm", "ahm", "hmm", "hm", "mmm", "хм", "ммм",
];

/// Filler words that are only safe to remove with evidence for the output
/// language, because the same token is a real word elsewhere (e.g. Portuguese
/// "um" = "a/an", German "um" = "at/around", Spanish "ha" = "has").
fn gated_filler_words_for_language(lang: &str) -> &'static [&'static str] {
    let base_lang = lang.split(&['-', '_'][..]).next().unwrap_or(lang);

    match base_lang {
        "en" => &["um", "ah", "eh", "ha"],
        "de" => &["äh", "ähm"],
        "fr" => &["euh"],
        _ => &[],
    }
}

static MULTI_SPACE_PATTERN: Lazy<Regex> = Lazy::new(|| Regex::new(r"\s{2,}").unwrap());

/// Collapses repeated words (3+ repetitions) to a single instance.
/// E.g., "wh wh wh wh" -> "wh", "I I I I" -> "I"
fn collapse_stutters(text: &str) -> String {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() {
        return text.to_string();
    }

    let mut result: Vec<&str> = Vec::new();
    let mut i = 0;

    while i < words.len() {
        let word = words[i];
        let word_lower = word.to_lowercase();

        if word_lower.chars().all(|c| c.is_alphabetic()) {
            // Count consecutive repetitions (case-insensitive)
            let mut count = 1;
            while i + count < words.len() && words[i + count].to_lowercase() == word_lower {
                count += 1;
            }

            // If 3+ repetitions, collapse to single instance
            if count >= 3 {
                result.push(word);
                i += count;
            } else {
                result.push(word);
                i += 1;
            }
        } else {
            result.push(word);
            i += 1;
        }
    }

    result.join(" ")
}

/// Removes filler words from transcription output when enabled.
///
/// Built-in removal is two-tiered: [`UNIVERSAL_FILLER_WORDS`] apply regardless
/// of language evidence, while [`gated_filler_words_for_language`] tokens are
/// only removed when the output language is known. A custom list is an
/// explicit user override and replaces both tiers without requiring language
/// evidence. `Some(empty vec)` disables removal, preserving the legacy
/// power-user setting. The master toggle takes precedence over both built-in
/// and custom lists.
///
/// # Arguments
/// * `text` - The raw transcription text to filter
/// * `language` - Evidence for the language of the transcription output
/// * `custom_filler_words` - Optional user-provided filler word list. `Some(vec)` overrides
///   language defaults; `Some(empty vec)` disables filtering; `None` uses language defaults.
/// * `enabled` - Whether filler-word removal is enabled
///
/// # Returns
/// The text with configured filler words removed
pub fn remove_filler_words(
    text: &str,
    language: &OutputLanguageEvidence,
    custom_filler_words: &Option<Vec<String>>,
    enabled: bool,
) -> String {
    if !enabled {
        return text.to_string();
    }

    // Build filler patterns from custom list or the built-in tiers
    let patterns: Vec<Regex> = match custom_filler_words {
        Some(words) => words
            .iter()
            .filter_map(|word| Regex::new(&format!(r"(?i)\b{}\b[,.]?", regex::escape(word))).ok())
            .collect(),
        None => UNIVERSAL_FILLER_WORDS
            .iter()
            .chain(
                language
                    .language()
                    .map(gated_filler_words_for_language)
                    .unwrap_or_default(),
            )
            .map(|word| Regex::new(&format!(r"(?i)\b{}\b[,.]?", regex::escape(word))).unwrap())
            .collect(),
    };

    // Remove filler words
    let mut filtered = text.to_string();
    for pattern in &patterns {
        filtered = pattern.replace_all(&filtered, "").to_string();
    }

    filtered
}

/// Applies non-filler transcription cleanup.
///
/// Kept separate from [`remove_filler_words`] so disabling filler deletion
/// does not also disable the existing repeated-word and whitespace cleanup.
pub fn normalize_transcription_output(text: &str) -> String {
    let mut normalized = collapse_stutters(text);

    // Clean up multiple spaces to single space
    normalized = MULTI_SPACE_PATTERN
        .replace_all(&normalized, " ")
        .to_string();

    // Trim leading/trailing whitespace
    normalized.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Exercise the complete cleanup sequence with an explicitly selected
    /// language. Individual tests below predate the split between filler
    /// removal and non-filler normalization.
    fn filter_transcription_output(
        text: &str,
        language: &str,
        custom_filler_words: &Option<Vec<String>>,
    ) -> String {
        let language = OutputLanguageEvidence::UserSelected(language.to_string());
        let filtered = remove_filler_words(text, &language, custom_filler_words, true);
        normalize_transcription_output(&filtered)
    }

    #[test]
    fn test_apply_custom_words_exact_match() {
        let text = "hello world";
        let custom_words = vec!["Hello".to_string(), "World".to_string()];
        let result = apply_custom_words(text, &custom_words, 0.5);
        assert_eq!(result, "Hello World");
    }

    #[test]
    fn test_apply_custom_words_fuzzy_match() {
        let text = "helo wrold";
        let custom_words = vec!["hello".to_string(), "world".to_string()];
        let result = apply_custom_words(text, &custom_words, 0.5);
        assert_eq!(result, "hello world");
    }

    #[test]
    fn test_adapt_case() {
        // A fully capitalised span stays capitalised.
        assert_eq!(adapt_case(&["HELLO"], "world", false), "WORLD");
        // A sentence-initial capital the transcript had is kept…
        assert_eq!(adapt_case(&["Hello"], "world", true), "World");
        // …but mid-sentence the dictionary's spelling wins.
        assert_eq!(adapt_case(&["Hello"], "world", false), "world");
        // An uncapitalised original never forces capitals.
        assert_eq!(adapt_case(&["hello"], "WORLD", true), "WORLD");
        assert_eq!(adapt_case(&["hello"], "world", true), "world");
        // A leading abbreviation does not make the whole span "shouting".
        assert_eq!(
            adapt_case(
                &["AV", "Knoten", "Reentrytachykardie"],
                "AV-Knoten-Reentrytachykardie",
                true
            ),
            "AV-Knoten-Reentrytachykardie"
        );
    }

    #[test]
    fn test_extract_punctuation() {
        assert_eq!(extract_punctuation("hello"), ("", ""));
        assert_eq!(extract_punctuation("!hello?"), ("!", "?"));
        assert_eq!(extract_punctuation("...hello..."), ("...", "..."));
    }

    #[test]
    fn test_extract_punctuation_uses_unicode_boundaries() {
        assert_eq!(extract_punctuation("你好。"), ("", "。"));
        assert_eq!(extract_punctuation("「你好」"), ("「", "」"));
        assert_eq!(extract_punctuation("你好！"), ("", "！"));
    }

    #[test]
    fn test_empty_custom_words() {
        let text = "hello world";
        let custom_words = vec![];
        let result = apply_custom_words(text, &custom_words, 0.5);
        assert_eq!(result, "hello world");
    }

    #[test]
    fn test_filter_filler_words() {
        let text = "So uhm I was thinking uh about this";
        let result = filter_transcription_output(text, "en", &None);
        assert_eq!(result, "So I was thinking about this");
    }

    #[test]
    fn test_filter_filler_words_case_insensitive() {
        let text = "UHM this is UH a test";
        let result = filter_transcription_output(text, "en", &None);
        assert_eq!(result, "this is a test");
    }

    #[test]
    fn test_filter_filler_words_with_punctuation() {
        let text = "Well, uhm, I think, uh. that's right";
        let result = filter_transcription_output(text, "en", &None);
        assert_eq!(result, "Well, I think, that's right");
    }

    #[test]
    fn test_filter_cleans_whitespace() {
        let text = "Hello    world   test";
        let result = filter_transcription_output(text, "en", &None);
        assert_eq!(result, "Hello world test");
    }

    #[test]
    fn test_filter_trims() {
        let text = "  Hello world  ";
        let result = filter_transcription_output(text, "en", &None);
        assert_eq!(result, "Hello world");
    }

    #[test]
    fn test_filter_combined() {
        let text = "  Uhm, so I was, uh, thinking about this  ";
        let result = filter_transcription_output(text, "en", &None);
        assert_eq!(result, "so I was, thinking about this");
    }

    #[test]
    fn test_filter_preserves_valid_text() {
        let text = "This is a completely normal sentence.";
        let result = filter_transcription_output(text, "en", &None);
        assert_eq!(result, "This is a completely normal sentence.");
    }

    #[test]
    fn test_filter_stutter_collapse() {
        let text = "w wh wh wh wh wh wh wh wh wh why";
        let result = filter_transcription_output(text, "en", &None);
        assert_eq!(result, "w wh why");
    }

    #[test]
    fn test_filter_stutter_short_words() {
        let text = "I I I I think so so so so";
        let result = filter_transcription_output(text, "en", &None);
        assert_eq!(result, "I think so");
    }

    #[test]
    fn test_filter_stutter_longer_words() {
        let text = "Check data doc doc doc doc documentation.";
        let result = filter_transcription_output(text, "en", &None);
        assert_eq!(result, "Check data doc documentation.");
    }

    #[test]
    fn test_filter_stutter_mixed_case() {
        let text = "No NO no NO no";
        let result = filter_transcription_output(text, "en", &None);
        assert_eq!(result, "No");
    }

    #[test]
    fn test_filter_stutter_preserves_two_repetitions() {
        let text = "no no is fine";
        let result = filter_transcription_output(text, "en", &None);
        assert_eq!(result, "no no is fine");
    }

    #[test]
    fn test_filter_english_removes_um() {
        let text = "um I think um this is good";
        let result = filter_transcription_output(text, "en", &None);
        assert_eq!(result, "I think this is good");
    }

    #[test]
    fn test_filter_portuguese_preserves_um() {
        // "um" means "a/an" in Portuguese
        let text = "um gato bonito";
        let result = filter_transcription_output(text, "pt", &None);
        assert_eq!(result, "um gato bonito");
    }

    #[test]
    fn test_filter_spanish_preserves_ha() {
        // "ha" means "has" in Spanish
        let text = "ha sido un buen día";
        let result = filter_transcription_output(text, "es", &None);
        assert_eq!(result, "ha sido un buen día");
    }

    #[test]
    fn test_filter_language_code_with_region() {
        // "pt-BR" should normalize to "pt"
        let text = "um gato bonito";
        let result = filter_transcription_output(text, "pt-BR", &None);
        assert_eq!(result, "um gato bonito");
    }

    #[test]
    fn test_filter_custom_filler_words_override() {
        let custom = Some(vec!["okay".to_string(), "right".to_string()]);
        let text = "okay so I think right this works";
        let result = filter_transcription_output(text, "en", &custom);
        assert_eq!(result, "so I think this works");
    }

    #[test]
    fn test_filter_custom_filler_words_empty_disables() {
        let custom = Some(vec![]);
        let text = "So uhm I was thinking uh about this";
        let result = filter_transcription_output(text, "en", &custom);
        // No filler words removed since custom list is empty
        assert_eq!(result, "So uhm I was thinking uh about this");
    }

    #[test]
    fn test_filter_unknown_language_still_removes_universal_fillers() {
        let text = "uh I think uhm this works";
        let result = filter_transcription_output(text, "xx", &None);
        assert_eq!(result, "I think this works");
    }

    #[test]
    fn test_filter_unknown_language_does_not_remove_um() {
        let text = "um I think this works";
        let result = filter_transcription_output(text, "xx", &None);
        assert_eq!(result, "um I think this works");
    }

    #[test]
    fn test_filter_unknown_evidence_removes_universal_keeps_gated() {
        let filtered = remove_filler_words(
            "uhh bueno hmm creo que um ha llegado",
            &OutputLanguageEvidence::Unknown,
            &None,
            true,
        );
        assert_eq!(
            normalize_transcription_output(&filtered),
            "bueno creo que um ha llegado"
        );

        let cyrillic = remove_filler_words(
            "хм я думаю ммм это работает",
            &OutputLanguageEvidence::Unknown,
            &None,
            true,
        );
        assert_eq!(
            normalize_transcription_output(&cyrillic),
            "я думаю это работает"
        );
    }

    #[test]
    fn test_filter_german_gated_fillers_require_evidence() {
        let text = "äh ich glaube ähm das passt";

        let unknown = remove_filler_words(text, &OutputLanguageEvidence::Unknown, &None, true);
        assert_eq!(normalize_transcription_output(&unknown), text);

        let result = filter_transcription_output(text, "de", &None);
        assert_eq!(result, "ich glaube das passt");
    }

    #[test]
    fn test_filter_preserves_millimetre_unit() {
        // "mm" was removed from the filler lists because it eats units.
        let text = "the screw is 5 mm long";
        let result = filter_transcription_output(text, "en", &None);
        assert_eq!(result, "the screw is 5 mm long");
    }

    #[test]
    fn test_filter_detected_evidence_unlocks_gated_fillers() {
        let model = remove_filler_words(
            "um I think this works",
            &OutputLanguageEvidence::ModelDetected("en".to_string()),
            &None,
            true,
        );
        assert_eq!(normalize_transcription_output(&model), "I think this works");

        let text = remove_filler_words(
            "euh je pense que ça marche",
            &OutputLanguageEvidence::TextDetected("fr".to_string()),
            &None,
            true,
        );
        assert_eq!(
            normalize_transcription_output(&text),
            "je pense que ça marche"
        );
    }

    #[test]
    fn test_filter_master_toggle_disables_custom_and_builtin_removal() {
        let text = "um customword I think";
        let language = OutputLanguageEvidence::UserSelected("en".to_string());
        let custom = Some(vec!["customword".to_string()]);

        let result = remove_filler_words(text, &language, &custom, false);

        assert_eq!(result, text);
    }

    #[test]
    fn test_filter_custom_words_apply_without_language_evidence() {
        let custom = Some(vec!["customword".to_string()]);
        let text = "customword should be removed but um should remain";

        let filtered = remove_filler_words(text, &OutputLanguageEvidence::Unknown, &custom, true);
        let result = normalize_transcription_output(&filtered);

        assert_eq!(result, "should be removed but um should remain");
    }

    #[test]
    fn test_apply_custom_words_ngram_two_words() {
        let text = "il cui nome è Charge B, che permette";
        let custom_words = vec!["ChargeBee".to_string()];
        let result = apply_custom_words(text, &custom_words, 0.5);
        assert!(result.contains("ChargeBee,"), "unexpected result: {result}");
        assert!(!result.contains("Charge B"));
    }

    #[test]
    fn test_apply_custom_words_ngram_three_words() {
        let text = "use Chat G P T for this";
        let custom_words = vec!["ChatGPT".to_string()];
        let result = apply_custom_words(text, &custom_words, 0.5);
        assert!(result.contains("ChatGPT"));
    }

    #[test]
    fn test_apply_custom_words_prefers_longer_ngram() {
        let text = "Open AI GPT model";
        let custom_words = vec!["OpenAI".to_string(), "GPT".to_string()];
        let result = apply_custom_words(text, &custom_words, 0.5);
        assert_eq!(result, "OpenAI GPT model");
    }

    #[test]
    fn test_apply_custom_words_ngram_preserves_case() {
        let text = "CHARGE B is great";
        let custom_words = vec!["ChargeBee".to_string()];
        let result = apply_custom_words(text, &custom_words, 0.5);
        assert!(result.contains("CHARGEBEE"));
    }

    #[test]
    fn test_apply_custom_words_ngram_with_spaces_in_custom() {
        // Custom word with space should also match against split words
        let text = "using Mac Book Pro";
        let custom_words = vec!["MacBook Pro".to_string()];
        let result = apply_custom_words(text, &custom_words, 0.5);
        assert_eq!(result, "using MacBook Pro");
    }

    #[test]
    fn test_apply_custom_words_trailing_number_not_doubled() {
        // Verify that trailing non-alpha chars (like numbers) aren't double-counted
        // between build_ngram stripping them and extract_punctuation capturing them
        let text = "use GPT4 for this";
        let custom_words = vec!["GPT-4".to_string()];
        let result = apply_custom_words(text, &custom_words, 0.5);
        // Should NOT produce "GPT-44" (double-counting the trailing 4)
        assert!(
            !result.contains("GPT-44"),
            "got double-counted result: {}",
            result
        );
    }

    #[test]
    fn test_apply_custom_words_matches_ampersand_word() {
        let text = "send it to RD for review";
        let custom_words = vec!["R&D".to_string()];
        let result = apply_custom_words(text, &custom_words, 0.18);
        assert_eq!(result, "send it to R&D for review");
    }

    #[test]
    fn test_apply_custom_words_matches_spoken_ampersand_word() {
        let text = "send it to R and D for review";
        let custom_words = vec!["R&D".to_string()];
        let result = apply_custom_words(text, &custom_words, 0.18);
        assert_eq!(result, "send it to R&D for review");
    }

    #[test]
    fn test_apply_custom_words_preserves_ampersand_word() {
        let text = "send it to R&D for review";
        let custom_words = vec!["R&D".to_string()];
        let result = apply_custom_words(text, &custom_words, 0.18);
        assert_eq!(result, "send it to R&D for review");
    }

    #[test]
    fn test_apply_custom_words_handles_unicode_punctuation() {
        let text = "「Handee。」";
        let custom_words = vec!["Handy".to_string()];
        let result = apply_custom_words(text, &custom_words, 0.5);
        assert_eq!(result, "「Handy。」");
    }

    #[test]
    fn test_apply_custom_words_skips_cjk_fuzzy_matching() {
        let text = "你好。";
        let custom_words = vec!["你号".to_string()];
        let result = apply_custom_words(text, &custom_words, 1.0);
        assert_eq!(result, text);
    }

    // -----------------------------------------------------------------------
    // German medical regression suite. Terms are taken verbatim from the
    // bundled dictionaries; threshold is the app default (0.18). Each negative
    // case is a correct transcript that an unsafe corrector would change.
    // -----------------------------------------------------------------------

    const T: f64 = 0.18;

    fn fix(text: &str, dict: &[&str]) -> String {
        let dict: Vec<String> = dict.iter().map(|s| s.to_string()).collect();
        apply_custom_words(text, &dict, T)
    }

    // --- positive: realistic ASR variants that should be repaired ----------

    #[test]
    fn de_umlaut_written_as_plain_vowel_is_restored() {
        assert_eq!(
            fix(
                "mit linksventrikularer Ausflusstrakt",
                &["linksventrikulärer Ausflusstrakt"]
            ),
            "mit linksventrikulärer Ausflusstrakt"
        );
        assert_eq!(
            fix("ein Praexzitationssyndrom", &["Präexzitationssyndrom"]),
            "ein Präexzitationssyndrom"
        );
    }

    #[test]
    fn de_sharp_s_written_as_ss_is_restored() {
        assert_eq!(
            fix("bekannte Mehrgefässerkrankung", &["Mehrgefäßerkrankung"]),
            "bekannte Mehrgefäßerkrankung"
        );
    }

    #[test]
    fn de_ph_f_spelling_variant_is_normalised() {
        assert_eq!(
            fix("die Echokardiografie zeigte", &["Echokardiographie"]),
            "die Echokardiographie zeigte"
        );
    }

    #[test]
    fn de_split_compounds_regain_hyphens_and_slashes() {
        assert_eq!(
            fix(
                "Verdacht auf Wolff Parkinson White Syndrom",
                &["Wolff-Parkinson-White-Syndrom"]
            ),
            "Verdacht auf Wolff-Parkinson-White-Syndrom"
        );
        assert_eq!(
            fix(
                "eine Torsade de Pointes Tachykardie",
                &["Torsade-de-pointes-Tachykardie"]
            ),
            "eine Torsade-de-pointes-Tachykardie"
        );
        assert_eq!(
            fix(
                "Umstellung auf Sacubitril Valsartan",
                &["Sacubitril/Valsartan"]
            ),
            "Umstellung auf Sacubitril/Valsartan"
        );
    }

    #[test]
    fn de_long_multiword_terms_are_reachable() {
        // 53-character key, five words — beyond the old 50-char / 3-word limits.
        assert_eq!(
            fix(
                "Herzinsuffizienz mit leicht reduzierter Ejektionsfraktio bekannt",
                &["Herzinsuffizienz mit leicht reduzierter Ejektionsfraktion"]
            ),
            "Herzinsuffizienz mit leicht reduzierter Ejektionsfraktion bekannt"
        );
        assert_eq!(
            fix(
                "Stenose der Arteria karotis communis dextra",
                &["Arteria carotis communis dextra"]
            ),
            "Stenose der Arteria carotis communis dextra"
        );
    }

    #[test]
    fn de_terms_with_digits_match_only_the_same_digits() {
        assert_eq!(
            fix("der CHA2DS2 VASc Score", &["CHA2DS2-VASc-Score"]),
            "der CHA2DS2-VASc-Score"
        );
        assert_eq!(
            fix("ein P2Y12 Inhibitor", &["P2Y12-Inhibitor"]),
            "ein P2Y12-Inhibitor"
        );
        // T2 is never "corrected" into T1 — the digits differ.
        assert_eq!(fix("im T2 Mapping", &["T1-Mapping"]), "im T2 Mapping");
        assert_eq!(
            fix("im T2 Mapping", &["T1-Mapping", "T2-Mapping"]),
            "im T2-Mapping"
        );
    }

    #[test]
    fn de_drug_typo_is_fixed_and_its_dose_kept() {
        assert_eq!(
            fix("Bisoprolo 2,5 mg morgens", &["Bisoprolol", "Metoprolol"]),
            "Bisoprolol 2,5 mg morgens"
        );
    }

    // --- negative: correct transcripts that must stay exactly as they are --

    #[test]
    fn de_a_different_diagnosis_is_never_substituted() {
        assert_eq!(
            fix("Mitralklappenstenose", &["Mitralklappeninsuffizienz"]),
            "Mitralklappenstenose"
        );
        assert_eq!(
            fix(
                "Z. n. Mitralklappenrekonstruktion",
                &["Mitralklappeninsuffizienz", "Mitralklappenanulus"]
            ),
            "Z. n. Mitralklappenrekonstruktion"
        );
        assert_eq!(
            fix(
                "Herzinsuffizienz mit erhaltener Ejektionsfraktion",
                &["Herzinsuffizienz mit reduzierter Ejektionsfraktion"]
            ),
            "Herzinsuffizienz mit erhaltener Ejektionsfraktion"
        );
    }

    /// Opposite findings that differ by only a few letters. The dictionary
    /// holds one side; the transcript correctly says the other.
    #[test]
    fn de_opposite_findings_are_never_substituted() {
        assert_eq!(
            fix("bekannte arterielle Hypotonie", &["arterielle Hypertonie"]),
            "bekannte arterielle Hypotonie"
        );
        assert_eq!(fix("Hypotonie", &["Hypertonie"]), "Hypotonie");
        assert_eq!(
            fix("Sinusbradykardie", &["Sinustachykardie"]),
            "Sinusbradykardie"
        );
        assert_eq!(
            fix(
                "im rechtsventrikulären Ausflusstrakt",
                &["linksventrikulärer Ausflusstrakt"]
            ),
            "im rechtsventrikulären Ausflusstrakt"
        );
        assert_eq!(
            fix(
                "im rechtsventrikulärer Ausflusstrakt",
                &["linksventrikulärer Ausflusstrakt"]
            ),
            "im rechtsventrikulärer Ausflusstrakt"
        );
    }

    #[test]
    fn de_sides_are_never_swapped() {
        let dict = [
            "Arteria carotis communis dextra",
            "Arteria carotis communis sinistra",
        ];
        assert_eq!(
            fix("Arteria carotis communis sinistra", &dict),
            "Arteria carotis communis sinistra"
        );
        assert_eq!(
            fix("Arteria carotis communis dextra", &dict),
            "Arteria carotis communis dextra"
        );
    }

    #[test]
    fn de_negations_are_never_removed_or_absorbed() {
        let dict = ["Aortenklappenstenose", "Perikarderguss"];
        assert_eq!(
            fix("kein Aortenklappenstenose", &dict),
            "kein Aortenklappenstenose"
        );
        assert_eq!(
            fix("keine Aortenklappenstenose", &dict),
            "keine Aortenklappenstenose"
        );
        assert_eq!(fix("kein Perikardergus", &dict), "kein Perikarderguss");
        assert_eq!(fix("nicht Perikarderguss", &dict), "nicht Perikarderguss");
        assert_eq!(fix("ohne Perikarderguss", &dict), "ohne Perikarderguss");
    }

    #[test]
    fn de_inflected_forms_are_left_as_spoken() {
        assert_eq!(
            fix("mehrere Aortenklappenstenosen", &["Aortenklappenstenose"]),
            "mehrere Aortenklappenstenosen"
        );
        assert_eq!(
            fix(
                "des linksventrikulären Ausflusstrakts",
                &["linksventrikulärer Ausflusstrakt"]
            ),
            "des linksventrikulären Ausflusstrakts"
        );
    }

    #[test]
    fn de_ordinary_words_are_not_turned_into_terms() {
        assert_eq!(
            fix("Urlaub in den Tropen", &["Troponin"]),
            "Urlaub in den Tropen"
        );
        assert_eq!(fix("ich habe ein EKG", &["EKG"]), "ich habe ein EKG");
        assert_eq!(
            fix("Die Werte sind normal", &["Aorta", "Ileus"]),
            "Die Werte sind normal"
        );
    }

    #[test]
    fn de_numbers_and_doses_are_never_changed() {
        let dict = ["Ramipril", "Bisoprolol"];
        assert_eq!(fix("Ramipril 5 mg täglich", &dict), "Ramipril 5 mg täglich");
        assert_eq!(fix("Ramipril 2,5 mg", &dict), "Ramipril 2,5 mg");
        assert_eq!(
            fix("Bisoprolol 10 mg 1-0-0", &dict),
            "Bisoprolol 10 mg 1-0-0"
        );
        assert_eq!(fix("RR 135/85 mmHg", &dict), "RR 135/85 mmHg");
    }

    #[test]
    fn de_sentence_boundaries_are_respected() {
        assert_eq!(
            fix("Vorhof. Flimmern", &["Vorhofflimmern"]),
            "Vorhof. Flimmern"
        );
    }

    #[test]
    fn de_near_equal_drug_candidates_cause_abstention() {
        // One edit to Nifedipin, two to Nitrendipin: too close to call.
        assert_eq!(
            fix("Gabe von Nitedipin", &["Nifedipin", "Nitrendipin"]),
            "Gabe von Nitedipin"
        );
        // With only one plausible drug, the fix is made.
        assert_eq!(
            fix("Gabe von Nitedipin", &["Nifedipin"]),
            "Gabe von Nifedipin"
        );
        let report = correct_with_vocabulary(
            "Gabe von Nitedipin",
            &["Nifedipin".to_string(), "Nitrendipin".to_string()],
            T,
        );
        assert_eq!(report.ambiguous_spans, vec!["Nitedipin".to_string()]);
    }

    #[test]
    fn de_report_lists_every_replacement_and_kept_inflection() {
        let report = correct_with_vocabulary(
            "Echokardiografie ohne Aortenklappenstenosen",
            &[
                "Echokardiographie".to_string(),
                "Aortenklappenstenose".to_string(),
            ],
            T,
        );
        assert_eq!(report.text, "Echokardiographie ohne Aortenklappenstenosen");
        let kinds: Vec<CorrectionKind> = report.events.iter().map(|e| e.kind).collect();
        assert_eq!(
            kinds,
            vec![CorrectionKind::Replaced, CorrectionKind::KeptInflected]
        );
        assert_eq!(report.events[0].original, "Echokardiografie");
        assert_eq!(report.events[0].term, "Echokardiographie");
    }

    #[test]
    fn de_correct_terms_are_untouched_and_unreported() {
        let report = correct_with_vocabulary(
            "Die Echokardiographie war unauffällig.",
            &["Echokardiographie".to_string()],
            T,
        );
        assert_eq!(report.text, "Die Echokardiographie war unauffällig.");
        assert!(report.events.is_empty());
    }

    // Found by the synthetic benchmark run on 2026-09-19: each of these was a
    // correct transcript that the corrector damaged.

    #[test]
    fn de_a_term_is_not_padded_with_a_word_the_speaker_did_not_say() {
        // Only the CT variant was in the active pool.
        assert_eq!(
            fix(
                "zur Koronarangiographie aufgenommen",
                &["CT-Koronarangiographie"]
            ),
            "zur Koronarangiographie aufgenommen"
        );
        // Joining without any edit is still a legitimate repair.
        assert_eq!(
            fix("eine Sinustachykardie", &["Sinus-Tachykardie"]),
            "eine Sinus-Tachykardie"
        );
    }

    #[test]
    fn de_derived_words_keep_their_part_of_speech() {
        assert_eq!(
            fix("Echokardiographisch zeigte sich", &["Echokardiographie"]),
            "Echokardiographisch zeigte sich"
        );
        assert_eq!(
            fix("echokardiographische Kontrolle", &["Echokardiographie"]),
            "echokardiographische Kontrolle"
        );
    }

    #[test]
    fn de_a_changed_onset_is_not_a_spelling_slip() {
        assert_eq!(
            fix(
                "Die linksventrikuläre Funktion war erhalten.",
                &["Punktion"]
            ),
            "Die linksventrikuläre Funktion war erhalten."
        );
        // Onset classes still cover real spelling variants.
        assert_eq!(
            fix("Arteria karotis communis", &["Arteria carotis communis"]),
            "Arteria carotis communis"
        );
        assert_eq!(fix("Fenprocoumon", &["Phenprocoumon"]), "Phenprocoumon");
    }

    #[test]
    fn de_capitals_abbreviations_never_replace_ordinary_words() {
        assert_eq!(
            fix("Pointe des Tachykardie", &["DES"]),
            "Pointe des Tachykardie"
        );
        assert_eq!(fix("Des Weiteren", &["DES"]), "Des Weiteren");
        // Written in capitals, the abbreviation is still recognised.
        assert_eq!(
            fix("Implantation eines DES", &["DES"]),
            "Implantation eines DES"
        );
        assert_eq!(fix("bei N S T E M E", &["NSTEMI"]), "bei NSTEMI");
        assert_eq!(fix("bei n s t e m i", &["NSTEMI"]), "bei n s t e m i");
    }

    /// Guards against pathological slowdowns with the full bundled vocabulary —
    /// every module at its largest tier, which is the worst case a user can
    /// configure (~9 900 terms).
    ///
    /// Measured 2026-09-20 on an M5: the release build needs 0.33 s for a
    /// 130-word dictation against that pool, the debug build roughly 30x that.
    /// The sentence is kept short so the suite stays quick; the bound is
    /// deliberately generous (debug build, shared CI machines), and the
    /// benchmark harness is where real latency is measured.
    #[test]
    fn de_full_vocabulary_stays_fast() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("dictionaries/de");
        let mut pool: Vec<String> = Vec::new();
        for entry in std::fs::read_dir(dir).unwrap() {
            let raw = std::fs::read_to_string(entry.unwrap().path()).unwrap();
            pool.extend(
                raw.lines()
                    .map(str::trim)
                    .filter(|l| !l.is_empty() && !l.starts_with('#'))
                    .map(String::from),
            );
        }
        assert!(
            pool.len() > 1000,
            "expected the full bundle, got {}",
            pool.len()
        );
        let sentence = "Bei Aufnahme zeigte sich in der Echokardiografie eine hochgradige \
            Aortenklappenstenose ohne Perikarderguss, Ramipril 5 mg wurde fortgeführt. ";
        let text = sentence.repeat(2); // ~33 words, a short dictation
        let started = std::time::Instant::now();
        let out = apply_custom_words(&text, &pool, T);
        let elapsed = started.elapsed();
        assert!(out.contains("Echokardiographie"));
        assert!(out.contains("ohne Perikarderguss"));
        assert!(out.contains("Ramipril 5 mg"));
        assert!(elapsed.as_secs_f64() < 10.0, "took {:?}", elapsed);
    }
}
