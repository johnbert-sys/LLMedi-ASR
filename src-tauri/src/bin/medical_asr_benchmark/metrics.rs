//! Scoring for the medical ASR benchmark.
//!
//! Everything here is deliberately conservative. The point of the harness is to
//! measure what the pipeline really produces, so normalisation must not quietly
//! repair errors: it trims, collapses whitespace and drops sentence
//! punctuation, and stops there. It never strips hyphens, folds umlauts,
//! decomposes compounds or equates synonyms — in German medical vocabulary
//! those are exactly the differences that decide whether a term was recognised.

/// Punctuation removed before comparison. Sentence marks only; `-` and `/` stay
/// because they are part of terms ("AV-Knoten-Reentrytachykardie",
/// "Sacubitril/Valsartan").
const STRIPPED_PUNCTUATION: &[char] = &['.', ',', ';', ':', '!', '?', '"', '„', '“', '(', ')'];

/// Collapse whitespace and drop sentence punctuation, preserving everything
/// that distinguishes one medical term from another.
pub fn normalize(text: &str) -> String {
    text.split_whitespace()
        .map(|word| word.trim_matches(|c| STRIPPED_PUNCTUATION.contains(&c)))
        .filter(|word| !word.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

/// Normalised words, lowercased — the token stream WER is computed over.
fn tokens(text: &str) -> Vec<String> {
    normalize(text)
        .split_whitespace()
        .map(|w| w.to_lowercase())
        .collect()
}

/// Word error rate: (substitutions + deletions + insertions) / reference words,
/// via Levenshtein distance over word tokens.
///
/// An empty reference yields 0.0 for an empty hypothesis and 1.0 otherwise —
/// there is no meaningful rate to divide, and reporting 0 for a hallucinated
/// sentence would flatter the model.
pub fn word_error_rate(reference: &str, hypothesis: &str) -> f64 {
    let reference = tokens(reference);
    let hypothesis = tokens(hypothesis);
    if reference.is_empty() {
        return if hypothesis.is_empty() { 0.0 } else { 1.0 };
    }
    edit_distance(&reference, &hypothesis) as f64 / reference.len() as f64
}

/// Levenshtein distance over whole words, two rows at a time.
fn edit_distance(a: &[String], b: &[String]) -> usize {
    let mut previous: Vec<usize> = (0..=b.len()).collect();
    let mut current = vec![0usize; b.len() + 1];
    for (i, a_word) in a.iter().enumerate() {
        current[0] = i + 1;
        for (j, b_word) in b.iter().enumerate() {
            let substitution = previous[j] + usize::from(a_word != b_word);
            let deletion = previous[j + 1] + 1;
            let insertion = current[j] + 1;
            current[j + 1] = substitution.min(deletion).min(insertion);
        }
        std::mem::swap(&mut previous, &mut current);
    }
    previous[b.len()]
}

/// Whether `term` appears in `text` as a whole token sequence.
///
/// Matching is case-insensitive but otherwise literal, and it is anchored to
/// token boundaries so "Aorta" does not count as a hit inside
/// "Aortenklappenstenose" — for a benchmark about compound medical terms, a
/// substring match would be self-deception.
pub fn contains_term(text: &str, term: &str) -> bool {
    let haystack = tokens(text);
    let needle = tokens(term);
    if needle.is_empty() || needle.len() > haystack.len() {
        return false;
    }
    haystack
        .windows(needle.len())
        .any(|w| w == needle.as_slice())
}

/// How many of `terms` survived into `text`.
pub fn terms_present(text: &str, terms: &[String]) -> usize {
    terms.iter().filter(|t| contains_term(text, t)).count()
}

/// Terms the model emitted that were never spoken — the bias-hallucination
/// signal. A negative term counts once whether it appears once or twice: the
/// question is whether the context leaked into the output at all.
pub fn false_bias_insertions(text: &str, negative_terms: &[String]) -> usize {
    terms_present(text, negative_terms)
}

/// Vocabulary terms that appear in `text` but were not spoken.
///
/// This is the general hallucination signal: unlike `false_bias_insertions` it
/// does not depend on someone having listed the right `negative_terms` — every
/// term of the active pool is checked. Matching here is deliberately looser
/// than for recognition: hyphens and slashes count as word breaks, so an
/// invented "Sinus-Transversus pericardii" is caught as "Sinus transversus
/// pericardii", and "EKG" counts as spoken when "Langzeit-EKG" was. Capital
/// abbreviations must match in case ("des" is an article, not "DES"). A term
/// counts as spoken when the reference contains it, or (single-word terms)
/// an inflected form: one word the other plus at most three letters
/// ("Stent"/"Stents"). The returned names let a reader check every hit.
pub fn unspoken_vocabulary(text: &str, reference: &str, vocabulary: &[String]) -> Vec<String> {
    let haystack = loose_tokens(text);
    let spoken = loose_tokens(reference);
    let lower = |words: &[String]| words.iter().map(|w| w.to_lowercase()).collect::<Vec<_>>();
    let (haystack_lower, spoken_lower) = (lower(&haystack), lower(&spoken));
    let mut found = Vec::new();
    for term in vocabulary {
        let needle = loose_tokens(term);
        if needle.is_empty() || needle.len() > haystack.len() {
            continue;
        }
        let abbreviation = is_abbreviation(term);
        let (text_words, reference_words, needle) = if abbreviation {
            (&haystack, &spoken, needle)
        } else {
            (&haystack_lower, &spoken_lower, lower(&needle))
        };
        if !text_words
            .windows(needle.len())
            .any(|w| w == needle.as_slice())
        {
            continue;
        }
        let in_reference = reference_words
            .windows(needle.len())
            .any(|w| w == needle.as_slice())
            || (needle.len() == 1
                && !abbreviation
                && reference_words
                    .iter()
                    .any(|word| inflected(word, &needle[0])));
        if !in_reference && !found.contains(term) {
            found.push(term.clone());
        }
    }
    // One invention, one count: "Valsartan" inside an invented
    // "Sacubitril/Valsartan" is the same error, not a second one.
    let words: Vec<Vec<String>> = found
        .iter()
        .map(|t| loose_tokens(t).iter().map(|w| w.to_lowercase()).collect())
        .collect();
    found
        .iter()
        .enumerate()
        .filter(|(i, _)| {
            !words.iter().enumerate().any(|(j, longer)| {
                j != *i
                    && longer.len() > words[*i].len()
                    && longer
                        .windows(words[*i].len())
                        .any(|w| w == words[*i].as_slice())
            })
        })
        .map(|(_, term)| term.clone())
        .collect()
}

/// Words split at whitespace, hyphens and slashes, sentence marks dropped,
/// case kept.
fn loose_tokens(text: &str) -> Vec<String> {
    normalize(text)
        .split(|c: char| c.is_whitespace() || c == '-' || c == '/')
        .filter(|w| !w.is_empty())
        .map(str::to_string)
        .collect()
}

/// "DES", "NSTEMI", "LVEF" — at least two letters, all of them capitals.
fn is_abbreviation(term: &str) -> bool {
    let letters: Vec<char> = term.chars().filter(|c| c.is_alphabetic()).collect();
    letters.len() >= 2 && letters.iter().all(|c| c.is_uppercase())
}

/// One word is the other plus at most three trailing letters.
fn inflected(a: &str, b: &str) -> bool {
    let (short, long) = if a.chars().count() <= b.chars().count() {
        (a, b)
    } else {
        (b, a)
    };
    long.starts_with(short) && long.chars().count() - short.chars().count() <= 3
}

/// Letters and digits only, lowercased — for telling a pure reformatting
/// ("st-hebungsinfarkt" → "ST-Hebungsinfarkt") from a changed word.
pub fn folded(text: &str) -> String {
    text.chars()
        .filter(|c| c.is_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

/// Exact match after normalisation, case-insensitively.
pub fn exact_match(reference: &str, hypothesis: &str) -> bool {
    tokens(reference) == tokens(hypothesis)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strings(values: &[&str]) -> Vec<String> {
        values.iter().map(|v| v.to_string()).collect()
    }

    #[test]
    fn unspoken_vocabulary_flags_inventions_but_not_inflections() {
        let pool = strings(&[
            "Sacubitril/Valsartan",
            "Valsartan",
            "Atorvastatin",
            "Stent",
            "Perikarderguss",
            "Vorhofflimmern",
        ]);
        let reference = "Atorvastatin 40 mg, zwei Stents, kein Perikarderguss";
        // Substituted drug: invented. Inflected and spoken terms: not.
        let output = "Sacubitril/Valsartan 40 mg, zwei Stents, kein Perikarderguss";
        assert_eq!(
            unspoken_vocabulary(output, reference, &pool),
            strings(&["Sacubitril/Valsartan"])
        );
        // A term that appears nowhere in the output is not reported.
        assert!(unspoken_vocabulary(reference, reference, &pool).is_empty());
        // Hyphenation neither hides an invention nor fakes one.
        let pool = strings(&["Sinus transversus pericardii", "EKG", "DES"]);
        assert_eq!(
            unspoken_vocabulary(
                "Im EKG Sinus-Transversus pericardii",
                "Im EKG Sinusrhythmus",
                &pool
            ),
            strings(&["Sinus transversus pericardii"])
        );
        assert!(unspoken_vocabulary("Im Langzeit EKG", "Im Langzeit-EKG", &pool).is_empty());
        // An article is not the abbreviation; the abbreviation is.
        assert!(unspoken_vocabulary("Pointe des Tachykardie", "Torsade", &pool).is_empty());
        assert_eq!(
            unspoken_vocabulary("ein DES implantiert", "ein Stent implantiert", &pool),
            strings(&["DES"])
        );
        let pool = strings(&[
            "Sacubitril/Valsartan",
            "Atorvastatin",
            "Stent",
            "Perikarderguss",
            "Vorhofflimmern",
        ]);
        // An unrelated appended finding is.
        assert_eq!(
            unspoken_vocabulary(
                "kein Perikarderguss. Vorhofflimmern",
                "kein Perikarderguss",
                &pool
            ),
            strings(&["Vorhofflimmern"])
        );
    }

    #[test]
    fn folding_ignores_case_and_punctuation_only() {
        assert_eq!(folded("ST-Hebungsinfarkt"), folded("st Hebungsinfarkt"));
        assert_ne!(folded("Funktion"), folded("Punktion"));
    }

    #[test]
    fn wer_counts_each_edit_operation() {
        // Identical.
        assert_eq!(word_error_rate("a b c", "a b c"), 0.0);
        // One substitution out of three words.
        assert!((word_error_rate("a b c", "a x c") - 1.0 / 3.0).abs() < 1e-9);
        // One deletion.
        assert!((word_error_rate("a b c", "a c") - 1.0 / 3.0).abs() < 1e-9);
        // One insertion.
        assert!((word_error_rate("a b c", "a b x c") - 1.0 / 3.0).abs() < 1e-9);
        // Nothing recognised at all.
        assert_eq!(word_error_rate("a b c", ""), 1.0);
        // Hypothesis longer than the reference can exceed 1.0.
        assert!(word_error_rate("a", "x y z") > 1.0);
    }

    #[test]
    fn wer_handles_an_empty_reference_honestly() {
        assert_eq!(word_error_rate("", ""), 0.0);
        // A hallucinated sentence against no reference is a total miss, not 0%.
        assert_eq!(word_error_rate("", "Aortenklappenstenose"), 1.0);
    }

    #[test]
    fn wer_ignores_punctuation_and_case_but_not_wording() {
        assert_eq!(
            word_error_rate(
                "Bei Aufnahme zeigte sich eine Aortenklappenstenose.",
                "bei aufnahme zeigte sich eine aortenklappenstenose"
            ),
            0.0
        );
        assert!(
            word_error_rate("eine Aortenklappenstenose", "eine Aortenstenose") > 0.0,
            "a different compound must count as an error"
        );
    }

    #[test]
    fn term_matching_is_anchored_to_word_boundaries() {
        let text = "Bei Aufnahme zeigte sich eine hochgradige Aortenklappenstenose.";
        assert!(contains_term(text, "Aortenklappenstenose"));
        assert!(
            contains_term(text, "aortenklappenstenose"),
            "case-insensitive"
        );
        // Substring of a compound must NOT count.
        assert!(!contains_term(text, "Aorta"));
        assert!(!contains_term(text, "Klappenstenose"));
    }

    #[test]
    fn term_matching_keeps_multiword_terms_together() {
        let text = "Es erfolgte eine transkatheter Aortenklappenimplantation heute.";
        assert!(contains_term(
            text,
            "transkatheter Aortenklappenimplantation"
        ));
        // The words must be adjacent and in order.
        assert!(!contains_term(
            "Aortenklappenimplantation ohne transkatheter Zugang",
            "transkatheter Aortenklappenimplantation"
        ));
    }

    #[test]
    fn term_matching_preserves_hyphens_and_slashes() {
        assert!(contains_term(
            "Verdacht auf AV-Knoten-Reentrytachykardie.",
            "AV-Knoten-Reentrytachykardie"
        ));
        // A differently written variant is a miss — we measure, we don't repair.
        assert!(!contains_term(
            "Verdacht auf AV Knoten Reentrytachykardie.",
            "AV-Knoten-Reentrytachykardie"
        ));
        assert!(contains_term(
            "Therapie mit Sacubitril/Valsartan begonnen.",
            "Sacubitril/Valsartan"
        ));
    }

    #[test]
    fn medical_term_accuracy_counts_hits() {
        let targets = strings(&["Echokardiographie", "linksventrikuläre Ejektionsfraktion"]);
        let both =
            "Die Echokardiographie ergab eine erhaltene linksventrikuläre Ejektionsfraktion.";
        assert_eq!(terms_present(both, &targets), 2);
        let one = "Die Echokardiographie war unauffällig.";
        assert_eq!(terms_present(one, &targets), 1);
        assert_eq!(terms_present("Kein Befund.", &targets), 0);
    }

    #[test]
    fn false_bias_detects_context_terms_that_were_never_spoken() {
        let negatives = strings(&["Tachykardie-Bradykardie-Syndrom", "Sacubitril/Valsartan"]);
        assert_eq!(
            false_bias_insertions("Die linksventrikuläre Funktion war erhalten.", &negatives),
            0
        );
        assert_eq!(
            false_bias_insertions(
                "Die Funktion war erhalten, Sacubitril/Valsartan angesetzt.",
                &negatives
            ),
            1
        );
        // A repeat is still one leaked term.
        assert_eq!(
            false_bias_insertions(
                "Sacubitril/Valsartan und nochmals Sacubitril/Valsartan.",
                &negatives
            ),
            1
        );
    }

    #[test]
    fn normalization_is_conservative() {
        assert_eq!(normalize("  viel   Raum \n hier. "), "viel Raum hier");
        // Hyphens, slashes and umlauts survive untouched.
        assert_eq!(
            normalize("AV-Knoten-Reentrytachykardie, Sacubitril/Valsartan."),
            "AV-Knoten-Reentrytachykardie Sacubitril/Valsartan"
        );
        assert_eq!(normalize("Ödem (links)"), "Ödem links");
    }

    #[test]
    fn exact_match_ignores_only_case_and_punctuation() {
        assert!(exact_match(
            "Die Funktion war erhalten.",
            "die funktion war erhalten"
        ));
        assert!(!exact_match(
            "Die Funktion war erhalten.",
            "Die Funktion war normal."
        ));
    }
}
