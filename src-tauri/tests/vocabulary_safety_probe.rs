//! Correct German text must survive the fuzzy post-correction untouched.
//!
//! The corrector's job is to repair misheard specialist terms. Its greater
//! risk is the opposite: replacing a word that was already right, because some
//! entry of the bundled vocabulary happens to look similar. That risk grows
//! with every dictionary module added to the bundle, so this probe measures it
//! against the **whole** bundle — every module at its largest tier, which is
//! the worst case a user can configure.
//!
//! The corpus is the recording package's reference texts (`diktate.jsonl`):
//! correct clinical German, written by hand, with drug names, doses,
//! negations and abbreviations. Nothing in them needs correcting, so any
//! change here is a false positive — the class of error that would put a
//! finding into a report that nobody dictated.

use handy_app_lib::audio_toolkit::correct_with_vocabulary;
use std::path::{Path, PathBuf};

/// The app's default; the probe is worthless at any other threshold.
const THRESHOLD: f64 = 0.18;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("src-tauri has a parent")
        .to_path_buf()
}

/// Every bundled term, all modules, largest tier — read from the files rather
/// than through the settings layer so the probe cannot be weakened by a tier
/// default changing.
fn full_bundle() -> Vec<String> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("dictionaries/de");
    let mut pool = Vec::new();
    for entry in std::fs::read_dir(&dir).expect("bundled dictionaries exist") {
        let raw = std::fs::read_to_string(entry.expect("readable entry").path()).expect("utf-8");
        pool.extend(
            raw.lines()
                .map(str::trim)
                .filter(|line| !line.is_empty() && !line.starts_with('#'))
                .map(String::from),
        );
    }
    pool
}

/// The `reference` field of every case in the recording package.
fn reference_sentences() -> Vec<String> {
    let path = repo_root().join("benchmarks/medical_asr/kardiologie_v1/diktate.jsonl");
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("could not read '{}': {}", path.display(), e));
    raw.lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            let value: serde_json::Value = serde_json::from_str(line).expect("valid JSONL");
            value["reference"]
                .as_str()
                .expect("every case has a reference")
                .to_string()
        })
        .collect()
}

#[test]
fn correct_dictations_survive_the_whole_bundle_unchanged() {
    let pool = full_bundle();
    assert!(
        pool.len() > 5_000,
        "expected the full bundle, got {} terms",
        pool.len()
    );
    let sentences = reference_sentences();
    assert!(
        sentences.len() >= 40,
        "expected the recording package, got {} sentences",
        sentences.len()
    );

    let mut changed = Vec::new();
    for sentence in &sentences {
        let report = correct_with_vocabulary(sentence, &pool, THRESHOLD);
        if &report.text != sentence {
            changed.push(format!(
                "\n  in:  {}\n  out: {}\n  via: {:?}",
                sentence,
                report.text,
                report
                    .events
                    .iter()
                    .map(|e| format!("{} → {}", e.original, e.term))
                    .collect::<Vec<_>>()
            ));
        }
    }
    assert!(
        changed.is_empty(),
        "fuzzy correction altered {} of {} already-correct dictations:{}",
        changed.len(),
        sentences.len(),
        changed.join("")
    );
}
