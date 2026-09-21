//! Benchmark dataset and run matrix.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// One benchmark utterance. See `benchmarks/medical_asr/README.md`.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TestCase {
    pub id: String,
    /// Audio path, relative to the dataset file's directory.
    pub audio: String,
    /// What was actually said.
    pub reference: String,
    pub specialty: String,
    /// Medical terms that were spoken and must survive into the transcript.
    #[serde(default)]
    pub target_terms: Vec<String>,
    /// Terms that were **not** spoken but may sit in the active context. Any of
    /// these appearing in the output is a bias hallucination.
    #[serde(default)]
    pub negative_terms: Vec<String>,
    #[serde(default = "default_difficulty")]
    pub difficulty: String,
    /// Drug names that were spoken. Scored as their own group because a wrong
    /// drug is a different and more dangerous error than a wrong adjective.
    #[serde(default)]
    pub medications: Vec<String>,
    /// Negated phrases that were spoken and must survive verbatim
    /// ("kein Perikarderguss"). Losing one inverts the finding.
    #[serde(default)]
    pub negations: Vec<String>,
    /// Numbers and doses that were spoken and must survive verbatim
    /// ("2,5 mg", "135/85 mmHg").
    #[serde(default)]
    pub numbers: Vec<String>,
    /// Similar-sounding terms that were **not** spoken — the one the speaker
    /// meant could be mistaken for these ("Mitralklappeninsuffizienz" when
    /// "Mitralklappenstenose" was said). Any appearance is a confusion error.
    #[serde(default)]
    pub confusable_terms: Vec<String>,
}

fn default_difficulty() -> String {
    "normal".to_string()
}

impl TestCase {
    pub fn audio_path(&self, dataset_dir: &Path) -> PathBuf {
        dataset_dir.join(&self.audio)
    }
}

/// Which dictionary modules and personal words the run is scored with. Mirrors
/// the fields of the app's own settings so the harness can hand them straight
/// to the production dictionary code.
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct BenchmarkConfig {
    #[serde(default)]
    pub active_dictionaries: Vec<String>,
    #[serde(default)]
    pub dictionary_levels: HashMap<String, u32>,
    #[serde(default)]
    pub custom_words: Vec<String>,
    /// Dictionaries allowed to offer context, as the settings page stores them.
    /// Absent means "the same ones", so an old config keeps measuring what it
    /// measured before; an empty list measures post-correction on its own.
    #[serde(default)]
    pub context_dictionaries: Option<Vec<String>>,
    /// Fuzzy threshold override. Absent means the app's own default, so a
    /// benchmark run scores the correction the app actually performs.
    #[serde(default)]
    pub word_correction_threshold: Option<f64>,
}

/// What the model is given, and what fuzzy correction is given afterwards.
///
/// The pairing is the whole point of the experiment: A and B isolate the two
/// mechanisms, while C and D differ *only* in whether terms already sent as
/// context are still available to the corrector.
///
/// **D is what the app does** since the per-dictionary usage setting landed.
/// C stays in the harness because that choice was made on synthetic audio
/// only: it is the comparison that human recordings still have to settle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
pub enum Mode {
    /// No context, no correction — the model on its own.
    A,
    /// No context; the full vocabulary pool corrects the output.
    B,
    /// Context biasing, and correction over the pool *minus* the context.
    /// This is what the app does today.
    C,
    /// Context biasing, and correction over the entire pool.
    D,
}

impl Mode {
    pub const ALL: [Mode; 4] = [Mode::A, Mode::B, Mode::C, Mode::D];

    pub fn as_str(self) -> &'static str {
        match self {
            Mode::A => "A",
            Mode::B => "B",
            Mode::C => "C",
            Mode::D => "D",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Mode::A => "Baseline (no context, no fuzzy)",
            Mode::B => "Fuzzy only",
            Mode::C => "Context + fuzzy over the rest",
            Mode::D => "Context + fuzzy over everything (what the app does)",
        }
    }

    /// Whether this mode sends a recognition context at all.
    pub fn uses_context(self) -> bool {
        matches!(self, Mode::C | Mode::D)
    }

    pub fn parse(value: &str) -> Option<Mode> {
        match value.trim().to_ascii_uppercase().as_str() {
            "A" => Some(Mode::A),
            "B" => Some(Mode::B),
            "C" => Some(Mode::C),
            "D" => Some(Mode::D),
            _ => None,
        }
    }
}

/// Read a JSONL dataset, one [`TestCase`] per non-empty line.
pub fn load_dataset(path: &Path) -> anyhow::Result<Vec<TestCase>> {
    let raw = std::fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("could not read dataset '{}': {}", path.display(), e))?;
    let mut cases = Vec::new();
    for (index, line) in raw.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let case: TestCase = serde_json::from_str(line).map_err(|e| {
            anyhow::anyhow!("{}:{}: invalid test case: {}", path.display(), index + 1, e)
        })?;
        cases.push(case);
    }
    if cases.is_empty() {
        anyhow::bail!("dataset '{}' contains no test cases", path.display());
    }
    Ok(cases)
}

pub fn load_config(path: &Path) -> anyhow::Result<BenchmarkConfig> {
    let raw = std::fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("could not read config '{}': {}", path.display(), e))?;
    serde_json::from_str(&raw)
        .map_err(|e| anyhow::anyhow!("invalid config '{}': {}", path.display(), e))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn modes_are_distinguished_by_what_each_stage_receives() {
        assert!(!Mode::A.uses_context());
        assert!(!Mode::B.uses_context());
        assert!(Mode::C.uses_context());
        assert!(Mode::D.uses_context());
        assert_eq!(Mode::parse("c"), Some(Mode::C));
        assert_eq!(Mode::parse("x"), None);
        assert_eq!(Mode::ALL.len(), 4);
    }

    #[test]
    fn dataset_parsing_accepts_the_documented_shape() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("d.jsonl");
        std::fs::write(
            &path,
            concat!(
                "# a comment line is skipped\n",
                "\n",
                r#"{"id":"a1","audio":"audio/a1.wav","reference":"Eine Aortenklappenstenose.","specialty":"internal_cardiology","target_terms":["Aortenklappenstenose"],"negative_terms":[],"difficulty":"normal"}"#,
                "\n",
                r#"{"id":"a2","audio":"audio/a2.wav","reference":"Kein Befund.","specialty":"internal_cardiology"}"#,
                "\n",
            ),
        )
        .unwrap();
        let cases = load_dataset(&path).unwrap();
        assert_eq!(cases.len(), 2);
        // Error-group fields are optional.
        assert!(cases[0].medications.is_empty() && cases[0].negations.is_empty());
        assert_eq!(cases[0].target_terms, vec!["Aortenklappenstenose"]);
        // Optional fields default rather than failing the load.
        assert!(cases[1].target_terms.is_empty());
        assert_eq!(cases[1].difficulty, "normal");
        assert_eq!(
            cases[0].audio_path(dir.path()),
            dir.path().join("audio/a1.wav")
        );
    }

    #[test]
    fn an_empty_or_malformed_dataset_is_rejected_with_its_line_number() {
        let dir = tempfile::tempdir().unwrap();
        let empty = dir.path().join("empty.jsonl");
        std::fs::write(&empty, "\n# nothing\n").unwrap();
        assert!(load_dataset(&empty).is_err());

        let bad = dir.path().join("bad.jsonl");
        std::fs::write(&bad, "{\"id\":\"x\"}\n").unwrap();
        let message = load_dataset(&bad).unwrap_err().to_string();
        assert!(message.contains(":1:"), "message was: {}", message);
    }
}
