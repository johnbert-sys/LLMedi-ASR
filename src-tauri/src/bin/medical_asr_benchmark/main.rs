//! Medical ASR benchmark harness.
//!
//! A harness *around* llmedi's real transcription pipeline, not a second
//! implementation of it: dictionary selection, context budgeting, context
//! rendering and fuzzy correction all come from the app's own code (see
//! `runner.rs`). What this binary adds is the experiment — running the same
//! audio through several modes and context budgets and scoring the results.
//!
//! ```text
//! cargo run --release --bin medical-asr-benchmark -- \
//!     --model ~/path/to/Qwen3-ASR-1.7B-Q5_K_M.gguf \
//!     --dataset benchmarks/medical_asr/dataset.jsonl \
//!     --config  benchmarks/medical_asr/config.json
//! ```
//!
//! See `benchmarks/medical_asr/README.md`.

mod dataset;
mod metrics;
mod report;
mod runner;

use anyhow::{Context, Result};
use clap::Parser;
use dataset::{BenchmarkConfig, Mode};
use handy_app_lib::settings::{get_default_settings, AppSettings};
use report::ResultRow;
use std::path::{Path, PathBuf};

#[derive(Parser, Debug)]
#[command(
    name = "medical-asr-benchmark",
    about = "Measure medical ASR context biasing and fuzzy correction against a fixed dataset"
)]
struct Args {
    /// GGUF model to benchmark. The architecture and its context capability are
    /// read from the model itself — nothing is assumed from the file name.
    #[arg(long, required_unless_present = "rescore")]
    model: Option<PathBuf>,

    /// Score an existing results JSON again with the current metrics instead
    /// of transcribing. Writes `<name>-rescored.*` next to it.
    #[arg(long)]
    rescore: Option<PathBuf>,

    /// JSONL dataset. Audio paths inside it are relative to its directory.
    #[arg(long, default_value = "benchmarks/medical_asr/dataset.jsonl")]
    dataset: PathBuf,

    /// Which dictionary modules, tiers and personal words to score with.
    #[arg(long, default_value = "benchmarks/medical_asr/config.json")]
    config: PathBuf,

    /// Modes to run, comma separated (A, B, C, D).
    #[arg(long, default_value = "A,B,C,D", value_delimiter = ',')]
    modes: Vec<String>,

    /// Context budgets to sweep, comma separated. Only the context modes (C, D)
    /// use them; 0 means no recognition context.
    #[arg(long, default_value = "0,30,60,120,240,400", value_delimiter = ',')]
    context_budgets: Vec<usize>,

    /// Where results/*.json, *.csv and the report land.
    #[arg(long, default_value = "benchmarks/medical_asr/results")]
    out_dir: PathBuf,

    /// Score only this specialty.
    #[arg(long)]
    specialty: Option<String>,

    /// Score only this difficulty class.
    #[arg(long)]
    difficulty: Option<String>,
}

/// Turn the benchmark config into the very `AppSettings` the app's dictionary
/// code expects, starting from the app's own defaults so every unrelated field
/// keeps its production value.
fn settings_from(config: &BenchmarkConfig) -> AppSettings {
    let mut settings = get_default_settings();
    settings.active_dictionaries = config.active_dictionaries.clone();
    settings.dictionary_levels = config.dictionary_levels.clone();
    // Post-correction takes the whole list of every active dictionary; context
    // selection takes the listed ones, or the same set when none are named.
    settings.context_dictionaries = config
        .context_dictionaries
        .clone()
        .unwrap_or_else(|| config.active_dictionaries.clone());
    settings.custom_words = config.custom_words.clone();
    if let Some(threshold) = config.word_correction_threshold {
        settings.word_correction_threshold = threshold;
    }
    settings
}

/// Fill every field that follows from the reference and the two transcripts
/// alone. Shared by live runs and `--rescore`, so a stored result scored again
/// with improved metrics is scored exactly like a fresh one.
fn score_transcripts(row: &mut ResultRow, case: &dataset::TestCase, pool: &[String]) {
    let raw = row.raw_transcript.as_str();
    let final_text = row.final_transcript.as_str();
    row.specialty = case.specialty.clone();
    row.difficulty = case.difficulty.clone();
    row.reference = case.reference.clone();
    row.wer_raw = metrics::word_error_rate(&case.reference, raw);
    row.wer_final = metrics::word_error_rate(&case.reference, final_text);
    row.target_terms_total = case.target_terms.len();
    row.target_terms_correct_raw = metrics::terms_present(raw, &case.target_terms);
    row.target_terms_correct_final = metrics::terms_present(final_text, &case.target_terms);
    row.false_bias_insertions = metrics::false_bias_insertions(final_text, &case.negative_terms);
    row.exact_match_final = metrics::exact_match(&case.reference, final_text);
    row.fuzzy_worsened = row.wer_final > row.wer_raw;
    row.medications_total = case.medications.len();
    row.medications_correct_raw = metrics::terms_present(raw, &case.medications);
    row.medications_correct_final = metrics::terms_present(final_text, &case.medications);
    row.negations_total = case.negations.len();
    row.negations_kept_raw = metrics::terms_present(raw, &case.negations);
    row.negations_kept_final = metrics::terms_present(final_text, &case.negations);
    row.numbers_total = case.numbers.len();
    row.numbers_kept_raw = metrics::terms_present(raw, &case.numbers);
    row.numbers_kept_final = metrics::terms_present(final_text, &case.numbers);
    row.confusable_insertions_raw = metrics::terms_present(raw, &case.confusable_terms);
    row.confusable_insertions_final = metrics::terms_present(final_text, &case.confusable_terms);
    row.unspoken_terms_raw = metrics::unspoken_vocabulary(raw, &case.reference, pool);
    row.unspoken_terms_final = metrics::unspoken_vocabulary(final_text, &case.reference, pool);
}

/// Write the JSON, CSV and readable report for `rows` under `stem`.
fn write_results(out_dir: &Path, stem: &str, label: &str, rows: &[ResultRow]) -> Result<()> {
    std::fs::create_dir_all(out_dir)
        .with_context(|| format!("could not create '{}'", out_dir.display()))?;
    // Built by appending, not via `with_extension`: model names carry dots
    // ("Qwen3-ASR-1.7B-Q5_K_M"), and `with_extension` would replace everything
    // after the last one, writing results as "Qwen3-ASR-1.json".
    let path_for = |extension: &str| out_dir.join(format!("{}.{}", stem, extension));

    let json_path = path_for("json");
    std::fs::write(&json_path, serde_json::to_string_pretty(rows)?)?;
    let csv_path = path_for("csv");
    std::fs::write(&csv_path, report::to_csv(rows))?;
    let rendered = report::render_report(label, rows);
    let report_path = path_for("txt");
    std::fs::write(&report_path, &rendered)?;

    println!("\n{}", rendered);
    println!("JSON:   {}", json_path.display());
    println!("CSV:    {}", csv_path.display());
    println!("Report: {}", report_path.display());
    Ok(())
}

/// Score a stored result file again — same transcripts, current metrics.
/// Model-side fields (latency, tokens, fuzzy events) are kept as recorded.
fn rescore(path: &Path, cases: &[dataset::TestCase], pool: &[String]) -> Result<()> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("could not read '{}'", path.display()))?;
    let mut rows: Vec<ResultRow> = serde_json::from_str(&text)
        .with_context(|| format!("'{}' is not a result file", path.display()))?;
    for row in &mut rows {
        let case = cases
            .iter()
            .find(|c| c.id == row.test_id)
            .ok_or_else(|| anyhow::anyhow!("case '{}' is not in the dataset", row.test_id))?;
        score_transcripts(row, case, pool);
    }
    let first = rows
        .first()
        .ok_or_else(|| anyhow::anyhow!("'{}' holds no rows", path.display()))?;
    let label = format!(
        "{} ({}, {}) — rescored",
        first.model, first.model_arch, first.quantization
    );
    let stem = format!(
        "{}-rescored",
        path.file_stem().unwrap_or_default().to_string_lossy()
    );
    let out_dir = path.parent().unwrap_or(Path::new("."));
    write_results(out_dir, &stem, &label, &rows)
}

/// The (mode, budget) pairs to run.
///
/// A and B ignore the budget entirely, so they run once rather than once per
/// budget — repeating identical work would only add runtime and duplicate rows.
fn run_matrix(modes: &[Mode], budgets: &[usize]) -> Vec<(Mode, usize)> {
    let mut matrix = Vec::new();
    for &mode in modes {
        if mode.uses_context() {
            for &budget in budgets {
                matrix.push((mode, budget));
            }
        } else {
            matrix.push((mode, 0));
        }
    }
    matrix
}

fn main() -> Result<()> {
    let args = Args::parse();

    let dataset_dir = args
        .dataset
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let cases = dataset::load_dataset(&args.dataset)?;
    let config = dataset::load_config(&args.config)?;
    let settings = settings_from(&config);
    // The whole active pool, for spotting any dictionary term that turns up
    // unspoken — whether the model or the corrector put it there.
    let pool = handy_app_lib::dictionaries::full_vocabulary(&settings);

    if let Some(path) = &args.rescore {
        return rescore(path, &cases, &pool);
    }

    let cases: Vec<_> = cases
        .into_iter()
        .filter(|c| args.specialty.as_ref().is_none_or(|s| &c.specialty == s))
        .filter(|c| args.difficulty.as_ref().is_none_or(|d| &c.difficulty == d))
        .collect();
    if cases.is_empty() {
        anyhow::bail!("no test cases left after filtering");
    }

    let mut modes = Vec::new();
    for raw in &args.modes {
        if raw.trim().eq_ignore_ascii_case("all") {
            modes.extend(Mode::ALL);
            continue;
        }
        modes.push(
            Mode::parse(raw)
                .ok_or_else(|| anyhow::anyhow!("unknown mode '{}' (use A/B/C/D or 'all')", raw))?,
        );
    }
    modes.dedup();

    let model_path = args.model.as_ref().expect("clap requires --model");
    println!("Loading {} …", model_path.display());
    let mut model = runner::LoadedModel::load(model_path)?;
    println!(
        "  arch={}  recognition context={}  quantization={}",
        model.arch,
        if model.supports_recognition_context {
            "supported"
        } else {
            "NOT supported — context modes will send nothing"
        },
        model.quantization
    );

    let matrix = run_matrix(&modes, &args.context_budgets);
    let total = matrix.len() * cases.len();
    println!("Running {} transcriptions …\n", total);

    let mut rows: Vec<ResultRow> = Vec::with_capacity(total);
    let mut done = 0usize;
    for (mode, budget) in matrix {
        for case in &cases {
            done += 1;
            let outcome =
                match runner::run_case(&mut model, &settings, case, &dataset_dir, mode, budget) {
                    Ok(outcome) => outcome,
                    Err(error) => {
                        // Only unreadable audio lands here — transcription
                        // failures are rows. One bad file must not throw away
                        // a long run.
                        eprintln!("  [{}/{}] {} skipped: {}", done, total, case.id, error);
                        continue;
                    }
                };

            let raw = &outcome.raw_transcript;
            let final_text = &outcome.final_transcript;
            let replaced: Vec<_> = outcome
                .correction
                .events
                .iter()
                .filter(|e| e.kind == handy_app_lib::audio_toolkit::CorrectionKind::Replaced)
                .collect();
            let mut row = ResultRow {
                test_id: case.id.clone(),
                model: model.name.clone(),
                model_path: model.path.clone(),
                model_arch: model.arch.clone(),
                quantization: model.quantization.clone(),
                mode: mode.as_str().to_string(),
                context_budget: budget,
                raw_transcript: raw.clone(),
                final_transcript: final_text.clone(),
                context: outcome.context.clone().unwrap_or_default(),
                context_terms: outcome.context_terms.len(),
                fuzzy_terms: outcome.fuzzy_terms.len(),
                targets_in_context: runner::targets_in_context(case, &outcome.context_terms),
                targets_in_context_correct: runner::targets_in_context_recognized(
                    case,
                    &outcome.context_terms,
                    final_text,
                ),
                selected_terms: outcome.selected_terms,
                deferred_terms: outcome.fit.deferred.len(),
                context_tokens: outcome.fit.context_tokens,
                context_room: outcome.fit.context_room,
                context_verified: outcome.fit.verified(),
                fuzzy_replacements: replaced.len(),
                fuzzy_ambiguous_skips: outcome.correction.ambiguous_spans.len(),
                // A replacement that only re-cased or re-hyphenated a word
                // ("st-hebungsinfarkt" → "ST-Hebungsinfarkt") introduced nothing.
                fuzzy_introduced_terms: replaced
                    .iter()
                    .filter(|e| metrics::folded(&e.original) != metrics::folded(&e.term))
                    .filter(|e| !metrics::contains_term(&case.reference, &e.term))
                    .count(),
                derived_kv_mib: outcome
                    .derived_kv_bytes
                    .map(|b| b as f64 / (1024.0 * 1024.0)),
                audio_seconds: outcome.audio_seconds,
                transcribe_ms: outcome.transcribe_ms,
                fuzzy_ms: outcome.fuzzy_ms,
                latency_ms: outcome.total_ms(),
                fell_back_without_context: outcome.fell_back_without_context,
                run_error: outcome.error.clone().unwrap_or_default(),
                ..Default::default()
            };
            score_transcripts(&mut row, case, &pool);
            rows.push(row);
            let last = rows.last().expect("row just pushed");
            let note = if !last.run_error.is_empty() {
                format!("  FAILED: {}", last.run_error)
            } else if last.fell_back_without_context {
                "  (fallback without context)".to_string()
            } else {
                String::new()
            };
            println!(
                "  [{}/{}] {} mode {} budget {:>3} — WER {:.1}% → {:.1}%{}",
                done,
                total,
                case.id,
                mode.as_str(),
                budget,
                last.wer_raw * 100.0,
                last.wer_final * 100.0,
                note
            );
        }
    }

    if rows.is_empty() {
        anyhow::bail!("every test case failed; nothing to report");
    }

    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or_default();
    let stem = format!("{}-{}", model.name, stamp);
    let label = format!("{} ({}, {})", model.name, model.arch, model.quantization);
    write_results(&args.out_dir, &stem, &label, &rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_free_modes_run_once_while_context_modes_sweep_budgets() {
        let matrix = run_matrix(&Mode::ALL, &[0, 60, 120]);
        let a = matrix.iter().filter(|(m, _)| *m == Mode::A).count();
        let b = matrix.iter().filter(|(m, _)| *m == Mode::B).count();
        let c: Vec<usize> = matrix
            .iter()
            .filter(|(m, _)| *m == Mode::C)
            .map(|(_, budget)| *budget)
            .collect();
        // A and B ignore the budget, so repeating them would be wasted runtime.
        assert_eq!(a, 1);
        assert_eq!(b, 1);
        assert_eq!(c, vec![0, 60, 120]);
        assert_eq!(matrix.len(), 1 + 1 + 3 + 3);
    }

    #[test]
    fn config_maps_onto_the_apps_own_settings_type() {
        let config = BenchmarkConfig {
            active_dictionaries: vec!["internal_cardiology".into()],
            dictionary_levels: [("internal_cardiology".to_string(), 250u32)]
                .into_iter()
                .collect(),
            custom_words: vec!["Blatt-Schmidt-Zeichen".into()],
            // Active for post-correction, but kept out of the context.
            context_dictionaries: Some(Vec::new()),
            word_correction_threshold: Some(0.75),
        };
        let settings = settings_from(&config);
        assert_eq!(settings.active_dictionaries, config.active_dictionaries);
        assert_eq!(
            settings.dictionary_levels.get("internal_cardiology"),
            Some(&250)
        );
        assert_eq!(settings.custom_words, config.custom_words);
        // A dictionary kept out of the context in the config is kept out of it
        // in the run, so the harness measures the selection the app would make.
        assert!(settings.context_dictionaries.is_empty());
        // Personal words are not part of any dictionary, so they stay; the
        // restricted dictionary contributes nothing.
        assert_eq!(
            handy_app_lib::dictionaries::context_vocabulary(&settings, 120),
            config.custom_words
        );
        assert!(handy_app_lib::dictionaries::full_vocabulary(&settings).len() > 100);
        assert!((settings.word_correction_threshold - 0.75).abs() < 1e-9);
        // Without an override the app's own default applies.
        let plain = settings_from(&BenchmarkConfig::default());
        assert_eq!(
            plain.word_correction_threshold,
            get_default_settings().word_correction_threshold
        );
        // Unrelated fields keep their production defaults.
        assert_eq!(
            settings.selected_language,
            get_default_settings().selected_language
        );
    }
}
