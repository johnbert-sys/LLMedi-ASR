//! Drives one dataset case through the **production** ASR path.
//!
//! Nothing here re-implements selection, formatting or correction. The harness
//! only decides *which* vocabulary each stage gets (that is what a mode is) and
//! then calls the same functions the app calls:
//!
//! - [`handy_app_lib::dictionaries::full_vocabulary`] / `context_vocabulary` /
//!   `fuzzy_only_vocabulary` — module round-robin, tiers, deduplication
//! - [`handy_app_lib::transcribe_cpp_context_biasing`] — per-architecture
//!   capability and budget
//! - [`handy_app_lib::fit_context`] — fitting the selection into the model's
//!   real token window, deferring what does not fit to fuzzy correction
//! - [`handy_app_lib::apply_context_to_run_options`] — the one place that knows
//!   which `RunOptions` slot a context goes into
//! - [`handy_app_lib::audio_toolkit::correct_with_vocabulary`] — fuzzy
//!   correction, with its audit trail
//!
//! Deliberately *not* covered: filler-word removal and output normalisation
//! from `post_process_transcription_text`. Those run identically in every mode,
//! so including them would add noise without separating any of the variables
//! under test.

use anyhow::{Context, Result};
use handy_app_lib::audio_toolkit::{correct_with_vocabulary, read_wav_samples, CorrectionReport};
use handy_app_lib::dictionaries::{context_vocabulary, full_vocabulary, fuzzy_only_vocabulary};
use handy_app_lib::settings::AppSettings;
use handy_app_lib::{
    apply_context_to_run_options, fit_context, run_with_context_fallback,
    transcribe_cpp_context_biasing, ContextBiasing, ContextTokenLimit, WindowFit,
};
use std::path::Path;
use std::time::Instant;
use transcribe_cpp::{Feature, Model, RunOptions, Session};

use crate::dataset::{Mode, TestCase};
use crate::metrics;

/// Sample rate the transcription pipeline assumes throughout.
const EXPECTED_SAMPLE_RATE: u32 = 16_000;

/// Everything one (case × mode × budget) run produced.
pub struct RunOutcome {
    pub raw_transcript: String,
    pub final_transcript: String,
    /// The context string actually handed to the model, if any.
    pub context: Option<String>,
    /// Terms the term budget selected, before token fitting.
    pub selected_terms: usize,
    /// Terms actually sent (after fitting).
    pub context_terms: Vec<String>,
    pub fuzzy_terms: Vec<String>,
    /// The token-fitting decision, exactly as the app makes it.
    pub fit: WindowFit,
    /// Every replacement fuzzy correction made or declined.
    pub correction: CorrectionReport,
    /// Decoder KV cache this run needs, derived from the library's allocation
    /// rule (power-of-two ≥ prompt + output, from 1024) — not measured.
    pub derived_kv_bytes: Option<u64>,
    pub audio_seconds: f64,
    pub transcribe_ms: f64,
    pub fuzzy_ms: f64,
    /// The run with context failed and was repeated without it — exactly what
    /// the app does, so a dictation is not lost to its own context.
    pub fell_back_without_context: bool,
    /// Transcription failed even without context: in the app this dictation
    /// would have produced no text. Transcripts are empty then.
    pub error: Option<String>,
}

impl RunOutcome {
    pub fn total_ms(&self) -> f64 {
        self.transcribe_ms + self.fuzzy_ms
    }
    // Real-time factor is derived in `report`, from the stored latency and
    // audio duration, so every consumer of a result row computes it the same
    // way rather than from a second source here.
}

/// A loaded model plus the facts the production capability probe needs.
pub struct LoadedModel {
    pub session: Session,
    pub arch: String,
    pub supports_recognition_context: bool,
    pub path: String,
    pub name: String,
    pub quantization: String,
}

impl LoadedModel {
    pub fn load(path: &Path) -> Result<Self> {
        let model = Model::load(path)
            .map_err(|e| anyhow::anyhow!("could not load '{}': {}", path.display(), e))?;
        let arch = model.arch();
        let supports_recognition_context = model.supports(Feature::Context);
        let name = path
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "unknown".to_string());
        // The GGUF file name is the only quantization hint available without
        // reaching past the safe API; recorded as-is rather than guessed at.
        let quantization = name
            .rsplit('-')
            .next()
            .filter(|q| q.starts_with('Q') || q.starts_with('F') || q.starts_with('B'))
            .unwrap_or("unknown")
            .to_string();
        let session = model
            .session()
            .map_err(|e| anyhow::anyhow!("could not open a session: {}", e))?;
        Ok(LoadedModel {
            session,
            arch,
            supports_recognition_context,
            path: path.display().to_string(),
            name,
            quantization,
        })
    }

    /// The production capability for this model at a given budget.
    pub fn biasing(&self, budget: usize) -> ContextBiasing {
        let probed = transcribe_cpp_context_biasing(&self.arch, self.supports_recognition_context);
        // Same channel and support as production; only the budget is the
        // variable under test, so it is overridden rather than re-derived.
        ContextBiasing {
            channel: probed.channel,
            budget,
        }
    }
}

/// Load benchmark audio as the 16 kHz mono f32 the engine expects.
pub fn load_audio(path: &Path) -> Result<(Vec<f32>, f64)> {
    let spec = hound::WavReader::open(path)
        .with_context(|| format!("could not open audio '{}'", path.display()))?
        .spec();
    if spec.sample_rate != EXPECTED_SAMPLE_RATE || spec.channels != 1 {
        anyhow::bail!(
            "'{}' is {} Hz / {} channel(s); the pipeline needs {} Hz mono — \
             convert it first (e.g. `afconvert -f WAVE -d LEI16@16000 -c 1 in.wav out.wav`)",
            path.display(),
            spec.sample_rate,
            spec.channels,
            EXPECTED_SAMPLE_RATE
        );
    }
    let samples = read_wav_samples(path)
        .map_err(|e| anyhow::anyhow!("could not read '{}': {}", path.display(), e))?;
    let seconds = samples.len() as f64 / EXPECTED_SAMPLE_RATE as f64;
    Ok((samples, seconds))
}

/// Terms the term budget selects as context for `mode`, before token fitting.
pub fn selected_for(settings: &AppSettings, mode: Mode, biasing: ContextBiasing) -> Vec<String> {
    if mode.uses_context() && biasing.supported() && biasing.budget > 0 {
        context_vocabulary(settings, biasing.budget)
    } else {
        Vec::new()
    }
}

/// Terms fuzzy correction receives for `mode`, given what was actually sent.
///
/// Mode C corrects only what the model did not see — which, after fitting,
/// includes every selected term that was deferred.
pub fn fuzzy_for(settings: &AppSettings, mode: Mode, sent: &[String]) -> Vec<String> {
    let full = full_vocabulary(settings);
    match mode {
        // The model alone: nothing corrected.
        Mode::A => Vec::new(),
        // The whole pool corrects the output.
        Mode::B | Mode::D => full,
        // Today's split: correct only what the model never saw.
        Mode::C => fuzzy_only_vocabulary(&full, sent),
    }
}

/// `(context, fuzzy)` for `mode` without token fitting — what the stages get
/// when every selected term fits.
#[cfg(test)]
pub fn vocabulary_for(
    settings: &AppSettings,
    mode: Mode,
    biasing: ContextBiasing,
) -> (Vec<String>, Vec<String>) {
    let context = selected_for(settings, mode, biasing);
    let fuzzy = fuzzy_for(settings, mode, &context);
    (context, fuzzy)
}

/// KV-cache bytes a run needs, following qwen3_asr's allocation rule: a
/// power-of-two token capacity, starting at 1024, at least prompt + output,
/// clamped to the window; bytes scale from the session's worst case.
fn derived_kv_bytes(limit: ContextTokenLimit, fit: &WindowFit, max_kv_bytes: i64) -> Option<u64> {
    let ContextTokenLimit::SharedWindow { n_ctx, .. } = limit else {
        return None;
    };
    if max_kv_bytes <= 0 || n_ctx == 0 {
        return None;
    }
    let prompt = handy_app_lib::QWEN_TEMPLATE_OVERHEAD
        + fit.context_tokens.unwrap_or(0)
        + fit.audio_tokens.unwrap_or(0);
    let need = prompt + handy_app_lib::QWEN_OUTPUT_RESERVE;
    let mut capacity = 1024usize;
    while capacity < need {
        capacity *= 2;
    }
    let capacity = capacity.min(n_ctx);
    Some((max_kv_bytes as u128 * capacity as u128 / n_ctx as u128) as u64)
}

/// Transcribe one case under one mode and budget.
pub fn run_case(
    model: &mut LoadedModel,
    settings: &AppSettings,
    case: &TestCase,
    dataset_dir: &Path,
    mode: Mode,
    budget: usize,
) -> Result<RunOutcome> {
    let (audio, audio_seconds) = load_audio(&case.audio_path(dataset_dir))?;
    let biasing = model.biasing(if mode.uses_context() { budget } else { 0 });

    // Selection, then the same token fitting the app applies.
    let selected = selected_for(settings, mode, biasing);
    let (n_ctx, max_audio_ms, max_kv_bytes) = model
        .session
        .limits()
        .map(|l| {
            (
                l.effective_n_ctx as i64,
                l.effective_max_audio_ms,
                l.max_kv_bytes,
            )
        })
        .unwrap_or((0, 0, 0));
    let limit = ContextTokenLimit::for_channel(biasing.channel, n_ctx, max_audio_ms);
    let audio_ms = (audio.len() as u64 * 1000) / EXPECTED_SAMPLE_RATE as u64;
    let tokenizer = model.session.model();
    let fit = fit_context(&selected, limit, audio_ms, &|text: &str| {
        tokenizer.tokenize(text).ok().map(|t| t.len())
    });
    let context_terms = fit.sent.clone();
    let fuzzy_terms = fuzzy_for(settings, mode, &context_terms);
    let kv = derived_kv_bytes(limit, &fit, max_kv_bytes);

    let mut run_options = RunOptions::default();
    apply_context_to_run_options(biasing, &context_terms, &mut run_options);
    // Read back what production actually put in, rather than re-rendering it.
    let context = run_options
        .context
        .clone()
        .or_else(|| match &run_options.family {
            Some(transcribe_cpp::RunExtension::Whisper(w)) => w.initial_prompt.clone(),
            _ => None,
        });

    let started = Instant::now();
    let (transcript, fell_back_without_context) =
        run_with_context_fallback(&mut model.session, &audio, &run_options);
    let transcribe_ms = started.elapsed().as_secs_f64() * 1000.0;
    let (raw_transcript, error) = match transcript {
        Ok(transcript) => (transcript.text, None),
        // Recorded, not skipped: a lost dictation is a result, and the worst one.
        Err(e) => (String::new(), Some(e.to_string())),
    };
    // As in the app: after a fallback the model saw no context, so the terms
    // it would have seen go back into fuzzy correction.
    let fuzzy_terms = if fell_back_without_context {
        fuzzy_for(settings, mode, &[])
    } else {
        fuzzy_terms
    };

    let fuzzy_started = Instant::now();
    let correction = if fuzzy_terms.is_empty() {
        CorrectionReport {
            text: raw_transcript.clone(),
            ..Default::default()
        }
    } else {
        correct_with_vocabulary(
            &raw_transcript,
            &fuzzy_terms,
            settings.word_correction_threshold,
        )
    };
    let fuzzy_ms = fuzzy_started.elapsed().as_secs_f64() * 1000.0;

    Ok(RunOutcome {
        final_transcript: correction.text.clone(),
        raw_transcript,
        context,
        selected_terms: selected.len(),
        context_terms,
        fuzzy_terms,
        fit,
        correction,
        derived_kv_bytes: kv,
        audio_seconds,
        transcribe_ms,
        fuzzy_ms,
        fell_back_without_context,
        error,
    })
}

/// Which of a case's target terms were in the context — the denominator for
/// context hit rate, and what separates "biasing worked" from "the model knew
/// it anyway".
pub fn targets_in_context(case: &TestCase, context_terms: &[String]) -> usize {
    case.target_terms
        .iter()
        .filter(|target| {
            context_terms
                .iter()
                .any(|term| term.eq_ignore_ascii_case(target))
        })
        .count()
}

/// Target terms that were in the context *and* made it into the transcript.
pub fn targets_in_context_recognized(
    case: &TestCase,
    context_terms: &[String],
    transcript: &str,
) -> usize {
    case.target_terms
        .iter()
        .filter(|target| {
            context_terms
                .iter()
                .any(|term| term.eq_ignore_ascii_case(target))
                && metrics::contains_term(transcript, target)
        })
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;
    use handy_app_lib::settings::get_default_settings;

    fn settings() -> AppSettings {
        let mut settings = get_default_settings();
        settings.active_dictionaries = vec![
            "core_medical".to_string(),
            "internal_cardiology".to_string(),
        ];
        settings
            .dictionary_levels
            .insert("core_medical".into(), 250);
        settings
            .dictionary_levels
            .insert("internal_cardiology".into(), 250);
        // Both steps, as a plain benchmark config maps them.
        settings.context_dictionaries = settings.active_dictionaries.clone();
        settings
    }

    fn biasing(budget: usize) -> ContextBiasing {
        ContextBiasing {
            channel: Some(handy_app_lib::ContextChannel::RecognitionContext),
            budget,
        }
    }

    #[test]
    fn mode_a_sends_and_corrects_nothing() {
        let (context, fuzzy) = vocabulary_for(&settings(), Mode::A, biasing(120));
        assert!(context.is_empty());
        assert!(fuzzy.is_empty());
    }

    #[test]
    fn mode_b_corrects_with_the_whole_pool_but_sends_no_context() {
        let settings = settings();
        let (context, fuzzy) = vocabulary_for(&settings, Mode::B, biasing(120));
        assert!(context.is_empty());
        assert_eq!(fuzzy, full_vocabulary(&settings));
    }

    #[test]
    fn mode_c_removes_context_terms_from_the_fuzzy_pool() {
        let settings = settings();
        let (context, fuzzy) = vocabulary_for(&settings, Mode::C, biasing(120));
        assert_eq!(context.len(), 120);
        let full = full_vocabulary(&settings);
        assert_eq!(fuzzy.len(), full.len() - context.len());
        for term in &context {
            assert!(
                !fuzzy.iter().any(|f| f.eq_ignore_ascii_case(term)),
                "{} should have been removed from the fuzzy pool",
                term
            );
        }
    }

    #[test]
    fn mode_d_keeps_context_terms_in_the_fuzzy_pool() {
        let settings = settings();
        let (context, fuzzy) = vocabulary_for(&settings, Mode::D, biasing(120));
        assert_eq!(context.len(), 120);
        assert_eq!(fuzzy, full_vocabulary(&settings));
        // The C/D difference is exactly the context terms, nothing else.
        let (_, fuzzy_c) = vocabulary_for(&settings, Mode::C, biasing(120));
        assert_eq!(fuzzy.len() - fuzzy_c.len(), context.len());
    }

    #[test]
    fn budget_zero_produces_no_context_even_in_a_context_mode() {
        let settings = settings();
        for mode in [Mode::C, Mode::D] {
            let (context, fuzzy) = vocabulary_for(&settings, mode, biasing(0));
            assert!(context.is_empty(), "{:?} sent context at budget 0", mode);
            // …and with nothing sent, mode C corrects the whole pool.
            assert_eq!(fuzzy, full_vocabulary(&settings));
        }
    }

    #[test]
    fn a_budget_caps_the_context_and_scales_with_it() {
        let settings = settings();
        for budget in [30usize, 60, 120, 240] {
            let (context, _) = vocabulary_for(&settings, Mode::C, biasing(budget));
            assert_eq!(context.len(), budget);
        }
    }

    #[test]
    fn an_unsupported_model_never_receives_context() {
        let settings = settings();
        let none = ContextBiasing::NONE;
        for mode in Mode::ALL {
            let (context, _) = vocabulary_for(&settings, mode, none);
            assert!(
                context.is_empty(),
                "{:?} sent context to a model with no channel",
                mode
            );
        }
    }

    #[test]
    fn context_terms_reach_run_options_through_the_production_helper() {
        let settings = settings();
        let (context_terms, _) = vocabulary_for(&settings, Mode::C, biasing(120));
        let mut options = RunOptions::default();
        apply_context_to_run_options(biasing(120), &context_terms, &mut options);
        let rendered = options.context.expect("no recognition context");
        assert_eq!(rendered.split(", ").count(), 120);
        assert!(rendered.contains(&context_terms[0]));
    }

    #[test]
    fn context_hit_rate_separates_covered_targets_from_uncovered_ones() {
        let case = TestCase {
            id: "t".into(),
            audio: "a.wav".into(),
            reference: "r".into(),
            specialty: "internal_cardiology".into(),
            target_terms: vec!["Aortenklappenstenose".into(), "Nichtvorhanden".into()],
            negative_terms: vec![],
            difficulty: "normal".into(),
            medications: vec![],
            negations: vec![],
            numbers: vec![],
            confusable_terms: vec![],
        };
        let context = vec!["Aortenklappenstenose".to_string()];
        assert_eq!(targets_in_context(&case, &context), 1);
        assert_eq!(
            targets_in_context_recognized(&case, &context, "eine Aortenklappenstenose liegt vor"),
            1
        );
        assert_eq!(
            targets_in_context_recognized(&case, &context, "kein Befund"),
            0
        );
    }
}
