//! Native smoke test for Qwen3-ASR recognition context.
//!
//! Everything else about the context path is covered by pure unit tests in
//! `managers::transcription`. This one closes the last gap: that a real
//! Qwen3-ASR GGUF, loaded through the patched transcribe-cpp, actually
//! *advertises* `Feature::Context` and *accepts* a populated
//! `RunOptions::context` instead of rejecting it the way the whisper-tagged run
//! extension would (INVALID_ARG on a foreign arch).
//!
//! Ignored by default: it needs a multi-GB model on disk, so it must never gate
//! a normal `cargo test`. Run it deliberately:
//!
//! ```text
//! LLMEDI_QWEN_GGUF=/path/to/Qwen3-ASR-1.7B-Q5_K_M.gguf \
//!   cargo test --test qwen_recognition_context -- --ignored --nocapture
//! ```
//!
//! The assertion is about the *contract*, not about transcription quality: a
//! synthetic clip cannot be expected to produce a given word, and asserting on
//! decoded text would make this flaky for no benefit.

use transcribe_cpp::{Feature, Model, RunOptions};

/// Terms a general-purpose ASR is unlikely to produce on its own — the kind of
/// vocabulary this whole feature exists for.
const CONTEXT: &str =
    "AV-Knoten-Reentrytachykardie, Torsade-de-pointes-Tachykardie, Sacubitril/Valsartan";

#[test]
#[ignore = "requires a local Qwen3-ASR GGUF; set LLMEDI_QWEN_GGUF"]
fn qwen_gguf_advertises_and_accepts_a_recognition_context() {
    let Ok(path) = std::env::var("LLMEDI_QWEN_GGUF") else {
        panic!("set LLMEDI_QWEN_GGUF to a Qwen3-ASR .gguf to run this test");
    };

    let model = Model::load(&path).expect("failed to load the Qwen3-ASR model");
    assert_eq!(model.arch(), "qwen3_asr", "not a Qwen3-ASR model: {}", path);

    // The capability our `transcribe_cpp_context_biasing` probes in production.
    assert!(
        model.supports(Feature::Context),
        "this build of transcribe-cpp does not expose recognition context for \
         qwen3_asr — is the PR #144 patch still applied in Cargo.toml?"
    );

    let mut session = model.session().expect("failed to open a session");
    // One second of silence at 16 kHz: enough to exercise the full prefill and
    // decode path, cheap enough to keep the test quick.
    let audio = vec![0.0f32; 16_000];

    let options = RunOptions {
        context: Some(CONTEXT.to_string()),
        ..Default::default()
    };
    let with_context = session
        .run(&audio, &options)
        .expect("run with a recognition context failed");

    // And the same clip without one, to show the context is genuinely optional
    // rather than load-bearing for the call to succeed.
    let without_context = session
        .run(&audio, &RunOptions::default())
        .expect("run without a context failed");

    println!(
        "qwen3_asr context smoke: with={:?} without={:?}",
        with_context.text, without_context.text
    );
}

/// Prints the real window arithmetic for the local Qwen model: decoder window,
/// audio ceiling, and what the app's actual 120-term context costs in tokens.
/// Evidence for the context-budget design, not an assertion suite.
#[test]
#[ignore = "requires a local Qwen3-ASR GGUF; set LLMEDI_QWEN_GGUF"]
fn qwen_window_arithmetic() {
    use handy_app_lib::dictionaries::context_vocabulary;
    use handy_app_lib::settings::get_default_settings;
    let path = std::env::var("LLMEDI_QWEN_GGUF").expect("set LLMEDI_QWEN_GGUF");
    let model = Model::load(&path).unwrap();
    let session = model.session().unwrap();
    let limits = session.limits().unwrap();
    let caps = model.capabilities();
    println!("WINDOW effective_n_ctx={} effective_max_audio_ms={} max_kv_bytes={} caps.max_audio_ms={} native_sr={}",
        limits.effective_n_ctx, limits.effective_max_audio_ms, limits.max_kv_bytes, caps.max_audio_ms, caps.native_sample_rate);

    let mut settings = get_default_settings();
    settings.active_dictionaries = [
        "core_medical",
        "internal_cardiology",
        "anatomy_heart_vessels",
        "meds_cardiology_generic",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    for (id, l) in [
        ("core_medical", 250u32),
        ("internal_cardiology", 250),
        ("anatomy_heart_vessels", 100),
        ("meds_cardiology_generic", 50),
    ] {
        settings.dictionary_levels.insert(id.to_string(), l);
    }
    for budget in [30usize, 60, 120, 240, 400] {
        let terms = context_vocabulary(&settings, budget);
        let text = terms.join(", ");
        let tokens = model.tokenize(&text).map(|t| t.len());
        println!(
            "CONTEXT budget={} terms={} chars={} tokens={:?} tokens_per_term={:.2}",
            budget,
            terms.len(),
            text.chars().count(),
            tokens,
            tokens
                .as_ref()
                .map(|t| *t as f64 / terms.len().max(1) as f64)
                .unwrap_or(0.0)
        );
    }
    for sample in [
        "Herz",
        "Aortenklappenstenose",
        "transkatheter Aortenklappenimplantation",
        "AV-Knoten-Reentrytachykardie",
    ] {
        println!(
            "TOKENS {:?} = {:?}",
            sample,
            model.tokenize(sample).map(|t| t.len())
        );
    }
}

/// Whisper's initial-prompt cost for the same context, measured with its own
/// tokenizer. The library left-truncates prompts beyond n_text_ctx/2 - 1 tokens.
#[test]
#[ignore = "requires a local whisper GGUF; set LLMEDI_WHISPER_GGUF"]
fn whisper_prompt_arithmetic() {
    use handy_app_lib::dictionaries::context_vocabulary;
    use handy_app_lib::settings::get_default_settings;
    let path = std::env::var("LLMEDI_WHISPER_GGUF").expect("set LLMEDI_WHISPER_GGUF");
    let model = Model::load(&path).unwrap();
    println!("ARCH {}", model.arch());
    let mut settings = get_default_settings();
    settings.custom_words = vec!["Blatt-Schmidt-Zeichen".to_string()];
    settings.active_dictionaries = [
        "core_medical",
        "internal_cardiology",
        "anatomy_heart_vessels",
        "meds_cardiology_generic",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    for (id, l) in [
        ("core_medical", 250u32),
        ("internal_cardiology", 250),
        ("anatomy_heart_vessels", 100),
        ("meds_cardiology_generic", 50),
    ] {
        settings.dictionary_levels.insert(id.to_string(), l);
    }
    let terms = context_vocabulary(&settings, 120);
    let text = terms.join(", ");
    let n = model.tokenize(&text).map(|t| t.len());
    println!("WHISPER 120 terms -> tokens {:?}", n);
    // How many leading terms fit in 223 tokens?
    let mut fit = 0;
    for k in 1..=terms.len() {
        if model
            .tokenize(&terms[..k].join(", "))
            .map(|t| t.len())
            .unwrap_or(usize::MAX)
            <= 223
        {
            fit = k;
        } else {
            break;
        }
    }
    println!("WHISPER terms fitting 223 tokens: {}", fit);
}
