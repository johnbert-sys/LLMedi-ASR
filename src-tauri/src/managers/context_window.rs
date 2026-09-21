//! Fitting the selected biasing terms into a model's real token window.
//!
//! Four quantities are kept apart on purpose, because conflating them is how
//! terms went missing before:
//!
//! 1. the **candidate pool** — every term of every active dictionary up to its
//!    tier (`dictionaries::full_vocabulary`);
//! 2. the **selection** — the pool's best N terms by priority and round-robin,
//!    N being the app's term budget (`dictionaries::context_vocabulary`);
//! 3. the **token cost** of that selection, measured with the model's own
//!    tokenizer — never estimated from a characters-per-token guess;
//! 4. the **token room** the backend actually gives the context for this run.
//!
//! When (3) exceeds (4), the lowest-priority selected terms are *deferred*: not
//! sent to the model, and therefore still handed to fuzzy post-correction, which
//! corrects against everything the model did not see. Nothing is dropped
//! silently — the old behaviour, where whisper's library cut the front of an
//! over-long prompt (our personal words and top-ranked terms) while the app
//! assumed they had been sent and withheld them from fuzzy correction too.
//!
//! The constants mirror values in transcribe.cpp (at the pinned PR-#144 rev)
//! and are named after them so they can be re-checked against a new version.

/// Whisper keeps at most `n_text_ctx / 2 - 1` prompt tokens and discards the
/// *front* of anything longer (`whisper/model.cpp`, "left-truncate, keep
/// most-recent"). `n_text_ctx` is 448 for every whisper size, tiny through
/// large-v3 and turbo, so the cap is 223.
pub const WHISPER_PROMPT_CAP: usize = 448 / 2 - 1;

/// Headroom kept below whisper's cap. The library tokenizes the prompt itself;
/// a few tokens of difference from `Model::tokenize` (leading-space handling)
/// must not tip a fitted prompt back into truncation.
pub const WHISPER_PROMPT_MARGIN: usize = 8;

/// qwen3_asr's allowance for its chat template (`k_prompt_overhead` in
/// `qwen3_asr/model.cpp`, which the library itself calls advisory).
pub const QWEN_TEMPLATE_OVERHEAD: usize = 48;

/// qwen3_asr's fixed generation budget per run (`k_max_new`). A transcript that
/// needs more is truncated, whatever the context does.
pub const QWEN_OUTPUT_RESERVE: usize = 256;

/// Our own headroom on top of the library's advisory template figure: the
/// language-hint prefix and any template drift must not turn a fitted run into
/// `TRANSCRIBE_ERR_INPUT_TOO_LONG`, which fails the whole transcription.
pub const SHARED_WINDOW_MARGIN: usize = 64;

/// Where a model's context-token ceiling comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContextTokenLimit {
    /// A fixed prompt cap (whisper).
    PromptCap(usize),
    /// One decoder window shared by template, context, audio and output
    /// (qwen3_asr). Both figures come from the session's reported limits.
    SharedWindow { n_ctx: usize, max_audio_ms: u64 },
    /// The backend reports no usable figure; nothing can be verified.
    Unknown,
}

impl ContextTokenLimit {
    /// Build the limit for a context channel from the session's reported
    /// `effective_n_ctx` and `effective_max_audio_ms`.
    pub fn for_channel(
        channel: Option<crate::managers::transcription::ContextChannel>,
        effective_n_ctx: i64,
        effective_max_audio_ms: i64,
    ) -> Self {
        use crate::managers::transcription::ContextChannel;
        match channel {
            Some(ContextChannel::InitialPrompt) => ContextTokenLimit::PromptCap(WHISPER_PROMPT_CAP),
            Some(ContextChannel::RecognitionContext)
                if effective_n_ctx > 0 && effective_max_audio_ms > 0 =>
            {
                ContextTokenLimit::SharedWindow {
                    n_ctx: effective_n_ctx as usize,
                    max_audio_ms: effective_max_audio_ms as u64,
                }
            }
            _ => ContextTokenLimit::Unknown,
        }
    }

    /// Audio tokens a clip of `audio_ms` occupies, using the library's own
    /// ratio: its `max_audio_ms` is defined as the audio that fills the window
    /// after template and output reserve, so tokens-per-ms follows exactly.
    pub fn audio_tokens(self, audio_ms: u64) -> Option<usize> {
        match self {
            ContextTokenLimit::SharedWindow {
                n_ctx,
                max_audio_ms,
            } => {
                let audio_room = n_ctx.checked_sub(QWEN_TEMPLATE_OVERHEAD + QWEN_OUTPUT_RESERVE)?;
                Some(
                    ((audio_ms as u128 * audio_room as u128).div_ceil(max_audio_ms as u128))
                        as usize,
                )
            }
            _ => None,
        }
    }

    /// Tokens the context may use for a clip of `audio_ms`, or `None` when the
    /// backend gives no figure to check against.
    pub fn context_room(self, audio_ms: u64) -> Option<usize> {
        match self {
            ContextTokenLimit::PromptCap(cap) => Some(cap.saturating_sub(WHISPER_PROMPT_MARGIN)),
            ContextTokenLimit::SharedWindow { n_ctx, .. } => {
                let used = QWEN_TEMPLATE_OVERHEAD
                    + QWEN_OUTPUT_RESERVE
                    + SHARED_WINDOW_MARGIN
                    + self.audio_tokens(audio_ms)?;
                Some(n_ctx.saturating_sub(used))
            }
            ContextTokenLimit::Unknown => None,
        }
    }
}

/// The outcome of fitting a selection into the window: what is sent, what is
/// deferred to fuzzy correction, and the numbers behind the decision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WindowFit {
    /// Terms sent to the model, in priority order.
    pub sent: Vec<String>,
    /// Selected terms that did not fit; fuzzy correction still covers them.
    pub deferred: Vec<String>,
    /// Measured token cost of what is sent. `None` if the tokenizer failed.
    pub context_tokens: Option<usize>,
    /// Tokens the context was allowed. `None` if the backend gave no figure.
    pub context_room: Option<usize>,
    /// Audio tokens assumed for this run, where the window is shared.
    pub audio_tokens: Option<usize>,
}

impl WindowFit {
    /// Whether the token cost was actually measured and checked.
    pub fn verified(&self) -> bool {
        self.context_tokens.is_some() && self.context_room.is_some()
    }
}

/// Fit `selected` (in priority order) into the room `limit` leaves for a clip
/// of `audio_ms`, measuring with `tokenize` (the model's own tokenizer; returns
/// `None` if it cannot encode).
///
/// Keeps the longest *prefix* that fits, so priority order is honoured:
/// personal words and each module's top-ranked terms are the last to go. When
/// either the room or the token count is unknown, everything selected is sent
/// unchecked and the result says so via [`WindowFit::verified`] — an estimate
/// would only pretend to a precision we do not have.
pub fn fit_context(
    selected: &[String],
    limit: ContextTokenLimit,
    audio_ms: u64,
    tokenize: &dyn Fn(&str) -> Option<usize>,
) -> WindowFit {
    let render = crate::managers::transcription::format_context_terms;
    let audio_tokens = limit.audio_tokens(audio_ms);
    let room = limit.context_room(audio_ms);
    let unchecked = |context_tokens| WindowFit {
        sent: selected.to_vec(),
        deferred: Vec::new(),
        context_tokens,
        context_room: room,
        audio_tokens,
    };

    if selected.is_empty() {
        return WindowFit {
            context_tokens: Some(0),
            ..unchecked(Some(0))
        };
    }
    let Some(room) = room else {
        return unchecked(tokenize(&render(selected)));
    };
    let Some(all_tokens) = tokenize(&render(selected)) else {
        return unchecked(None);
    };
    if all_tokens <= room {
        return unchecked(Some(all_tokens));
    }

    // Token cost grows with every appended term, so the largest fitting prefix
    // can be found by bisection instead of re-tokenizing term by term.
    let (mut fits, mut too_long) = (0usize, selected.len());
    let mut fits_tokens = 0usize;
    while too_long - fits > 1 {
        let mid = (fits + too_long) / 2;
        match tokenize(&render(&selected[..mid])) {
            Some(tokens) if tokens <= room => {
                fits = mid;
                fits_tokens = tokens;
            }
            Some(_) => too_long = mid,
            // A prefix the tokenizer cannot encode is treated as not fitting.
            None => too_long = mid,
        }
    }
    WindowFit {
        sent: selected[..fits].to_vec(),
        deferred: selected[fits..].to_vec(),
        context_tokens: Some(fits_tokens),
        context_room: Some(room),
        audio_tokens,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::managers::transcription::ContextChannel;

    fn terms(n: usize) -> Vec<String> {
        (0..n).map(|i| format!("Begriff{:03}", i)).collect()
    }

    /// A stand-in tokenizer with a known cost: one token per four bytes,
    /// rounded up — enough to exercise the fitting logic exactly.
    fn by_bytes(text: &str) -> Option<usize> {
        Some(text.len().div_ceil(4))
    }

    #[test]
    fn whisper_cap_is_derived_from_its_fixed_text_context() {
        assert_eq!(WHISPER_PROMPT_CAP, 223);
        let limit = ContextTokenLimit::for_channel(Some(ContextChannel::InitialPrompt), 0, 0);
        assert_eq!(limit, ContextTokenLimit::PromptCap(223));
        assert_eq!(
            limit.context_room(60_000),
            Some(223 - WHISPER_PROMPT_MARGIN)
        );
    }

    #[test]
    fn a_selection_that_fits_is_sent_whole_and_measured() {
        let selected = terms(10);
        let fit = fit_context(&selected, ContextTokenLimit::PromptCap(223), 0, &by_bytes);
        assert_eq!(fit.sent, selected);
        assert!(fit.deferred.is_empty());
        assert!(fit.verified());
        assert!(fit.context_tokens.unwrap() <= 223 - WHISPER_PROMPT_MARGIN);
    }

    #[test]
    fn overflow_defers_the_lowest_priority_tail_and_loses_nothing() {
        let selected = terms(120);
        let fit = fit_context(&selected, ContextTokenLimit::PromptCap(223), 0, &by_bytes);
        let room = 223 - WHISPER_PROMPT_MARGIN;
        // The highest-priority terms are the ones kept — the front, not the back.
        assert_eq!(fit.sent[0], selected[0]);
        assert!(!fit.deferred.is_empty());
        // Sent + deferred is exactly the selection, in order.
        let mut rejoined = fit.sent.clone();
        rejoined.extend(fit.deferred.clone());
        assert_eq!(rejoined, selected);
        // And the prefix is maximal: one more term would not fit.
        let sent_tokens = fit.context_tokens.unwrap();
        assert!(sent_tokens <= room);
        let one_more = format!("{}, {}", fit.sent.join(", "), fit.deferred[0]);
        assert!(by_bytes(&one_more).unwrap() > room);
    }

    #[test]
    fn shared_window_subtracts_template_output_audio_and_margin() {
        // Real figures from Qwen3-ASR-1.7B: 65 536-token window, 5 218 560 ms.
        let limit = ContextTokenLimit::SharedWindow {
            n_ctx: 65_536,
            max_audio_ms: 5_218_560,
        };
        // 60 s of audio at the library's own ratio is 750 tokens.
        assert_eq!(limit.audio_tokens(60_000), Some(750));
        assert_eq!(
            limit.context_room(60_000),
            Some(65_536 - 48 - 256 - 64 - 750)
        );
        // At the advertised maximum the audio alone fills the window.
        assert_eq!(limit.context_room(5_218_560), Some(0));
    }

    #[test]
    fn long_audio_shrinks_the_context_instead_of_failing_the_run() {
        // A small window where audio leaves only a little room.
        let limit = ContextTokenLimit::SharedWindow {
            n_ctx: 1_000,
            max_audio_ms: 100_000,
        };
        let selected = terms(50);
        let short = fit_context(&selected, limit, 1_000, &by_bytes);
        let long = fit_context(&selected, limit, 90_000, &by_bytes);
        assert!(long.sent.len() < short.sent.len());
        assert_eq!(long.sent.len() + long.deferred.len(), selected.len());
        assert!(long.context_tokens.unwrap() <= long.context_room.unwrap());
    }

    #[test]
    fn unknown_limit_or_tokenizer_is_reported_as_unverified() {
        let selected = terms(20);
        let no_room = fit_context(&selected, ContextTokenLimit::Unknown, 0, &by_bytes);
        assert_eq!(no_room.sent, selected);
        assert!(!no_room.verified());

        let no_tokenizer = fit_context(&selected, ContextTokenLimit::PromptCap(223), 0, &|_| None);
        assert_eq!(no_tokenizer.sent, selected);
        assert!(!no_tokenizer.verified());
    }

    #[test]
    fn an_unsupported_channel_has_no_limit() {
        assert_eq!(
            ContextTokenLimit::for_channel(None, 65_536, 5_218_560),
            ContextTokenLimit::Unknown
        );
        // A recognition-context model that reports no window is not guessed at.
        assert_eq!(
            ContextTokenLimit::for_channel(Some(ContextChannel::RecognitionContext), 0, 0),
            ContextTokenLimit::Unknown
        );
    }

    #[test]
    fn an_empty_selection_costs_nothing() {
        let fit = fit_context(&[], ContextTokenLimit::PromptCap(223), 0, &by_bytes);
        assert!(fit.sent.is_empty());
        assert_eq!(fit.context_tokens, Some(0));
    }
}
