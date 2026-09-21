use crate::audio_toolkit::{
    detect_output_language, normalize_transcription_output, remove_filler_words,
    OutputLanguageEvidence,
};
use crate::managers::audio::AudioRecordingManager;
use crate::managers::context_window::{fit_context, ContextTokenLimit};
use crate::managers::model::{EngineType, ModelManager};
use crate::settings::{
    get_settings, AppSettings, ModelUnloadTimeout, OrtAcceleratorSetting,
    TranscribeAcceleratorSetting,
};
use anyhow::Result;
use log::{debug, error, info, warn};
use serde::{Deserialize, Serialize};
use specta::Type;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Condvar, Mutex, MutexGuard, OnceLock};
use std::thread;
use std::time::{Duration, Instant, SystemTime};
use tauri::{AppHandle, Emitter, Manager};
use tauri_specta::Event;
use transcribe_cpp::{
    Backend, Feature, Model, ModelOptions, RunExtension, RunOptions, Session, StreamOptions, Task,
    WhisperRunOptions,
};
use transcribe_rs::{
    onnx::{
        canary::CanaryModel,
        cohere::CohereModel,
        gigaam::GigaAMModel,
        moonshine::{MoonshineModel, MoonshineVariant, StreamingModel},
        parakeet::{ParakeetModel, ParakeetParams, TimestampGranularity},
        sense_voice::{SenseVoiceModel, SenseVoiceParams},
        Quantization,
    },
    SpeechModel, TranscribeOptions,
};

const STREAM_PERF_LOG_INTERVAL: Duration = Duration::from_secs(5);
const STREAM_FINALIZE_REPLY_TIMEOUT: Duration = Duration::from_secs(30);
/// Whisper's *selection* budget, counted in terms.
///
/// This is how many terms are picked, not how many reach the model: whisper
/// keeps at most 223 prompt tokens, and German medical terms cost about eight
/// tokens each with its tokenizer (measured with whisper-large-v3-turbo: 120
/// terms = 995 tokens, 23 fit). The earlier rationale here ("~2 tokens per
/// term") was wrong by a factor of four. The real cut is now made in tokens by
/// `context_window::fit_context`, which keeps the highest-priority prefix and
/// defers the rest to fuzzy correction; this number only bounds how many
/// candidates that fitting considers.
pub const WHISPER_CONTEXT_BUDGET: usize = 120;

/// Conservative starting *selection* budget for an LLM-style decoder
/// (qwen3_asr and relatives), counted in terms.
///
/// Not a model limit. Measured on Qwen3-ASR-1.7B: the decoder window is 65 536
/// tokens and 120 selected terms cost 947 tokens, so the window is rarely what
/// binds for dictation-length audio; `context_window::fit_context` still checks
/// every run against it, including long clips where audio and context compete.
/// The value is a deliberate caution — an over-stuffed biasing context invites
/// the model to emit terms that were never spoken — and stays equal to
/// whisper's until the benchmark shows a better one. Choosing it is an open
/// decision, not a result.
pub const LLM_DECODER_CONTEXT_BUDGET: usize = 120;

/// How a model takes biasing terms. The two channels are genuinely different
/// API surfaces, not styles of the same one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContextChannel {
    /// Whisper's decode prompt, carried by the whisper-tagged run extension.
    /// Kind-checked natively, so it may only be attached to a whisper model.
    InitialPrompt,
    /// `RunOptions::context` — free background text a model uses to bias
    /// recognition of names and jargon. Explicitly *not* an instruction prompt,
    /// and passed through byte-for-byte by the library.
    RecognitionContext,
}

/// Decode-time context biasing — an initial prompt, hotword list or system
/// prompt vocabulary — that a loaded model can actually receive *through the
/// backend this app calls*.
///
/// Deliberately a property of the ASR layer rather than the dictionary layer:
/// `crate::dictionaries` only knows how to pick the best N terms, and the
/// backend decides whether it can deliver any and how many.
///
/// `budget` counts terms, not tokens. A token budget would be the better unit
/// ("Herz" and "transkatheter Aortenklappenimplantation" are not equally
/// expensive), and the shape here allows that later: the field and
/// [`crate::dictionaries::context_vocabulary`]'s parameter change together,
/// with no effect on module selection or tiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContextBiasing {
    /// `None` when the model takes no biasing terms at all.
    pub channel: Option<ContextChannel>,
    pub budget: usize,
}

impl ContextBiasing {
    pub const NONE: Self = ContextBiasing {
        channel: None,
        budget: 0,
    };

    pub fn supported(self) -> bool {
        self.channel.is_some()
    }

    /// The terms to hand this model, already budgeted and fairly shared across
    /// the active dictionary modules. Empty when the model takes no context —
    /// the full pool then reaches fuzzy post-correction instead.
    pub fn terms(self, settings: &AppSettings) -> Vec<String> {
        if !self.supported() {
            return Vec::new();
        }
        crate::dictionaries::context_vocabulary(settings, self.budget)
    }
}

/// How a loaded transcribe-cpp model can receive biasing terms.
///
/// `supports_recognition_context` is the model's own `Feature::Context` probe,
/// passed in rather than read here so this stays a pure function.
///
/// Whisper is matched on its architecture, not on a feature: its channel is the
/// kind-tagged whisper run extension, which is rejected with INVALID_ARG on any
/// other arch (see #1601) — a feature flag alone would not make it safe to
/// attach. Every other model goes through `RunOptions::context`, which is a
/// plain field with no kind check, so probing the capability is both sufficient
/// and future-proof: a new arch that advertises it works without a code change.
pub fn transcribe_cpp_context_biasing(
    arch: &str,
    supports_recognition_context: bool,
) -> ContextBiasing {
    if arch == "whisper" {
        return ContextBiasing {
            channel: Some(ContextChannel::InitialPrompt),
            budget: WHISPER_CONTEXT_BUDGET,
        };
    }
    if supports_recognition_context {
        return ContextBiasing {
            channel: Some(ContextChannel::RecognitionContext),
            budget: LLM_DECODER_CONTEXT_BUDGET,
        };
    }
    ContextBiasing::NONE
}

/// Render biasing terms into the string a backend sends to the model.
///
/// A bare comma-separated list. It is what whisper's initial prompt expects,
/// and it suits a recognition context too: upstream documents that field as
/// background text kept byte-for-byte and explicitly *not* an instruction, so
/// framing it with a sentence would only add tokens the model might echo.
/// Multi-word terms stay whole — nothing here splits on whitespace. No scores,
/// module names, rankings or other metadata ever appear: the dictionary layer
/// hands over terms only.
pub fn format_context_terms(terms: &[String]) -> String {
    terms.join(", ")
}

/// Run a transcription; if it fails while a biasing context was attached, run
/// it once more without one.
///
/// A context can push a model into a loop until it hits its output cap —
/// measured on Qwen3-ASR with "Atorvastatin 40 mg zur Nacht" and 240+ context
/// terms — and the run then errors out. Losing the dictation to the context is
/// never acceptable, so the context is dropped instead. Returns the result and
/// whether the retry was needed; the caller must then hand fuzzy correction the
/// whole pool, since the model saw no context after all.
pub fn run_with_context_fallback(
    session: &mut transcribe_cpp::Session,
    audio: &[f32],
    options: &RunOptions,
) -> (transcribe_cpp::Result<transcribe_cpp::Transcript>, bool) {
    let had_context = options.context.is_some() || options.family.is_some();
    match session.run(audio, options) {
        Err(error) if had_context => {
            warn!(
                "transcription with biasing context failed ({}); retrying without context",
                error
            );
            let mut bare = options.clone();
            bare.context = None;
            bare.family = None;
            (session.run(audio, &bare), true)
        }
        other => (other, false),
    }
}

/// What the most recent transcription actually did with the vocabulary, kept so
/// the settings page can show it. Numbers only — no transcript text.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct LastContextUsage {
    pub model_arch: String,
    /// Terms the term budget selected.
    pub selected: u32,
    /// Terms actually sent to the model.
    pub sent: u32,
    /// Selected terms that did not fit the token window; covered by fuzzy
    /// correction instead.
    pub deferred: u32,
    /// Measured token cost of what was sent, if the tokenizer could measure it.
    pub context_tokens: Option<u32>,
    /// Token room the backend left the context, if it reported one.
    pub context_room: Option<u32>,
    /// Size of the active pool fuzzy correction drew from.
    pub pool: u32,
    /// The run failed with context and was repeated without it.
    pub fell_back_without_context: bool,
}

static LAST_CONTEXT_USAGE: std::sync::Mutex<Option<LastContextUsage>> = std::sync::Mutex::new(None);

fn remember_context_usage(usage: LastContextUsage) {
    if let Ok(mut slot) = LAST_CONTEXT_USAGE.lock() {
        *slot = Some(usage);
    }
}

fn mark_context_fallback() {
    if let Ok(mut slot) = LAST_CONTEXT_USAGE.lock() {
        if let Some(usage) = slot.as_mut() {
            usage.fell_back_without_context = true;
            usage.sent = 0;
            usage.context_tokens = Some(0);
        }
    }
}

#[tauri::command]
#[specta::specta]
pub fn get_last_context_usage() -> Option<LastContextUsage> {
    LAST_CONTEXT_USAGE.lock().ok().and_then(|slot| slot.clone())
}

/// Put the rendered biasing terms into whichever `RunOptions` slot `biasing`
/// names, leaving both untouched for a model that takes none rather than
/// forcing an extension the native side would reject.
///
/// This is the single place that knows how a context reaches a model, shared by
/// the live transcription path and the benchmark harness so the two can never
/// drift apart.
pub fn apply_context_to_run_options(
    biasing: ContextBiasing,
    terms: &[String],
    options: &mut RunOptions,
) {
    let rendered = (!terms.is_empty()).then(|| format_context_terms(terms));
    match biasing.channel {
        Some(ContextChannel::InitialPrompt) => {
            options.family = rendered.map(|prompt| {
                RunExtension::Whisper(WhisperRunOptions {
                    initial_prompt: Some(prompt),
                    ..Default::default()
                })
            });
        }
        Some(ContextChannel::RecognitionContext) => options.context = rendered,
        None => {}
    }
}

fn panic_payload_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "unknown panic".to_string()
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct ModelStateEvent {
    pub event_type: String,
    pub model_id: Option<String>,
    pub model_name: Option<String>,
    pub error: Option<String>,
}

/// Live transcription snapshot emitted to the overlay during a streaming run.
/// `committed` is the append-only, flicker-free prefix; `tentative` is the
/// volatile suffix the model may still rewrite.
#[derive(Clone, Debug, Serialize, Deserialize, Type, tauri_specta::Event)]
pub struct StreamTextEvent {
    pub committed: String,
    pub tentative: String,
}

/// Phase of the streaming overlay card, emitted to drive its UI state.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Type)]
#[serde(rename_all = "lowercase")]
pub enum StreamPhase {
    /// Receiving audio / live text (or waiting for the stream to begin). Rust
    /// does not emit this today; the frontend starts in this phase and Rust only
    /// emits transitions away from it.
    Listening,
    /// Finalizing or post-processing — show a spinner.
    Working,
}

/// Semantic kind of "working" phase, used to localize the spinner label.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Type)]
#[serde(rename_all = "lowercase")]
pub enum StreamWorkKind {
    Transcribing,
    Polishing,
}

/// Emitted to switch the streaming overlay to a working spinner.
#[derive(Clone, Debug, Serialize, Deserialize, Type, tauri_specta::Event)]
pub struct StreamPhaseEvent {
    pub phase: StreamPhase,
    /// Present only when `phase` is `Working`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kind: Option<StreamWorkKind>,
}

/// Commands sent to the streaming worker thread. Audio frames and the finalize
/// request travel the same channel so FIFO ordering guarantees every fed frame
/// is processed before finalize runs.
enum StreamCmd {
    Feed(Vec<f32>),
    /// Flush the stream and reply with the final text, or `None` if no stream
    /// was ever active (caller should fall back to batch transcription).
    Finalize(mpsc::Sender<Option<FinalizedStreamText>>),
    Cancel,
}

struct FinalizedStreamText {
    text: String,
    output_language: OutputLanguageEvidence,
    /// The streaming model's supported languages, for text-based detection.
    supported_languages: Vec<String>,
}

/// Routes real-time audio frames to the active streaming worker. Shared between
/// the [`TranscriptionManager`] (opens/closes the route) and the audio recorder's
/// per-frame callback (feeds frames). The recorder holds an `Arc<StreamRouter>`
/// directly, so a frame with no stream pending costs a single relaxed atomic
/// load — no Tauri state lookup, no mutex lock.
pub struct StreamRouter {
    /// Command channel to the active streaming worker, present from
    /// `start_stream` until `finalize_stream`/`cancel_stream`.
    tx: Mutex<Option<mpsc::Sender<StreamCmd>>>,
    /// True while a stream is pending or active (channel is open). The audio
    /// callback checks this first to avoid the mutex lock when no stream runs.
    open: Arc<AtomicBool>,
}

impl StreamRouter {
    fn new() -> Self {
        Self {
            tx: Mutex::new(None),
            open: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Open a fresh command channel for a new streaming session, returning the
    /// receiver the worker should drain. Caller must ensure no prior channel is
    /// still open.
    fn open(&self) -> mpsc::Receiver<StreamCmd> {
        let (tx, rx) = mpsc::channel::<StreamCmd>();
        *self.tx.lock().unwrap() = Some(tx);
        self.open.store(true, Ordering::Relaxed);
        rx
    }

    /// Take the sender out (closing the channel to new feeds). Returns the
    /// sender so the caller can send the final `Finalize`/`Cancel` command.
    fn take(&self) -> Option<mpsc::Sender<StreamCmd>> {
        self.open.store(false, Ordering::Relaxed);
        self.tx.lock().unwrap().take()
    }

    /// Drop the channel and mark closed without sending a final command (used
    /// when the worker exits without a finalize/cancel handshake).
    fn clear(&self) {
        self.open.store(false, Ordering::Relaxed);
        *self.tx.lock().unwrap() = None;
    }

    /// Forward a 16 kHz frame to the active streaming worker. Cheap no-op (a
    /// single relaxed atomic load) when no stream is pending.
    pub fn feed(&self, frame: &[f32]) {
        if !self.open.load(Ordering::Relaxed) {
            return;
        }
        if let Some(tx) = self.tx.lock().unwrap().as_ref() {
            let _ = tx.send(StreamCmd::Feed(frame.to_vec()));
        }
    }

    /// Whether a stream is pending or active.
    pub fn is_open(&self) -> bool {
        self.open.load(Ordering::Relaxed)
    }
}

enum LoadedEngine {
    /// Whisper-family models (whisper, breeze-asr, custom .bin/.gguf) via
    /// transcribe-cpp. Holds the live `Session`, which keeps its `Model` alive
    /// internally, so repeated dictation reuses the session without reloading.
    TranscribeCpp(Session),
    Parakeet(ParakeetModel),
    Moonshine(MoonshineModel),
    MoonshineStreaming(StreamingModel),
    SenseVoice(SenseVoiceModel),
    GigaAM(GigaAMModel),
    Canary(CanaryModel),
    Cohere(CohereModel),
}

/// RAII guard that clears the `is_loading` flag and notifies waiters on drop.
/// Ensures the loading flag is always reset, even on early returns or panics.
pub struct LoadingGuard {
    is_loading: Arc<Mutex<bool>>,
    loading_condvar: Arc<Condvar>,
}

impl Drop for LoadingGuard {
    fn drop(&mut self) {
        // Recover from a poisoned mutex instead of panicking —
        // a panic inside Drop calls abort().
        let mut is_loading = match self.is_loading.lock() {
            Ok(g) => g,
            Err(e) => {
                warn!("Recovered poisoned is_loading mutex during LoadingGuard drop — a panic occurred earlier this session");
                e.into_inner()
            }
        };
        *is_loading = false;
        self.loading_condvar.notify_all();
    }
}

/// RAII guard that clears the streaming worker/lease flags on any worker exit -
/// normal return, early return, or a panic in an engine call that unwinds the
/// detached worker thread. Tokens prevent an older worker from clearing a newer
/// worker's state if a start/finalize race ever slips through.
struct StreamWorkerGuard {
    worker_id: u64,
    active_stream_worker: Arc<AtomicU64>,
    active_engine_lease: Arc<AtomicU64>,
    stream_active: Arc<AtomicBool>,
}

impl Drop for StreamWorkerGuard {
    fn drop(&mut self) {
        if self.active_stream_worker.load(Ordering::Acquire) == self.worker_id {
            self.stream_active.store(false, Ordering::Release);
        }
        let _ = self.active_engine_lease.compare_exchange(
            self.worker_id,
            0,
            Ordering::AcqRel,
            Ordering::Acquire,
        );
        let _ = self.active_stream_worker.compare_exchange(
            self.worker_id,
            0,
            Ordering::AcqRel,
            Ordering::Acquire,
        );
    }
}

#[derive(Clone)]
pub struct TranscriptionManager {
    engine: Arc<Mutex<Option<LoadedEngine>>>,
    model_manager: Arc<ModelManager>,
    app_handle: AppHandle,
    current_model_id: Arc<Mutex<Option<String>>>,
    last_activity: Arc<AtomicU64>,
    shutdown_signal: Arc<AtomicBool>,
    watcher_handle: Arc<Mutex<Option<thread::JoinHandle<()>>>>,
    is_loading: Arc<Mutex<bool>>,
    loading_condvar: Arc<Condvar>,
    reload_model_on_next_use: Arc<AtomicBool>,
    /// Routes real-time audio frames to the active streaming worker; see
    /// [`StreamRouter`]. Shared with the audio recorder so per-frame feeds skip
    /// Tauri state and the manager lock.
    router: Arc<StreamRouter>,
    /// True only while a transcribe-cpp `Stream` is actually in flight (set by
    /// the worker once `stream()` succeeds). Used for overlay/UI decisions.
    stream_active: Arc<AtomicBool>,
    /// Streaming uses four independent flags: router open = frames should route,
    /// worker active = no second worker may start, engine lease = engine is out
    /// of the mutex, stream active = UI should show a live session.
    ///
    /// Monotonic id source for stream workers; zero means "no worker".
    next_stream_worker_id: Arc<AtomicU64>,
    /// Nonzero while a stream worker exists, even if it has not leased the engine
    /// yet. This prevents a second worker from starting after finalize/cancel
    /// closes the router but before the first worker has fully exited.
    active_stream_worker: Arc<AtomicU64>,
    /// Nonzero while the streaming worker has taken the engine out of `engine`.
    /// `is_model_loaded()` consults this so the model still reports "loaded"
    /// while the worker holds it.
    active_engine_lease: Arc<AtomicU64>,
}

impl TranscriptionManager {
    pub fn new(app_handle: &AppHandle, model_manager: Arc<ModelManager>) -> Result<Self> {
        let manager = Self {
            engine: Arc::new(Mutex::new(None)),
            model_manager,
            app_handle: app_handle.clone(),
            current_model_id: Arc::new(Mutex::new(None)),
            last_activity: Arc::new(AtomicU64::new(Self::now_ms())),
            shutdown_signal: Arc::new(AtomicBool::new(false)),
            watcher_handle: Arc::new(Mutex::new(None)),
            is_loading: Arc::new(Mutex::new(false)),
            loading_condvar: Arc::new(Condvar::new()),
            reload_model_on_next_use: Arc::new(AtomicBool::new(false)),
            router: Arc::new(StreamRouter::new()),
            stream_active: Arc::new(AtomicBool::new(false)),
            next_stream_worker_id: Arc::new(AtomicU64::new(1)),
            active_stream_worker: Arc::new(AtomicU64::new(0)),
            active_engine_lease: Arc::new(AtomicU64::new(0)),
        };

        // Start the idle watcher
        {
            let app_handle_cloned = app_handle.clone();
            let manager_cloned = manager.clone();
            let shutdown_signal = manager.shutdown_signal.clone();
            let handle = thread::spawn(move || {
                debug!("Idle watcher thread started");
                while !shutdown_signal.load(Ordering::Relaxed) {
                    thread::sleep(Duration::from_secs(10)); // Check every 10 seconds

                    // Check shutdown signal again after sleep
                    if shutdown_signal.load(Ordering::Relaxed) {
                        break;
                    }

                    let settings = get_settings(&app_handle_cloned);
                    let timeout = settings.model_unload_timeout;

                    // Skip Immediately — that variant is handled by
                    // maybe_unload_immediately() after each transcription.
                    // Treating it as 0s here would unload the model mid-recording.
                    if timeout == ModelUnloadTimeout::Immediately {
                        continue;
                    }

                    // While recording, keep the idle timer fresh so the
                    // model is never unloaded mid-session.
                    let is_recording = app_handle_cloned
                        .try_state::<Arc<AudioRecordingManager>>()
                        .is_some_and(|a| a.is_recording());
                    if is_recording {
                        manager_cloned.touch_activity();
                        continue;
                    }

                    if let Some(limit_seconds) = timeout.to_seconds() {
                        let last = manager_cloned.last_activity.load(Ordering::Relaxed);
                        let now_ms = TranscriptionManager::now_ms();
                        let idle_ms = now_ms.saturating_sub(last);
                        let limit_ms = limit_seconds * 1000;

                        if idle_ms > limit_ms {
                            // idle -> unload
                            if manager_cloned.is_model_loaded() {
                                let unload_start = std::time::Instant::now();
                                info!(
                                    "Model idle for {}s (limit: {}s), unloading",
                                    idle_ms / 1000,
                                    limit_seconds
                                );
                                match manager_cloned.unload_model() {
                                    Ok(()) => {
                                        let unload_duration = unload_start.elapsed();
                                        info!(
                                            "Model unloaded due to inactivity (took {}ms)",
                                            unload_duration.as_millis()
                                        );
                                    }
                                    Err(e) => {
                                        error!("Failed to unload idle model: {}", e);
                                    }
                                }
                            }
                        }
                    }
                }
                debug!("Idle watcher thread shutting down gracefully");
            });
            *manager.watcher_handle.lock().unwrap() = Some(handle);
        }

        Ok(manager)
    }

    /// Lock the engine mutex, recovering from poison if a previous transcription panicked.
    fn lock_engine(&self) -> MutexGuard<'_, Option<LoadedEngine>> {
        self.engine.lock().unwrap_or_else(|poisoned| {
            warn!("Engine mutex was poisoned by a previous panic, recovering");
            poisoned.into_inner()
        })
    }

    pub fn is_model_loaded(&self) -> bool {
        // The engine may be leased out to the streaming worker (taken out of
        // the mutex). It's still loaded, just in use, so report true.
        self.lock_engine().is_some() || self.active_engine_lease.load(Ordering::Acquire) != 0
    }

    /// Accelerator changes should not disturb the current transcription. Mark
    /// the cached engine stale; the next model-use path reloads it with the
    /// latest settings.
    pub fn reload_model_on_next_use(&self) {
        self.reload_model_on_next_use.store(true, Ordering::Release);
    }

    /// Atomically check whether a model load is in progress and, if not, mark
    /// one as starting. Returns a [`LoadingGuard`] whose [`Drop`] impl will
    /// clear the flag and wake waiters. Returns `None` if a load is already in
    /// progress.
    pub fn try_start_loading(&self) -> Option<LoadingGuard> {
        let mut is_loading = self.is_loading.lock().unwrap();
        if *is_loading {
            return None;
        }
        *is_loading = true;
        Some(LoadingGuard {
            is_loading: self.is_loading.clone(),
            loading_condvar: self.loading_condvar.clone(),
        })
    }

    pub fn unload_model(&self) -> Result<()> {
        let unload_start = std::time::Instant::now();
        debug!("Starting to unload model");

        {
            let mut engine = self.lock_engine();
            // Dropping the engine frees all resources
            *engine = None;
        }
        {
            let mut current_model = self.current_model_id.lock().unwrap();
            *current_model = None;
        }

        // Emit unloaded event
        let _ = self.app_handle.emit(
            "model-state-changed",
            ModelStateEvent {
                event_type: "unloaded".to_string(),
                model_id: None,
                model_name: None,
                error: None,
            },
        );

        let unload_duration = unload_start.elapsed();
        debug!(
            "Model unloaded manually (took {}ms)",
            unload_duration.as_millis()
        );
        Ok(())
    }

    fn now_ms() -> u64 {
        SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64
    }

    /// Reset the idle timer to now.
    fn touch_activity(&self) {
        self.last_activity.store(Self::now_ms(), Ordering::Relaxed);
    }

    /// Unloads the model immediately if the setting is enabled and the model is loaded
    pub fn maybe_unload_immediately(&self, context: &str) {
        let settings = get_settings(&self.app_handle);
        if settings.model_unload_timeout == ModelUnloadTimeout::Immediately
            && self.is_model_loaded()
        {
            info!("Immediately unloading model after {}", context);
            if let Err(e) = self.unload_model() {
                warn!("Failed to immediately unload model: {}", e);
            }
        }
    }

    pub fn load_model(&self, model_id: &str) -> Result<()> {
        self.load_model_with_device(model_id, None)
    }

    /// Like [`load_model`](Self::load_model), but lets a caller hard-select the
    /// compute device for this one load by its `transcribe_cpp::devices()`
    /// registry index (the index shown by `--list-devices`). `None` keeps the
    /// persisted accelerator setting (which may be Auto). Only affects
    /// transcribe-cpp (whisper-family) models; the selection is not persisted.
    pub fn load_model_with_device(
        &self,
        model_id: &str,
        device_index: Option<usize>,
    ) -> Result<()> {
        apply_accelerator_settings(&self.app_handle);

        let load_start = std::time::Instant::now();
        debug!("Starting to load model: {}", model_id);

        // Emit loading started event
        let _ = self.app_handle.emit(
            "model-state-changed",
            ModelStateEvent {
                event_type: "loading_started".to_string(),
                model_id: Some(model_id.to_string()),
                model_name: None,
                error: None,
            },
        );

        let model_info = self
            .model_manager
            .get_model_info(model_id)
            .ok_or_else(|| anyhow::anyhow!("Model not found: {}", model_id))?;

        if !model_info.is_downloaded {
            let error_msg = "Model not downloaded";
            let _ = self.app_handle.emit(
                "model-state-changed",
                ModelStateEvent {
                    event_type: "loading_failed".to_string(),
                    model_id: Some(model_id.to_string()),
                    model_name: Some(model_info.name.clone()),
                    error: Some(error_msg.to_string()),
                },
            );
            return Err(anyhow::anyhow!(error_msg));
        }

        let model_path = self.model_manager.get_model_path(model_id)?;

        // Drop the current engine BEFORE building the new one so transcribe-cpp
        // frees the previous native context first — avoids holding two models at
        // once (peak memory on large GGUFs). Clear the id too: if the new load
        // fails, status should read "no loaded model", not the dropped engine.
        {
            let mut engine = self.lock_engine();
            *engine = None;
        }
        {
            let mut current_model = self.current_model_id.lock().unwrap();
            *current_model = None;
        }

        // Create appropriate engine based on model type
        let emit_loading_failed = |error_msg: &str| {
            let _ = self.app_handle.emit(
                "model-state-changed",
                ModelStateEvent {
                    event_type: "loading_failed".to_string(),
                    model_id: Some(model_id.to_string()),
                    model_name: Some(model_info.name.clone()),
                    error: Some(error_msg.to_string()),
                },
            );
        };

        let loaded_engine = match model_info.engine_type {
            EngineType::TranscribeCpp => {
                // The whisper backend is chosen at load time (transcribe-cpp has
                // no runtime global). With an explicit `device_index` (the
                // --device-index flag) hard-select that registered device;
                // otherwise re-read the persisted accelerator preference (so an
                // accelerator change marked for reload takes effect here).
                let (backend, device) = match device_index {
                    Some(index) => resolve_device_index(index).inspect_err(|e| {
                        emit_loading_failed(&e.to_string());
                    })?,
                    None => {
                        let settings = get_settings(&self.app_handle);
                        let accelerator = settings.transcribe_accelerator;
                        let device = resolve_gpu_device(
                            accelerator,
                            settings.transcribe_gpu_device.as_deref(),
                        );
                        // Backend::Auto accepts an exact GPU device. Without a
                        // valid exact device, backend selection handles the
                        // retired generic GPU state and host CPU guard.
                        let backend = if device.is_some() {
                            Backend::Auto
                        } else {
                            select_transcribe_backend(accelerator)
                        };
                        (backend, device)
                    }
                };
                let requested_device = device
                    .as_ref()
                    .map(transcribe_device_label)
                    .unwrap_or_else(|| "automatic".to_string());
                let model_options = ModelOptions { backend, device };
                let model = Model::load_with(&model_path, &model_options).map_err(|e| {
                    let error_msg = format!("Failed to load whisper model {}: {}", model_id, e);
                    emit_loading_failed(&error_msg);
                    anyhow::anyhow!(error_msg)
                })?;
                // The bound backend may differ from the request (e.g. CPU
                // fallback under Auto); log what actually loaded.
                let bound_backend = model.backend();
                let session = model.session().map_err(|e| {
                    let error_msg = format!(
                        "Failed to create session for whisper model {}: {}",
                        model_id, e
                    );
                    emit_loading_failed(&error_msg);
                    anyhow::anyhow!(error_msg)
                })?;
                // Reconcile the registry's advertised capabilities with the
                // loaded model's real ones (GGUF metadata) so badges/gating
                // reflect runtime truth, not the pre-download probe. The
                // load-completed event below triggers the frontend refresh.
                let caps = session.model().capabilities();
                self.model_manager.set_runtime_capabilities(
                    model_id,
                    caps.supports_streaming,
                    caps.supports_translate,
                    caps.supports_language_detect,
                    caps.languages.clone(),
                );
                let bound_device = model
                    .device()
                    .map(|device| transcribe_device_label(&device))
                    .unwrap_or_else(|_| "unknown".to_string());
                info!(
                    "Loaded whisper model '{}' (requested {:?}, requested device '{}', \
                     bound backend '{}', bound device '{}', supports_streaming={}, \
                     supports_translate={}, supports_language_detect={})",
                    model_id,
                    backend,
                    requested_device,
                    bound_backend,
                    bound_device,
                    caps.supports_streaming,
                    caps.supports_translate,
                    caps.supports_language_detect
                );
                LoadedEngine::TranscribeCpp(session)
            }
            EngineType::Parakeet => {
                let engine =
                    ParakeetModel::load(&model_path, &Quantization::Int8).map_err(|e| {
                        let error_msg =
                            format!("Failed to load parakeet model {}: {}", model_id, e);
                        emit_loading_failed(&error_msg);
                        anyhow::anyhow!(error_msg)
                    })?;
                LoadedEngine::Parakeet(engine)
            }
            EngineType::Moonshine => {
                let engine = MoonshineModel::load(
                    &model_path,
                    MoonshineVariant::Base,
                    &Quantization::default(),
                )
                .map_err(|e| {
                    let error_msg = format!("Failed to load moonshine model {}: {}", model_id, e);
                    emit_loading_failed(&error_msg);
                    anyhow::anyhow!(error_msg)
                })?;
                LoadedEngine::Moonshine(engine)
            }
            EngineType::MoonshineStreaming => {
                let engine = StreamingModel::load(&model_path, 0, &Quantization::default())
                    .map_err(|e| {
                        let error_msg = format!(
                            "Failed to load moonshine streaming model {}: {}",
                            model_id, e
                        );
                        emit_loading_failed(&error_msg);
                        anyhow::anyhow!(error_msg)
                    })?;
                LoadedEngine::MoonshineStreaming(engine)
            }
            EngineType::SenseVoice => {
                let engine =
                    SenseVoiceModel::load(&model_path, &Quantization::Int8).map_err(|e| {
                        let error_msg =
                            format!("Failed to load SenseVoice model {}: {}", model_id, e);
                        emit_loading_failed(&error_msg);
                        anyhow::anyhow!(error_msg)
                    })?;
                LoadedEngine::SenseVoice(engine)
            }
            EngineType::GigaAM => {
                let engine = GigaAMModel::load(&model_path, &Quantization::Int8).map_err(|e| {
                    let error_msg = format!("Failed to load gigaam model {}: {}", model_id, e);
                    emit_loading_failed(&error_msg);
                    anyhow::anyhow!(error_msg)
                })?;
                LoadedEngine::GigaAM(engine)
            }
            EngineType::Canary => {
                let engine = CanaryModel::load(&model_path, &Quantization::Int8).map_err(|e| {
                    let error_msg = format!("Failed to load canary model {}: {}", model_id, e);
                    emit_loading_failed(&error_msg);
                    anyhow::anyhow!(error_msg)
                })?;
                LoadedEngine::Canary(engine)
            }
            EngineType::Cohere => {
                let engine = CohereModel::load(&model_path, &Quantization::Int8).map_err(|e| {
                    let error_msg = format!("Failed to load cohere model {}: {}", model_id, e);
                    emit_loading_failed(&error_msg);
                    anyhow::anyhow!(error_msg)
                })?;
                LoadedEngine::Cohere(engine)
            }
        };

        // Update the current engine and model ID
        {
            let mut engine = self.lock_engine();
            *engine = Some(loaded_engine);
        }
        {
            let mut current_model = self.current_model_id.lock().unwrap();
            *current_model = Some(model_id.to_string());
        }

        // Reset idle timer so the watcher doesn't immediately unload a just-loaded model
        self.touch_activity();

        // Emit loading completed event
        let _ = self.app_handle.emit(
            "model-state-changed",
            ModelStateEvent {
                event_type: "loading_completed".to_string(),
                model_id: Some(model_id.to_string()),
                model_name: Some(model_info.name.clone()),
                error: None,
            },
        );

        let load_duration = load_start.elapsed();
        debug!(
            "Successfully loaded transcription model: {} (took {}ms)",
            model_id,
            load_duration.as_millis()
        );
        Ok(())
    }

    /// Kicks off the model loading in a background thread if it's not already loaded
    pub fn initiate_model_load(&self) {
        let mut is_loading = self.is_loading.lock().unwrap();
        if *is_loading {
            return;
        }

        let reload_pending = self.reload_model_on_next_use.load(Ordering::Acquire);
        if !reload_pending && self.is_model_loaded() {
            return;
        }

        *is_loading = true;
        let self_clone = self.clone();
        thread::spawn(move || {
            if reload_pending {
                self_clone
                    .reload_model_on_next_use
                    .store(false, Ordering::Release);
            }
            let settings = get_settings(&self_clone.app_handle);
            if let Err(e) = self_clone.load_model(&settings.selected_model) {
                error!("Failed to load model: {}", e);
            }
            let mut is_loading = self_clone.is_loading.lock().unwrap();
            *is_loading = false;
            self_clone.loading_condvar.notify_all();
        });
    }

    pub fn get_current_model(&self) -> Option<String> {
        let current_model = self.current_model_id.lock().unwrap();
        current_model.clone()
    }

    /// The compute backend the currently-loaded engine is bound to, for
    /// diagnostics (e.g. confirming `--device-index` actually bound a GPU rather
    /// than falling back to CPU/auto). transcribe-cpp (whisper-family) reports
    /// its real backend string; ONNX engines report "onnx"; `None` when no
    /// model is loaded.
    pub fn current_backend(&self) -> Option<String> {
        match self.lock_engine().as_ref() {
            Some(LoadedEngine::TranscribeCpp(session)) => {
                Some(session.model().backend().to_string())
            }
            Some(_) => Some("onnx".to_string()),
            None => None,
        }
    }

    /// Whether a live streaming run is currently in flight.
    pub fn is_streaming(&self) -> bool {
        self.stream_active.load(Ordering::Acquire)
    }

    /// Shared handle to the stream router, used by the audio recorder to feed
    /// real-time frames without going through Tauri state on every frame.
    pub fn stream_router(&self) -> Arc<StreamRouter> {
        Arc::clone(&self.router)
    }

    /// Begin a live streaming transcription on the held engine's session.
    /// Audio frames pushed via [`StreamRouter::feed`] (captured directly by the
    /// audio recorder) are decoded incrementally and emitted to the overlay as
    /// [`StreamTextEvent`].
    ///
    /// Non-blocking: spawns a worker that waits for any in-progress model load,
    /// verifies the model supports streaming, then begins the stream. If the
    /// model can't stream, the worker idles until finalize/cancel and reports
    /// `None` so the caller falls back to batch transcription. Frames sent
    /// before the stream begins queue on the channel and are not lost.
    pub fn start_stream(&self) {
        if self.router.is_open() || self.active_stream_worker.load(Ordering::Acquire) != 0 {
            warn!("start_stream called while a stream worker is already active");
            return;
        }
        let worker_id = self.next_stream_worker_id.fetch_add(1, Ordering::Relaxed);
        if self
            .active_stream_worker
            .compare_exchange(0, worker_id, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            warn!("start_stream lost a race with another stream worker");
            return;
        }
        let rx = self.router.open();
        self.stream_active.store(false, Ordering::Release);

        let manager = self.clone();
        thread::spawn(move || manager.run_stream_worker(rx, worker_id));
    }

    fn run_stream_worker(&self, rx: mpsc::Receiver<StreamCmd>, worker_id: u64) {
        let _worker = StreamWorkerGuard {
            worker_id,
            active_stream_worker: Arc::clone(&self.active_stream_worker),
            active_engine_lease: Arc::clone(&self.active_engine_lease),
            stream_active: Arc::clone(&self.stream_active),
        };

        // Wait for any in-progress model load to finish (start_stream races the
        // background load kicked off when recording starts).
        {
            let mut is_loading = self.is_loading.lock().unwrap();
            while *is_loading {
                is_loading = self.loading_condvar.wait(is_loading).unwrap();
            }
        }

        let model_id = self.get_current_model().unwrap_or_default();

        // Take the engine out of the mutex so we own it during streaming,
        // structurally excluding any concurrent batch transcription (which
        // transcribe-cpp's compute_lock would refuse anyway). Returned when the
        // worker exits, or dropped if the model was switched/unloaded mid-stream.
        if self
            .active_engine_lease
            .compare_exchange(0, worker_id, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            warn!("Live preview: another worker already holds the transcription engine");
            self.router.clear();
            drain_until_finalize(rx);
            return;
        }
        let mut engine = match self.lock_engine().take() {
            Some(e) => e,
            None => {
                info!(
                    "Live preview: model '{}' was unloaded before streaming could begin; \
                     falling back to batch transcription",
                    model_id
                );
                let _ = self.active_engine_lease.compare_exchange(
                    worker_id,
                    0,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                );
                self.router.clear();
                drain_until_finalize(rx);
                return;
            }
        };

        // Only transcribe-cpp models expose streaming; ONNX engines fall back to
        // batch. The loaded session (not the ModelManager copy) is the source of
        // truth for run-path capabilities.
        let (supports_streaming, supports_translate, languages) = match &engine {
            LoadedEngine::TranscribeCpp(session) => {
                let model = session.model();
                let caps = model.capabilities();
                info!(
                    "Live preview: model '{}' arch='{}' variant='{}' supports_streaming={} \
                     supports_translate={} languages={:?}",
                    model_id,
                    model.arch(),
                    model.variant(),
                    caps.supports_streaming,
                    caps.supports_translate,
                    caps.languages,
                );
                (
                    caps.supports_streaming,
                    caps.supports_translate,
                    caps.languages,
                )
            }
            _ => {
                info!(
                    "Live preview: model '{}' is not a transcribe-cpp model; \
                     streaming is unavailable, using batch transcription",
                    model_id
                );
                (false, false, Vec::new())
            }
        };

        if !supports_streaming {
            self.return_engine(engine, &model_id);
            self.router.clear();
            drain_until_finalize(rx);
            return;
        }

        // Build run options mirroring the offline transcribe-cpp path: task +
        // language gated against what the model actually advertises.
        let settings = get_settings(&self.app_handle);
        let effective_language =
            effective_language_for_model(&settings, self.model_manager.as_ref(), &model_id);
        let run_plan = transcribe_cpp_run_plan(
            settings.translate_to_english,
            &effective_language,
            &languages,
            supports_translate,
        );
        let output_language = resolve_output_language_evidence(
            &settings,
            run_plan.language.as_deref(),
            &languages,
            run_plan.target_language.as_deref() == Some("en"),
        );
        let run_options = RunOptions {
            task: run_plan.task,
            language: run_plan.language,
            target_language: run_plan.target_language,
            ..Default::default()
        };

        // Run the stream on the held session. The Stream borrows the session
        // (and thus the engine) for its lifetime, so the feed/finalize loop
        // lives in a labeled block — when it exits, the borrow is released and
        // the engine can be moved into return_engine().
        let mut finalize_reply: Option<mpsc::Sender<Option<FinalizedStreamText>>> = None;
        let mut finalize_result: Option<Option<FinalizedStreamText>> = None;
        let stream_started = 'stream: {
            let session = match &mut engine {
                LoadedEngine::TranscribeCpp(s) => s,
                _ => break 'stream false,
            };

            // Read the backend string before beginning the stream — the
            // `Stream` borrows `session` mutably for its lifetime, so we can't
            // call `session.model()` once it exists.
            let backend = session.model().backend();

            // StreamOptions::default() uses CommitPolicy::Auto and lets the
            // family pick its own streaming strategy (no family-specific ext).
            let mut stream = match session.stream(&run_options, &StreamOptions::default()) {
                Ok(s) => s,
                Err(e) => {
                    error!("Failed to begin stream: {}", e);
                    break 'stream false;
                }
            };

            self.stream_active.store(true, Ordering::Release);
            self.touch_activity();
            info!(
                "Live streaming transcription started (model '{}', backend '{}')",
                model_id, backend
            );

            let mut perf = StreamPerf::new();
            while let Ok(cmd) = rx.recv() {
                match cmd {
                    StreamCmd::Feed(pcm) => {
                        self.touch_activity();
                        perf.record_feed(pcm.len());
                        let feed_start = Instant::now();
                        match stream.feed(&pcm) {
                            Ok(update) => {
                                perf.record_compute(feed_start.elapsed());
                                perf.record_update(
                                    update.revision,
                                    update.input_received_ms,
                                    update.audio_committed_ms,
                                    update.buffered_ms,
                                );
                                if update.committed_changed || update.tentative_changed {
                                    let text = stream.text();
                                    perf.record_emit();
                                    self.emit_stream_text(&text.committed, &text.tentative);
                                }
                                perf.maybe_log();
                            }
                            Err(e) => {
                                perf.record_compute(feed_start.elapsed());
                                warn!("stream feed failed: {}", e);
                            }
                        }
                    }
                    StreamCmd::Finalize(reply) => {
                        let finalize_start = Instant::now();
                        let result = match stream.finalize() {
                            // After finalize the committed prefix holds the full
                            // text; display() = committed + tentative is the safe read.
                            Ok(update) => {
                                perf.record_compute(finalize_start.elapsed());
                                perf.record_update(
                                    update.revision,
                                    update.input_received_ms,
                                    update.audio_committed_ms,
                                    update.buffered_ms,
                                );
                                // In auto mode the model's own LID is the best
                                // remaining evidence; the snapshot is only
                                // materialized when it can change the outcome.
                                let output_language = match &output_language {
                                    OutputLanguageEvidence::Unknown => {
                                        with_model_detected_language(
                                            OutputLanguageEvidence::Unknown,
                                            stream.snapshot().language,
                                        )
                                    }
                                    resolved => resolved.clone(),
                                };
                                Some(FinalizedStreamText {
                                    text: stream.text().full,
                                    output_language,
                                    supported_languages: languages.clone(),
                                })
                            }
                            Err(e) => {
                                perf.record_compute(finalize_start.elapsed());
                                error!(
                                    "stream finalize failed: {}; falling back to batch transcription",
                                    e
                                );
                                None
                            }
                        };
                        let chars = match &result {
                            Some(finalized) => finalized.text.len(),
                            _ => 0,
                        };
                        perf.log_finalized(chars);
                        finalize_reply = Some(reply);
                        finalize_result = Some(result);
                        break;
                    }
                    StreamCmd::Cancel => {
                        stream.reset();
                        break;
                    }
                }
            }

            true
        };
        // `stream` + the `&mut engine` borrow are released here.

        if !stream_started {
            // Stream never began (model doesn't support streaming or begin
            // failed); drain so the finalize handshake still completes and the
            // caller falls back to batch transcription. Return the engine first
            // so the fallback can immediately use it.
            self.return_engine(engine, &model_id);
            drain_until_finalize(rx);
            return;
        }

        self.return_engine(engine, &model_id);
        if let (Some(reply), Some(result)) = (finalize_reply, finalize_result) {
            let _ = reply.send(result);
        }
        // `_worker` drops here, clearing this worker's active/lease flags after
        // the engine has been returned to the pool.
    }

    /// Return the leased engine to the mutex, unless the model was switched or
    /// unloaded during transcription (in which case the stale engine is dropped).
    fn return_engine(&self, engine: LoadedEngine, expected_model_id: &str) {
        let still_current =
            self.current_model_id.lock().unwrap().as_deref() == Some(expected_model_id);
        if still_current {
            *self.lock_engine() = Some(engine);
        } else {
            info!(
                "Model changed/unloaded during transcription; dropping stale engine (was '{}')",
                expected_model_id
            );
            // `engine` drops here, freeing its resources.
        }
    }

    /// Flush the active stream and return its final, post-filtered text.
    ///
    /// `Ok(None)` means no usable stream was active and the caller may fall back
    /// to batch transcription. `Err` means finalize itself failed or timed out.
    /// A timeout may still leave the worker holding the engine, so callers
    /// should surface it instead of immediately starting a batch fallback.
    pub fn finalize_stream(&self) -> Result<Option<String>> {
        let Some(tx) = self.router.take() else {
            return Ok(None);
        };
        let (reply_tx, reply_rx) = mpsc::channel();
        if tx.send(StreamCmd::Finalize(reply_tx)).is_err() {
            return Ok(None);
        }
        let finalized = match reply_rx.recv_timeout(STREAM_FINALIZE_REPLY_TIMEOUT) {
            Ok(Some(finalized)) => finalized,
            Ok(None) => return Ok(None),
            Err(mpsc::RecvTimeoutError::Disconnected) => return Ok(None),
            Err(mpsc::RecvTimeoutError::Timeout) => {
                self.stream_active.store(false, Ordering::Release);
                return Err(anyhow::anyhow!(
                    "Timed out waiting {:?} for live transcription to finalize",
                    STREAM_FINALIZE_REPLY_TIMEOUT
                ));
            }
        };

        let settings = get_settings(&self.app_handle);
        // Streaming models do not receive a decode prompt, so custom words
        // reach the text only through the shared fuzzy post-correction path.
        let filtered = post_process_transcription_text(
            finalized.text,
            &settings,
            &finalized.output_language,
            &finalized.supported_languages,
        );

        self.maybe_unload_immediately("streaming transcription");
        Ok(Some(filtered))
    }

    /// Abandon any active stream without producing text (e.g. on cancel).
    pub fn cancel_stream(&self) {
        if let Some(tx) = self.router.take() {
            let _ = tx.send(StreamCmd::Cancel);
        }
        self.stream_active.store(false, Ordering::Release);
    }

    /// Emit a working-phase event to the streaming overlay (spinner + label).
    pub fn emit_stream_working(&self, kind: StreamWorkKind) {
        let _ = StreamPhaseEvent {
            phase: StreamPhase::Working,
            kind: Some(kind),
        }
        .emit(&self.app_handle);
    }

    fn emit_stream_text(&self, committed: &str, tentative: &str) {
        let _ = StreamTextEvent {
            committed: committed.to_string(),
            tentative: tentative.to_string(),
        }
        .emit(&self.app_handle);
    }

    pub fn transcribe(&self, audio: Vec<f32>) -> Result<String> {
        #[cfg(debug_assertions)]
        if std::env::var("HANDY_FORCE_TRANSCRIPTION_FAILURE").is_ok() {
            return Err(anyhow::anyhow!(
                "Simulated transcription failure (HANDY_FORCE_TRANSCRIPTION_FAILURE)"
            ));
        }

        // Update last activity timestamp
        self.touch_activity();

        let st = std::time::Instant::now();
        let audio_len = audio.len();

        debug!("Audio vector length: {}", audio_len);

        if audio.is_empty() {
            debug!("Empty audio vector");
            self.maybe_unload_immediately("empty audio");
            return Ok(String::new());
        }

        // Check if model is loaded, if not try to load it
        {
            // If the model is loading, wait for it to complete.
            let mut is_loading = self.is_loading.lock().unwrap();
            while *is_loading {
                is_loading = self.loading_condvar.wait(is_loading).unwrap();
            }

            let engine_guard = self.lock_engine();
            if engine_guard.is_none() {
                return Err(anyhow::anyhow!("Model is not loaded for transcription."));
            }
        }

        // Get current settings for configuration
        let settings = get_settings(&self.app_handle);

        // Validate selected language against the model's supported languages.
        // If the language isn't supported, fall back to "auto" to prevent errors.
        // Validate against the model that's actually loaded (which can differ
        // from settings.selected_model when a caller loaded a specific model —
        // e.g. the --transcribe-file path's --model), not the persisted
        // selection.
        let active_model = self
            .get_current_model()
            .unwrap_or_else(|| settings.selected_model.clone());
        // Resolve the persisted language *intent* into the language this model
        // will actually use. The coercion is capability-aware (a must-pick model
        // never receives "auto") and computed fresh here — it is never written
        // back to settings, so the intent survives switching models and back.
        let validated_language =
            effective_language_for_model(&settings, self.model_manager.as_ref(), &active_model);
        if validated_language != settings.selected_language {
            debug!(
                "Language intent '{}' resolved to '{}' for model '{}'",
                settings.selected_language, validated_language, active_model
            );
        }

        // Filled in once the loaded model's architecture is known, since what
        // it can receive as context — and how much — is a property of the
        // model, not a global constant. `context_words` outlives the engine
        // block because post-correction needs to know what the model already
        // saw.
        let mut context_biasing = ContextBiasing::NONE;
        let context_words: Vec<String>;
        let mut fitted_context: Option<Vec<String>> = None;
        // Set when a run with context failed and was repeated without it.
        let mut context_fell_back = false;

        // Perform transcription with the appropriate engine.
        // We use catch_unwind to prevent engine panics from poisoning the mutex,
        // which would make the app hang indefinitely on subsequent operations.
        let (result, output_language, model_languages) = {
            let mut engine_guard = self.lock_engine();

            // Take the engine out so we own it during transcription.
            // If the engine panics, we simply don't put it back (effectively unloading it)
            // instead of poisoning the mutex.
            let mut engine = match engine_guard.take() {
                Some(e) => e,
                None => {
                    return Err(anyhow::anyhow!(
                        "Model failed to load after auto-load attempt. Please check your model settings."
                    ));
                }
            };

            // Release the lock before transcribing — no mutex held during the engine call
            drop(engine_guard);

            // Probe live transcribe-cpp capabilities once (cheap GGUF-metadata
            // reads); the loaded session is the source of truth, not the
            // ModelManager copy. The whisper run extension is kind-tagged, so
            // non-whisper archs (parakeet, voxtral, …) reject it with
            // INVALID_ARG; attach it — and translate — only where supported.
            let mut model_supports_translate = false;
            let mut model_languages = self
                .model_manager
                .get_model_info(&active_model)
                .map(|info| info.supported_languages)
                .unwrap_or_default();
            let mut output_was_translated = false;
            let mut applied_language_hint: Option<String> = None;
            let mut model_detected_language: Option<String> = None;
            if let LoadedEngine::TranscribeCpp(session) = &engine {
                let model = session.model();
                let caps = model.capabilities();
                // Informational only: the whisper extension is gated on the
                // architecture inside `transcribe_cpp_context_biasing`, because
                // a non-whisper arch (Voxtral Small) can advertise
                // Feature::InitialPrompt and still reject the whisper-kind
                // extension with INVALID_ARG (#1601).
                let model_takes_initial_prompt = model.supports(Feature::InitialPrompt);
                context_biasing =
                    transcribe_cpp_context_biasing(&model.arch(), model.supports(Feature::Context));
                model_supports_translate = caps.supports_translate;
                model_languages = caps.languages;
                debug!(
                    "transcribe-cpp model '{}' on '{}': initial_prompt={}, translate={}, languages={:?}",
                    settings.selected_model,
                    model.backend(),
                    model_takes_initial_prompt,
                    model_supports_translate,
                    model_languages
                );

                // Select by the term budget, then fit what was selected into the
                // token room this model and this clip actually leave. Terms that
                // do not fit are deferred to fuzzy correction rather than handed
                // to a backend that would silently cut them.
                let selected = context_biasing.terms(&settings);
                if !selected.is_empty() {
                    let (n_ctx, max_audio_ms) = session
                        .limits()
                        .map(|l| (l.effective_n_ctx as i64, l.effective_max_audio_ms))
                        .unwrap_or((0, 0));
                    let limit = ContextTokenLimit::for_channel(
                        context_biasing.channel,
                        n_ctx,
                        max_audio_ms,
                    );
                    let audio_ms = (audio.len() as u64 * 1000) / 16_000;
                    let tokenize = |text: &str| model.tokenize(text).ok().map(|t| t.len());
                    let fit = fit_context(&selected, limit, audio_ms, &tokenize);
                    let pool = crate::dictionaries::full_vocabulary(&settings).len();
                    info!(
                        "context: pool={} selected={} sent={} deferred={} tokens={} room={} audio_tokens={} verified={}",
                        pool,
                        selected.len(),
                        fit.sent.len(),
                        fit.deferred.len(),
                        fit.context_tokens.map_or("?".into(), |t| t.to_string()),
                        fit.context_room.map_or("?".into(), |t| t.to_string()),
                        fit.audio_tokens.map_or("-".into(), |t| t.to_string()),
                        fit.verified()
                    );
                    if !fit.deferred.is_empty() {
                        debug!("context deferred to fuzzy correction: {:?}", fit.deferred);
                    }
                    remember_context_usage(LastContextUsage {
                        model_arch: model.arch(),
                        selected: selected.len() as u32,
                        sent: fit.sent.len() as u32,
                        deferred: fit.deferred.len() as u32,
                        context_tokens: fit.context_tokens.map(|t| t as u32),
                        context_room: fit.context_room.map(|t| t as u32),
                        pool: pool as u32,
                        fell_back_without_context: false,
                    });
                    fitted_context = Some(fit.sent);
                }
            }

            // Terms the loaded model actually receives: budgeted for its
            // architecture, fairly shared across the active dictionary modules
            // and fitted into its token window. Empty for a model that accepts
            // none — its vocabulary then reaches fuzzy post-correction in full.
            context_words = fitted_context.take().unwrap_or_default();

            let transcribe_result = catch_unwind(AssertUnwindSafe(|| -> Result<String> {
                match &mut engine {
                    LoadedEngine::TranscribeCpp(session) => {
                        let run_plan = transcribe_cpp_run_plan(
                            settings.translate_to_english,
                            &validated_language,
                            &model_languages,
                            model_supports_translate,
                        );
                        output_was_translated = run_plan.target_language.as_deref() == Some("en");
                        applied_language_hint = run_plan.language.clone();

                        let mut run_options = RunOptions {
                            task: run_plan.task,
                            language: run_plan.language,
                            target_language: run_plan.target_language,
                            ..Default::default()
                        };
                        apply_context_to_run_options(
                            context_biasing,
                            &context_words,
                            &mut run_options,
                        );

                        debug!(
                            "transcribe-cpp run: task={:?}, language={:?}, channel={:?}, terms={}",
                            run_options.task,
                            run_options.language,
                            context_biasing.channel,
                            context_words.len()
                        );

                        let (outcome, fell_back) =
                            run_with_context_fallback(session, &audio, &run_options);
                        context_fell_back = fell_back;
                        outcome
                            .map(|t| {
                                // Whisper's audio-based LID (auto mode only;
                                // `None` when a language hint was passed).
                                model_detected_language = t.language;
                                t.text
                            })
                            .map_err(|e| {
                                anyhow::anyhow!("transcribe-cpp transcription failed: {}", e)
                            })
                    }
                    LoadedEngine::Parakeet(parakeet_engine) => {
                        let params = ParakeetParams {
                            timestamp_granularity: Some(TimestampGranularity::Segment),
                            ..Default::default()
                        };
                        parakeet_engine
                            .transcribe_with(&audio, &params)
                            .map(|r| r.text)
                            .map_err(|e| anyhow::anyhow!("Parakeet transcription failed: {}", e))
                    }
                    LoadedEngine::Moonshine(moonshine_engine) => moonshine_engine
                        .transcribe(&audio, &TranscribeOptions::default())
                        .map(|r| r.text)
                        .map_err(|e| anyhow::anyhow!("Moonshine transcription failed: {}", e)),
                    LoadedEngine::MoonshineStreaming(streaming_engine) => streaming_engine
                        .transcribe(&audio, &TranscribeOptions::default())
                        .map(|r| r.text)
                        .map_err(|e| {
                            anyhow::anyhow!("Moonshine streaming transcription failed: {}", e)
                        }),
                    LoadedEngine::SenseVoice(sense_voice_engine) => {
                        let language = match normalize_cjk_language(&validated_language) {
                            "zh" => Some("zh".to_string()),
                            "en" => Some("en".to_string()),
                            "ja" => Some("ja".to_string()),
                            "ko" => Some("ko".to_string()),
                            "yue" => Some("yue".to_string()),
                            _ => None,
                        };
                        applied_language_hint = language.clone();
                        let params = SenseVoiceParams {
                            language,
                            use_itn: Some(true),
                        };
                        sense_voice_engine
                            .transcribe_with(&audio, &params)
                            .map(|r| r.text)
                            .map_err(|e| anyhow::anyhow!("SenseVoice transcription failed: {}", e))
                    }
                    LoadedEngine::GigaAM(gigaam_engine) => gigaam_engine
                        .transcribe(&audio, &TranscribeOptions::default())
                        .map(|r| r.text)
                        .map_err(|e| anyhow::anyhow!("GigaAM transcription failed: {}", e)),
                    LoadedEngine::Canary(canary_engine) => {
                        output_was_translated = settings.translate_to_english;
                        let lang = if validated_language == "auto" {
                            None
                        } else {
                            Some(validated_language.clone())
                        };
                        applied_language_hint = lang.clone();
                        let options = TranscribeOptions {
                            language: lang,
                            translate: settings.translate_to_english,
                            ..Default::default()
                        };
                        canary_engine
                            .transcribe(&audio, &options)
                            .map(|r| r.text)
                            .map_err(|e| anyhow::anyhow!("Canary transcription failed: {}", e))
                    }
                    LoadedEngine::Cohere(cohere_engine) => {
                        let lang = if validated_language == "auto" {
                            None
                        } else {
                            Some(normalize_cjk_language(&validated_language).to_string())
                        };
                        applied_language_hint = lang.clone();
                        let options = TranscribeOptions {
                            language: lang,
                            ..Default::default()
                        };
                        cohere_engine
                            .transcribe(&audio, &options)
                            .map(|r| r.text)
                            .map_err(|e| anyhow::anyhow!("Cohere transcription failed: {}", e))
                    }
                }
            }));

            let text = match transcribe_result {
                Ok(inner_result) => {
                    // Success or normal error: return the engine unless a model
                    // switch/unload invalidated it while it was in use.
                    self.return_engine(engine, &active_model);
                    inner_result?
                }
                Err(panic_payload) => {
                    // Engine panicked — do NOT put it back (it's in an unknown state).
                    // The engine is dropped here, effectively unloading it.
                    let panic_msg = panic_payload_message(panic_payload.as_ref());
                    error!(
                        "Transcription engine panicked: {}. Model has been unloaded.",
                        panic_msg
                    );

                    // Clear the model ID so it will be reloaded on next attempt
                    {
                        let mut current_model = self
                            .current_model_id
                            .lock()
                            .unwrap_or_else(|e| e.into_inner());
                        *current_model = None;
                    }

                    let _ = self.app_handle.emit(
                        "model-state-changed",
                        ModelStateEvent {
                            event_type: "unloaded".to_string(),
                            model_id: None,
                            model_name: None,
                            error: Some(format!("Engine panicked: {}", panic_msg)),
                        },
                    );

                    return Err(anyhow::anyhow!(
                        "Transcription engine panicked: {}. The model has been unloaded and will reload on next attempt.",
                        panic_msg
                    ));
                }
            };

            let output_language = with_model_detected_language(
                resolve_output_language_evidence(
                    &settings,
                    applied_language_hint.as_deref(),
                    &model_languages,
                    output_was_translated,
                ),
                model_detected_language,
            );
            debug!("Output language evidence: {:?}", output_language);

            (text, output_language, model_languages)
        };

        if context_fell_back {
            mark_context_fallback();
        }
        // Fuzzy correction always works against the *whole* active pool,
        // including the terms the model already received as context: a term in
        // the prompt is no guarantee the model used it. Whether a run sent
        // context, sent none, or fell back after a failure therefore makes no
        // difference here.
        let filtered_result =
            post_process_transcription_text(result, &settings, &output_language, &model_languages);

        let et = std::time::Instant::now();
        let translation_note = if settings.translate_to_english {
            " (translated)"
        } else {
            ""
        };
        // Real-time factor. Input PCM is 16 kHz mono, so audio length in seconds
        // is samples / 16000. `speedup` is audio_secs / elapsed_secs — e.g. 4.00x
        // means transcribed 4x faster than real time
        let elapsed_secs = (et - st).as_secs_f64();
        let audio_secs = audio_len as f64 / 16_000.0;
        let speedup = real_time_factor(audio_secs, elapsed_secs);
        info!(
            "Transcription completed in {:.2}s for {:.2}s of audio ({:.2}x real-time){}",
            elapsed_secs, audio_secs, speedup, translation_note
        );

        let final_result = filtered_result;

        if final_result.is_empty() {
            info!("Transcription result is empty");
        } else {
            info!(
                "Transcription result: {}",
                crate::utils::redact_text(&final_result)
            );
        }

        self.maybe_unload_immediately("transcription");

        Ok(final_result)
    }
}

struct StreamPerf {
    feed_count: u64,
    emit_count: u64,
    streamed_samples: u64,
    stream_compute_elapsed: Duration,
    last_log: Instant,
    latest_revision: i32,
    latest_input_received_ms: i64,
    latest_audio_committed_ms: i64,
    latest_buffered_ms: i64,
}

impl StreamPerf {
    fn new() -> Self {
        Self {
            feed_count: 0,
            emit_count: 0,
            streamed_samples: 0,
            stream_compute_elapsed: Duration::ZERO,
            last_log: Instant::now(),
            latest_revision: 0,
            latest_input_received_ms: 0,
            latest_audio_committed_ms: 0,
            latest_buffered_ms: 0,
        }
    }

    fn record_feed(&mut self, samples: usize) {
        self.feed_count += 1;
        self.streamed_samples += samples as u64;
    }

    fn record_compute(&mut self, elapsed: Duration) {
        self.stream_compute_elapsed += elapsed;
    }

    fn record_update(
        &mut self,
        revision: i32,
        input_received_ms: i64,
        audio_committed_ms: i64,
        buffered_ms: i64,
    ) {
        self.latest_revision = revision;
        self.latest_input_received_ms = input_received_ms;
        self.latest_audio_committed_ms = audio_committed_ms;
        self.latest_buffered_ms = buffered_ms;
    }

    fn record_emit(&mut self) {
        self.emit_count += 1;
    }

    fn maybe_log(&mut self) {
        if self.last_log.elapsed() < STREAM_PERF_LOG_INTERVAL {
            return;
        }

        let audio_secs = self.audio_secs();
        let compute_secs = self.compute_secs();
        debug!(
            "Live preview perf: {:.2}s streamed audio, {:.2}s model compute ({:.2}x real-time), \
             input_received={:.2}s, committed_audio={:.2}s, buffered={}ms, revision={}, \
             {} frames fed, {} updates emitted",
            audio_secs,
            compute_secs,
            real_time_factor(audio_secs, compute_secs),
            self.latest_input_received_ms as f64 / 1000.0,
            self.latest_audio_committed_ms as f64 / 1000.0,
            self.latest_buffered_ms,
            self.latest_revision,
            self.feed_count,
            self.emit_count,
        );
        self.last_log = Instant::now();
    }

    fn log_finalized(&self, chars: usize) {
        let audio_secs = self.audio_secs();
        let compute_secs = self.compute_secs();
        info!(
            "Live preview finalized in {:.2}s model compute for {:.2}s streamed audio ({:.2}x real-time): \
             input_received={:.2}s, committed_audio={:.2}s, buffered={}ms, revision={}, \
             {} frames fed, {} updates emitted, {} chars",
            compute_secs,
            audio_secs,
            real_time_factor(audio_secs, compute_secs),
            self.latest_input_received_ms as f64 / 1000.0,
            self.latest_audio_committed_ms as f64 / 1000.0,
            self.latest_buffered_ms,
            self.latest_revision,
            self.feed_count,
            self.emit_count,
            chars
        );
    }

    fn audio_secs(&self) -> f64 {
        self.streamed_samples as f64 / 16_000.0
    }

    fn compute_secs(&self) -> f64 {
        self.stream_compute_elapsed.as_secs_f64()
    }
}

fn real_time_factor(audio_secs: f64, compute_secs: f64) -> f64 {
    if compute_secs > 0.0 {
        audio_secs / compute_secs
    } else {
        0.0
    }
}

fn normalize_cjk_language(language: &str) -> &str {
    match language {
        "zh-Hans" | "zh-Hant" => "zh",
        other => other,
    }
}

fn base_language_code(language: &str) -> &str {
    language.split(&['-', '_'][..]).next().unwrap_or(language)
}

/// Resolve the persisted language intent into the language a specific model can
/// use without writing the coerced value back to settings.
fn effective_language_for_model(
    settings: &AppSettings,
    model_manager: &ModelManager,
    model_id: &str,
) -> String {
    match model_manager.get_model_info(model_id) {
        Some(info) => crate::managers::model::effective_language(
            &settings.selected_language,
            &info.supported_languages,
            info.supports_language_detection,
        ),
        None => settings.selected_language.clone(),
    }
}

/// Resolve how confidently Handy knows the language of the text produced by a
/// transcription run. The UI language is deliberately not part of this
/// decision.
fn resolve_output_language_evidence(
    settings: &AppSettings,
    applied_language_hint: Option<&str>,
    supported_languages: &[String],
    translated_to_english: bool,
) -> OutputLanguageEvidence {
    if translated_to_english {
        return OutputLanguageEvidence::TranslatedToEnglish;
    }

    // Stored language intent is only evidence when this specific engine run
    // actually received the hint. Some multilingual engines (notably Parakeet
    // V3) always auto-detect and ignore Handy's selection; transcribe-cpp also
    // drops a requested hint when the loaded model does not advertise it.
    if let Some(language) = applied_language_hint.filter(|lang| !lang.is_empty() && *lang != "auto")
    {
        if settings.selected_language != "auto"
            && base_language_code(&settings.selected_language) == base_language_code(language)
        {
            return OutputLanguageEvidence::UserSelected(language.to_string());
        }

        // The engine may have required a concrete fallback even though the
        // user's persisted language was auto or unsupported.
        return OutputLanguageEvidence::ModelConstrained(language.to_string());
    }

    // A single-language model has a known output language without needing a
    // selectable language hint.
    if let [language] = supported_languages {
        return OutputLanguageEvidence::ModelConstrained(language.clone());
    }

    OutputLanguageEvidence::Unknown
}

/// Upgrade [`OutputLanguageEvidence::Unknown`] with the language the model
/// itself detected during the run (audio-based LID, e.g. Whisper in auto
/// mode). Stronger evidence resolved before the run is never overridden.
fn with_model_detected_language(
    evidence: OutputLanguageEvidence,
    detected: Option<String>,
) -> OutputLanguageEvidence {
    match (evidence, detected) {
        (OutputLanguageEvidence::Unknown, Some(language))
            if !language.is_empty() && language != "auto" =>
        {
            OutputLanguageEvidence::ModelDetected(language)
        }
        (evidence, _) => evidence,
    }
}

struct TranscribeCppRunPlan {
    task: Task,
    language: Option<String>,
    target_language: Option<String>,
}

/// Build the transcribe-cpp language/task options shared by batch and live
/// streaming paths.
fn transcribe_cpp_run_plan(
    translate_to_english: bool,
    effective_language: &str,
    model_languages: &[String],
    model_supports_translate: bool,
) -> TranscribeCppRunPlan {
    let requested_language = match effective_language {
        "auto" => None,
        other => Some(normalize_cjk_language(other).to_string()),
    };
    // Only pass a language the loaded model actually advertises (per
    // capabilities().languages); otherwise auto-detect rather than failing with
    // UNSUPPORTED_LANGUAGE. Language-agnostic models report an empty list, so
    // they always stay on auto.
    let language = requested_language.filter(|lang| model_languages.iter().any(|l| l == lang));
    let (task, target_language) = cpp_translation_task(
        translate_to_english,
        model_supports_translate,
        language.as_deref(),
    );

    TranscribeCppRunPlan {
        task,
        language,
        target_language,
    }
}

/// Fuzzy-correct and normalise a finished transcript.
///
/// The correction pool is the *whole* active vocabulary
/// ([`crate::dictionaries::full_vocabulary`]), regardless of what the model was
/// given as context: terms that were sent stay available for the comparison
/// afterwards, because a term appearing in a prompt does not mean the model
/// wrote it. This is the arrangement the benchmark calls strategy D; the
/// earlier one (pool minus context, strategy C) remains available there for
/// comparison. Which of the two is better for real dictation is not settled —
/// that needs recordings from human speakers, not synthetic ones.
fn post_process_transcription_text(
    raw: String,
    settings: &AppSettings,
    output_language: &OutputLanguageEvidence,
    supported_languages: &[String],
) -> String {
    fail_open_text_transform(raw, |raw| {
        let fuzzy_words = crate::dictionaries::full_vocabulary(settings);
        let corrected = if !fuzzy_words.is_empty() {
            let report = crate::audio_toolkit::correct_with_vocabulary(
                &raw,
                &fuzzy_words,
                settings.word_correction_threshold,
            );
            let replaced = report
                .events
                .iter()
                .filter(|e| e.kind == crate::audio_toolkit::CorrectionKind::Replaced)
                .count();
            info!(
                "fuzzy correction: pool={} replaced={} kept_inflected={} skipped_ambiguous={}",
                fuzzy_words.len(),
                replaced,
                report.events.len() - replaced,
                report.ambiguous_spans.len()
            );
            for event in &report.events {
                debug!(
                    "fuzzy {:?}: '{}' -> '{}' (score {:.3})",
                    event.kind,
                    crate::utils::redact_text(&event.original),
                    event.term,
                    event.score
                );
            }
            report.text
        } else {
            raw
        };

        // Last-resort language evidence: confidence-gated detection from the
        // transcribed text itself, constrained to the model's languages. Only
        // consulted when it can change the outcome (built-in gated fillers).
        let output_language = match output_language {
            OutputLanguageEvidence::Unknown
                if settings.filler_word_removal_enabled
                    && settings.custom_filler_words.is_none() =>
            {
                match detect_output_language(&corrected, supported_languages) {
                    Some(language) => {
                        debug!("Text-based language detection resolved '{}'", language);
                        OutputLanguageEvidence::TextDetected(language)
                    }
                    None => OutputLanguageEvidence::Unknown,
                }
            }
            other => other.clone(),
        };

        let without_fillers = remove_filler_words(
            &corrected,
            &output_language,
            &settings.custom_filler_words,
            settings.filler_word_removal_enabled,
        );

        normalize_transcription_output(&without_fillers)
    })
}

/// Optional text cleanup must never discard a successful model result. The
/// transform is pure and owns its input, so recovering the untouched text is
/// safe even if a bug in custom-word or filler filtering unwinds.
fn fail_open_text_transform<F>(raw: String, transform: F) -> String
where
    F: FnOnce(String) -> String,
{
    let fallback = raw.clone();
    match catch_unwind(AssertUnwindSafe(|| transform(raw))) {
        Ok(processed) => processed,
        Err(payload) => {
            error!(
                "Optional transcription text post-processing panicked: {}; using the raw transcription",
                panic_payload_message(payload.as_ref())
            );
            fallback
        }
    }
}

/// Decide a transcribe-cpp run's task + translation target from settings.
///
/// "Translate to English" only fires where the model advertises translation.
/// Unlike transcribe-rs (which forces the target to English itself when its
/// `translate` flag is set), transcribe-cpp requires an explicit
/// `target_language`: a null target defaults to the *source*, so a non-English
/// source silently becomes e.g. es→es and Canary rejects the unadvertised pair.
/// An English source is skipped entirely — en→en is not a real translation, and
/// it's reachable by default since auto-detect-less models coerce intent to "en".
///
/// Returns `(task, target_language)` ready to drop into `RunOptions`.
fn cpp_translation_task(
    translate_to_english: bool,
    model_supports_translate: bool,
    source_language: Option<&str>,
) -> (Task, Option<String>) {
    let translate_to_en =
        translate_to_english && model_supports_translate && source_language != Some("en");
    if translate_to_en {
        (Task::Translate, Some("en".to_string()))
    } else {
        (Task::Transcribe, None)
    }
}

/// Drain a stream command channel, ignoring fed audio, until the caller
/// finalizes or cancels. Used when streaming can't actually run (model not
/// loaded / not streaming-capable) so the finalize handshake still completes
/// and the caller falls back to batch transcription.
fn drain_until_finalize(rx: mpsc::Receiver<StreamCmd>) {
    while let Ok(cmd) = rx.recv() {
        match cmd {
            StreamCmd::Feed(_) => {}
            StreamCmd::Finalize(reply) => {
                let _ = reply.send(None);
                break;
            }
            StreamCmd::Cancel => break,
        }
    }
}

/// Initialize the transcribe-cpp native backend once at startup: route native +
/// ggml diagnostics into the `log` facade and register compute backend modules.
/// In a static build (macOS Metal) `init_backends_default` is a harmless no-op;
/// in a `dynamic-backends` build it loads the per-ISA CPU / GPU modules. Must run
/// before the first model load.
pub fn init_transcribe_backend() {
    transcribe_cpp::init_logging();
    match transcribe_cpp::init_backends_default() {
        Ok(()) => {
            if transcribe_gpu_disabled_for_host() {
                warn!(
                    "Windows x64 build is running under emulation on an ARM64 host; \
                     disabling transcribe.cpp GPU acceleration and using CPU"
                );
            }
            let devices = transcribe_compute_devices();
            info!(
                "transcribe-cpp initialized with {} compute device(s): [{}]",
                devices.len(),
                devices
                    .iter()
                    .map(|d| format!("{} ({})", d.name, d.kind))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }
        Err(e) => warn!("Failed to initialize transcribe-cpp backends: {}", e),
    }
}

/// Human-readable list of the transcribe-cpp compute devices registered at
/// startup, for the `--list-devices` flag. The reported `index` is the
/// value to pass to `--device-index`. Backends must be initialized first
/// (see [`init_transcribe_backend`]).
pub fn describe_compute_devices() -> Vec<String> {
    transcribe_compute_devices()
        .into_iter()
        .map(|d| {
            let idx = d
                .index
                .map(|i| i.to_string())
                .unwrap_or_else(|| "-".to_string());
            let name = if d.description.is_empty() {
                d.name
            } else {
                d.description
            };
            let vram_mb = d.memory_total / (1024 * 1024);
            format!(
                "index={} kind={} name={} vram={}MB",
                idx, d.kind, name, vram_mb
            )
        })
        .collect()
}

/// Resolve a `--list-devices` registry index to an exact opaque device handle
/// for a transcribe-cpp model load (the `--device-index` flag). In 0.2 index 0
/// is an exact selection too; only an omitted index requests automatic device
/// selection. Errors if the index isn't a registered, loadable primary device.
fn resolve_device_index(index: usize) -> Result<(Backend, Option<transcribe_cpp::Device>)> {
    let device = transcribe_compute_devices()
        .into_iter()
        .find(|d| d.index == Some(index))
        .ok_or_else(|| {
            anyhow::anyhow!("No compute device with index {index} (see --list-devices)")
        })?;
    if matches!(
        device.device_type,
        transcribe_cpp::DeviceType::Accel | transcribe_cpp::DeviceType::Unknown
    ) {
        return Err(anyhow::anyhow!(
            "Device index {index} ({}) cannot host a model",
            device.kind
        ));
    }

    // 0.2's opaque handle makes every index, including zero, an exact
    // selection. Backend::Auto accepts any primary device and cannot conflict
    // with the selected device's vendor backend.
    Ok((Backend::Auto, Some(device)))
}

/// Map Handy's whisper accelerator setting to a transcribe-cpp [`Backend`].
///
/// `Auto` lets the library pick the best device (with CPU fallback), while
/// `Cpu` forces strict CPU. `Gpu` only remains as the companion setting for an
/// exact device; without a valid exact device it has the retired generic GPU
/// state's new Auto semantics. An emulated x64 process on Windows ARM64 forces
/// strict CPU for every setting.
fn select_transcribe_backend(setting: TranscribeAcceleratorSetting) -> Backend {
    select_transcribe_backend_for_host(setting, transcribe_gpu_disabled_for_host())
}

fn select_transcribe_backend_for_host(
    setting: TranscribeAcceleratorSetting,
    gpu_disabled: bool,
) -> Backend {
    match effective_transcribe_accelerator(setting, gpu_disabled) {
        TranscribeAcceleratorSetting::Cpu => Backend::Cpu,
        TranscribeAcceleratorSetting::Auto | TranscribeAcceleratorSetting::Gpu => Backend::Auto,
    }
}

/// Resolve the user's persisted GPU identity to a fresh opaque 0.2 device
/// handle. Registry indices and handles are process-local, so settings store a
/// key based on the backend's stable `device_id` (falling back to name for
/// backends such as Metal that do not report one).
fn resolve_gpu_device(
    setting: TranscribeAcceleratorSetting,
    gpu_device: Option<&str>,
) -> Option<transcribe_cpp::Device> {
    if transcribe_gpu_disabled_for_host() || setting != TranscribeAcceleratorSetting::Gpu {
        return None;
    }
    let gpu_device = gpu_device?;
    let resolved = transcribe_compute_devices().into_iter().find(|device| {
        is_transcribe_gpu_device(device) && transcribe_device_key(device) == gpu_device
    });
    if resolved.is_none() {
        warn!(
            "Stored transcribe GPU device '{}' is no longer available; using automatic device selection",
            gpu_device
        );
    }
    resolved
}

fn transcribe_device_key(device: &transcribe_cpp::Device) -> String {
    let (identity_kind, identity) = match device.device_id.as_deref() {
        Some(device_id) => ("id", device_id),
        None => ("name", device.name.as_str()),
    };
    serde_json::to_string(&(device.kind.as_str(), identity_kind, identity))
        .expect("transcribe device identity is always JSON serializable")
}

fn transcribe_device_label(device: &transcribe_cpp::Device) -> String {
    if device.description.is_empty() {
        device.name.clone()
    } else {
        device.description.clone()
    }
}

/// Apply the user's ORT accelerator preference to the transcribe-rs global.
/// Called on startup and before loading a model.
///
/// The transcribe.cpp (whisper-family) backend is no longer set here: it is
/// chosen at model-load time from [`select_transcribe_backend`], so changing the
/// accelerator only needs a model reload (see `reload_model_on_next_use`).
pub fn apply_accelerator_settings(app: &tauri::AppHandle) {
    use transcribe_rs::accel;

    let settings = get_settings(app);

    info!(
        "transcribe.cpp accelerator preference: {:?} (applied on next model load)",
        settings.transcribe_accelerator
    );

    let ort_pref = match settings.ort_accelerator {
        OrtAcceleratorSetting::Auto => accel::OrtAccelerator::Auto,
        OrtAcceleratorSetting::Cpu => accel::OrtAccelerator::CpuOnly,
        OrtAcceleratorSetting::Cuda => accel::OrtAccelerator::Cuda,
        OrtAcceleratorSetting::DirectMl => accel::OrtAccelerator::DirectMl,
        OrtAcceleratorSetting::Rocm => accel::OrtAccelerator::Rocm,
    };
    accel::set_ort_accelerator(ort_pref);
    info!("ORT accelerator set to: {}", ort_pref);
}

#[derive(Serialize, Clone, Debug, Type)]
pub struct GpuDeviceOption {
    pub id: String,
    pub name: String,
    pub total_vram_mb: usize,
}

static GPU_DEVICES: OnceLock<Vec<GpuDeviceOption>> = OnceLock::new();

fn transcribe_gpu_disabled_for_host() -> bool {
    crate::utils::is_windows_x64_emulated_on_arm64()
}

fn effective_transcribe_accelerator(
    setting: TranscribeAcceleratorSetting,
    gpu_disabled: bool,
) -> TranscribeAcceleratorSetting {
    if gpu_disabled {
        TranscribeAcceleratorSetting::Cpu
    } else {
        setting
    }
}

fn is_transcribe_gpu_device(device: &transcribe_cpp::Device) -> bool {
    matches!(
        device.device_type,
        transcribe_cpp::DeviceType::Gpu | transcribe_cpp::DeviceType::Igpu
    )
}

fn transcribe_device_allowed(kind: &str, gpu_disabled: bool) -> bool {
    !gpu_disabled || matches!(kind, "cpu" | "accel")
}

fn transcribe_compute_devices() -> Vec<transcribe_cpp::Device> {
    let devices = transcribe_cpp::devices();
    let gpu_disabled = transcribe_gpu_disabled_for_host();
    if !gpu_disabled {
        return devices;
    }

    devices
        .into_iter()
        .filter(|device| transcribe_device_allowed(&device.kind, gpu_disabled))
        .collect()
}

fn available_transcribe_accelerators(gpu_disabled: bool) -> Vec<String> {
    if gpu_disabled {
        vec!["cpu".to_string()]
    } else {
        vec!["auto".to_string(), "cpu".to_string(), "gpu".to_string()]
    }
}

fn cached_gpu_devices() -> &'static [GpuDeviceOption] {
    // GPU compute devices transcribe-cpp registered at startup. `id` is a
    // persistent identity key, never the process-local registry index. It uses
    // the backend's device_id where available and its name otherwise (Metal).
    // `total_vram_mb` is 0 when the backend does not report capacity.
    GPU_DEVICES.get_or_init(|| {
        transcribe_compute_devices()
            .into_iter()
            .filter(is_transcribe_gpu_device)
            .map(|d| GpuDeviceOption {
                id: transcribe_device_key(&d),
                name: transcribe_device_label(&d),
                total_vram_mb: (d.memory_total / (1024 * 1024)) as usize,
            })
            .collect()
    })
}

#[derive(Serialize, Clone, Debug, Type)]
pub struct AvailableAccelerators {
    pub transcribe: Vec<String>,
    pub ort: Vec<String>,
    pub gpu_devices: Vec<GpuDeviceOption>,
}

/// Return the accelerators available to this process on its current host.
pub fn get_available_accelerators() -> AvailableAccelerators {
    use transcribe_rs::accel::OrtAccelerator;

    let ort_options: Vec<String> = OrtAccelerator::available()
        .into_iter()
        .map(|a| a.to_string())
        .collect();

    let transcribe_options = available_transcribe_accelerators(transcribe_gpu_disabled_for_host());

    AvailableAccelerators {
        transcribe: transcribe_options,
        ort: ort_options,
        gpu_devices: cached_gpu_devices().to_vec(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn languages(codes: &[&str]) -> Vec<String> {
        codes.iter().map(|code| (*code).to_string()).collect()
    }

    // ---------------------------------------------------------------------
    // Dictionary → allocator → backend → model context
    //
    // These walk the same two steps the run path takes for a loaded model:
    // `transcribe_cpp_context_biasing(arch)` decides what the model may
    // receive, `ContextBiasing::terms` fills the budget, and
    // `format_context_terms` renders exactly the string handed to the engine.
    // Only the native call itself (which needs a real model on disk) is left
    // out.
    // ---------------------------------------------------------------------

    /// Build the very `RunOptions` the run path hands to `session.run()` for a
    /// model of `arch`, so a test can read the context out of the same struct
    /// that crosses the FFI boundary rather than out of an intermediate list.
    ///
    /// Mirrors the channel dispatch in `transcribe`: the rendered terms go into
    /// `RunOptions::context` for a recognition-context model and into the
    /// whisper run extension's `initial_prompt` for whisper.
    fn run_options_for(settings: &AppSettings, arch: &str, supports_context: bool) -> RunOptions {
        let biasing = transcribe_cpp_context_biasing(arch, supports_context);
        let terms = biasing.terms(settings);
        let rendered = (!terms.is_empty()).then(|| format_context_terms(&terms));
        let (family, context) = match biasing.channel {
            Some(ContextChannel::InitialPrompt) => (
                rendered.map(|prompt| {
                    RunExtension::Whisper(WhisperRunOptions {
                        initial_prompt: Some(prompt),
                        ..Default::default()
                    })
                }),
                None,
            ),
            Some(ContextChannel::RecognitionContext) => (None, rendered),
            None => (None, None),
        };
        RunOptions {
            family,
            context,
            ..Default::default()
        }
    }

    /// The recognition context actually sitting in `RunOptions::context`.
    fn qwen_run_context(settings: &AppSettings) -> Option<String> {
        run_options_for(settings, "qwen3_asr", true).context
    }

    /// The initial prompt actually sitting in the whisper run extension.
    fn whisper_run_prompt(settings: &AppSettings) -> Option<String> {
        match run_options_for(settings, "whisper", false).family {
            Some(RunExtension::Whisper(options)) => options.initial_prompt,
            _ => None,
        }
    }

    /// The context string a model would be given through whichever channel it
    /// uses, or `None` when it takes none.
    fn model_context_string(
        settings: &AppSettings,
        arch: &str,
        supports_context: bool,
    ) -> Option<String> {
        let options = run_options_for(settings, arch, supports_context);
        options.context.or(match options.family {
            Some(RunExtension::Whisper(w)) => w.initial_prompt,
            _ => None,
        })
    }

    /// Dictionaries switched on for both steps — post-correction over the whole
    /// list, context selection limited by the tier.
    fn settings_with(dictionaries: &[&str]) -> AppSettings {
        let mut settings = crate::settings::get_default_settings();
        settings.active_dictionaries = dictionaries.iter().map(|d| d.to_string()).collect();
        settings.context_dictionaries = settings.active_dictionaries.clone();
        settings
    }

    #[test]
    fn whisper_receives_a_medical_context_built_from_the_active_dictionary() {
        let settings = settings_with(&["internal_cardiology"]);
        let context = whisper_run_prompt(&settings).expect("whisper takes context");
        // A high-priority cardiology term reaches the actual model string, not
        // just the dictionary layer's return value.
        assert!(
            context.contains("AV-Knoten-Reentrytachykardie"),
            "context was: {}",
            context
        );
        // Multi-word terms survive the trip intact.
        assert!(context.contains("paroxysmale supraventrikuläre Tachykardie"));
        // No metadata rides along.
        assert!(!context.contains("rank") && !context.contains("module_id"));
    }

    #[test]
    fn model_context_stays_within_the_architecture_budget() {
        let settings = settings_with(&["internal_cardiology", "core_medical"]);
        let terms = transcribe_cpp_context_biasing("whisper", false).terms(&settings);
        assert_eq!(terms.len(), WHISPER_CONTEXT_BUDGET);
        // The rendered string carries exactly those terms, comma separated.
        let context = format_context_terms(&terms);
        assert_eq!(context.split(", ").count(), WHISPER_CONTEXT_BUDGET);
    }

    #[test]
    fn every_active_module_is_represented_in_the_model_context() {
        let ids = [
            "internal_cardiology",
            "anatomy_heart_vessels",
            "meds_cardiology_generic",
        ];
        let settings = settings_with(&ids);
        let context = whisper_run_prompt(&settings).unwrap();
        // One recognisable term from each module's top ranks.
        assert!(
            context.contains("Tachykardie-Bradykardie-Syndrom"),
            "cardiology missing"
        );
        assert!(
            context.contains("Sinus transversus pericardii"),
            "anatomy missing"
        );
        assert!(
            context.contains("unfraktioniertes Heparin"),
            "medication missing"
        );
    }

    #[test]
    fn a_personal_word_leads_the_model_context() {
        let mut settings = settings_with(&["internal_cardiology"]);
        settings.custom_words = vec!["Blatt-Schmidt-Zeichen".to_string()];
        let context = whisper_run_prompt(&settings).unwrap();
        assert!(
            context.starts_with("Blatt-Schmidt-Zeichen, "),
            "context was: {}",
            context
        );
    }

    #[test]
    fn a_term_beyond_the_budget_is_absent_from_the_context_but_kept_for_correction() {
        let mut settings = settings_with(&["internal_cardiology"]);
        settings
            .dictionary_levels
            .insert("internal_cardiology".to_string(), 500);
        let terms = transcribe_cpp_context_biasing("whisper", false).terms(&settings);
        let context = format_context_terms(&terms);

        let full = crate::dictionaries::full_vocabulary(&settings);
        let dropped = &full[WHISPER_CONTEXT_BUDGET + 20];
        assert!(
            !context.contains(dropped.as_str()),
            "{} should not be in context",
            dropped
        );

        // Correction draws on the whole pool (strategy D), so the dropped term
        // is caught afterwards — and so are the ones that were sent.
        assert!(full.contains(dropped), "{} lost from the pool", dropped);
    }

    #[test]
    fn qwen_advertising_the_capability_receives_a_recognition_context() {
        let biasing = transcribe_cpp_context_biasing("qwen3_asr", true);
        assert!(biasing.supported(), "qwen3_asr should take context");
        assert_eq!(biasing.channel, Some(ContextChannel::RecognitionContext));
        assert_eq!(biasing.budget, LLM_DECODER_CONTEXT_BUDGET);

        // …and it lands in the field that crosses the FFI boundary, not in the
        // whisper extension.
        let settings = settings_with(&["internal_cardiology"]);
        let options = run_options_for(&settings, "qwen3_asr", true);
        assert!(options.family.is_none(), "qwen must not get a whisper ext");
        let context = options.context.expect("recognition context missing");
        assert!(
            context.contains("AV-Knoten-Reentrytachykardie"),
            "context was: {}",
            context
        );
    }

    #[test]
    fn qwen_recognition_context_carries_every_active_module_and_leads_with_personal_words() {
        let ids = [
            "core_medical",
            "internal_cardiology",
            "anatomy_heart_vessels",
            "meds_cardiology_generic",
        ];
        let mut settings = settings_with(&ids);
        settings.custom_words = vec!["Blatt-Schmidt-Zeichen".to_string()];
        let context = qwen_run_context(&settings).expect("recognition context missing");

        assert!(context.starts_with("Blatt-Schmidt-Zeichen, "));
        for term in [
            "reduzierter Allgemeinzustand",
            "Tachykardie-Bradykardie-Syndrom",
            "Sinus transversus pericardii",
            "unfraktioniertes Heparin",
        ] {
            assert!(context.contains(term), "missing {}", term);
        }
        // Budget honoured, multi-word terms whole, no metadata. The markers are
        // harness field names and JSON punctuation — deliberately not bare words
        // like "rank", which is a substring of "koronare Herzkrankheit".
        assert_eq!(context.split(", ").count(), LLM_DECODER_CONTEXT_BUDGET);
        for marker in [
            "term_id",
            "module_id",
            "asr_priority_score",
            "frequency_score",
            "{",
            "}",
            "\"",
        ] {
            assert!(
                !context.contains(marker),
                "metadata marker {} leaked",
                marker
            );
        }
    }

    #[test]
    fn a_qwen_term_beyond_the_budget_is_absent_but_still_fuzzy_corrected() {
        let mut settings = settings_with(&["internal_cardiology"]);
        settings
            .dictionary_levels
            .insert("internal_cardiology".to_string(), 500);
        let context = qwen_run_context(&settings).unwrap();

        let full = crate::dictionaries::full_vocabulary(&settings);
        let dropped = &full[LLM_DECODER_CONTEXT_BUDGET + 20];
        assert!(!context.contains(dropped.as_str()), "{} leaked", dropped);
        assert!(full.contains(dropped), "{} lost from the pool", dropped);
    }

    /// A dictionary limited to post-correction must not reach any context
    /// channel — neither whisper's prompt nor Qwen's recognition context —
    /// while every one of its terms stays available for the correction.
    #[test]
    fn a_fuzzy_only_dictionary_reaches_no_context_channel() {
        let mut settings = settings_with(&["internal_cardiology"]);
        // Post-correction only: in the correction pool, out of every context.
        settings.context_dictionaries.clear();
        for (arch, supports) in [("whisper", false), ("qwen3_asr", true)] {
            let biasing = transcribe_cpp_context_biasing(arch, supports);
            assert!(biasing.supported(), "{} should support context", arch);
            assert!(
                biasing.terms(&settings).is_empty(),
                "{} received restricted terms",
                arch
            );
            assert_eq!(model_context_string(&settings, arch, supports), None);
        }
        let full = crate::dictionaries::full_vocabulary(&settings);
        assert!(full.iter().any(|w| w == "AV-Knoten-Reentrytachykardie"));
    }

    /// Strategy D end to end: a term that *was* sent as context is still
    /// corrected in the finished text. Under the earlier split (strategy C)
    /// this misspelling would have survived untouched.
    #[test]
    fn a_term_that_was_sent_as_context_is_still_corrected_afterwards() {
        let settings = settings_with(&["internal_cardiology"]);
        let sent = transcribe_cpp_context_biasing("qwen3_asr", true).terms(&settings);
        assert!(
            sent.iter().any(|w| w == "Herzkatheteruntersuchung"),
            "the term must be in the context for this test to mean anything"
        );
        let corrected = post_process_transcription_text(
            "Es erfolgte eine Herzkatheterunterschung.".to_string(),
            &settings,
            &OutputLanguageEvidence::Unknown,
            &languages(&["de"]),
        );
        assert!(
            corrected.contains("Herzkatheteruntersuchung"),
            "correction skipped a context term: {}",
            corrected
        );
    }

    /// After a failed context run the app repeats without context. The
    /// correction pool is the whole active vocabulary either way, so nothing
    /// about that second attempt narrows what can still be repaired.
    #[test]
    fn the_correction_pool_is_the_same_with_context_without_and_after_a_fallback() {
        let settings = settings_with(&["internal_cardiology"]);
        let pool = crate::dictionaries::full_vocabulary(&settings);
        for arch in ["qwen3_asr", "whisper", "parakeet"] {
            let sent = transcribe_cpp_context_biasing(arch, arch == "qwen3_asr").terms(&settings);
            // Whatever the run sent — many terms, none, or none after a
            // fallback — post-correction sees the identical pool.
            assert_eq!(
                crate::dictionaries::full_vocabulary(&settings),
                pool,
                "{} changed the correction pool (sent {})",
                arch,
                sent.len()
            );
        }
    }

    /// A model that advertises neither channel must be left alone: forcing the
    /// whisper extension onto a foreign arch is rejected natively with
    /// INVALID_ARG (#1601), and `RunOptions::context` would be ignored at best.
    /// Its vocabulary reaches fuzzy correction in full instead.
    #[test]
    fn a_model_without_any_context_channel_gets_none_and_keeps_its_full_pool() {
        let settings = settings_with(&["internal_cardiology"]);
        for arch in ["parakeet", "moonshine", "sensevoice"] {
            let biasing = transcribe_cpp_context_biasing(arch, false);
            assert!(!biasing.supported(), "{} claims context support", arch);
            assert!(biasing.terms(&settings).is_empty(), "{} got terms", arch);

            let options = run_options_for(&settings, arch, false);
            assert!(options.context.is_none(), "{} got a context", arch);
            assert!(options.family.is_none(), "{} got an extension", arch);
            assert_eq!(model_context_string(&settings, arch, false), None);
        }
        let full = crate::dictionaries::full_vocabulary(&settings);
        assert!(full.iter().any(|w| w == "AV-Knoten-Reentrytachykardie"));
    }

    #[test]
    fn each_architecture_gets_its_own_channel_and_budget() {
        let whisper = transcribe_cpp_context_biasing("whisper", false);
        assert_eq!(whisper.channel, Some(ContextChannel::InitialPrompt));
        assert_eq!(whisper.budget, WHISPER_CONTEXT_BUDGET);

        let qwen = transcribe_cpp_context_biasing("qwen3_asr", true);
        assert_eq!(qwen.channel, Some(ContextChannel::RecognitionContext));
        assert_eq!(qwen.budget, LLM_DECODER_CONTEXT_BUDGET);

        // The capability probe, not a hardcoded arch list, is what opens the
        // recognition-context channel — a future arch needs no code change.
        assert!(transcribe_cpp_context_biasing("some_future_arch", true).supported());
        assert_eq!(
            transcribe_cpp_context_biasing("qwen3_asr", false),
            ContextBiasing::NONE
        );
    }

    #[test]
    fn formatting_keeps_terms_verbatim_and_adds_nothing() {
        let terms = vec![
            "AV-Knoten-Reentrytachykardie".to_string(),
            "Arteria cerebri media".to_string(),
            "Sacubitril/Valsartan".to_string(),
        ];
        assert_eq!(
            format_context_terms(&terms),
            "AV-Knoten-Reentrytachykardie, Arteria cerebri media, Sacubitril/Valsartan"
        );
        assert!(format_context_terms(&[]).is_empty());
    }

    #[test]
    fn normal_hosts_preserve_every_transcribe_accelerator_setting() {
        for setting in [
            TranscribeAcceleratorSetting::Auto,
            TranscribeAcceleratorSetting::Cpu,
            TranscribeAcceleratorSetting::Gpu,
        ] {
            assert_eq!(effective_transcribe_accelerator(setting, false), setting);
        }
        assert_eq!(
            available_transcribe_accelerators(false),
            ["auto", "cpu", "gpu"]
        );
        assert_eq!(
            select_transcribe_backend_for_host(TranscribeAcceleratorSetting::Auto, false),
            Backend::Auto
        );
        assert_eq!(
            select_transcribe_backend_for_host(TranscribeAcceleratorSetting::Cpu, false),
            Backend::Cpu
        );
        assert_eq!(
            select_transcribe_backend_for_host(TranscribeAcceleratorSetting::Gpu, false),
            Backend::Auto
        );
        for kind in ["cpu", "accel", "metal", "cuda", "vulkan", "gpu"] {
            assert!(transcribe_device_allowed(kind, false));
        }
    }

    #[test]
    fn emulated_x64_on_arm64_forces_every_transcribe_setting_to_cpu() {
        for setting in [
            TranscribeAcceleratorSetting::Auto,
            TranscribeAcceleratorSetting::Cpu,
            TranscribeAcceleratorSetting::Gpu,
        ] {
            assert_eq!(
                effective_transcribe_accelerator(setting, true),
                TranscribeAcceleratorSetting::Cpu
            );
            assert_eq!(
                select_transcribe_backend_for_host(setting, true),
                Backend::Cpu
            );
        }
        assert_eq!(available_transcribe_accelerators(true), ["cpu"]);
        assert!(transcribe_device_allowed("cpu", true));
        assert!(transcribe_device_allowed("accel", true));
        for kind in ["metal", "cuda", "vulkan", "gpu", "unknown"] {
            assert!(!transcribe_device_allowed(kind, true));
        }
    }

    #[test]
    fn optional_text_transform_falls_back_to_raw_text_after_panic() {
        let raw = "原始轉錄。".to_string();
        let result = fail_open_text_transform(raw.clone(), |_| {
            panic!("simulated optional cleanup failure")
        });

        assert_eq!(result, raw);
    }

    #[test]
    fn portuguese_transcription_does_not_use_english_ui_filler_words() {
        let settings = AppSettings {
            app_language: "en".to_string(),
            selected_language: "pt-BR".to_string(),
            ..Default::default()
        };
        let supported = languages(&["en", "pt"]);
        let evidence = resolve_output_language_evidence(&settings, Some("pt"), &supported, false);

        let result = post_process_transcription_text(
            "eu vi um carro".to_string(),
            &settings,
            &evidence,
            &supported,
        );

        assert_eq!(
            evidence,
            OutputLanguageEvidence::UserSelected("pt".to_string())
        );
        assert_eq!(result, "eu vi um carro");
    }

    #[test]
    fn auto_language_without_detection_skips_gated_filler_removal() {
        let settings = AppSettings {
            selected_language: "auto".to_string(),
            ..Default::default()
        };
        let evidence =
            resolve_output_language_evidence(&settings, None, &languages(&["en", "pt"]), false);

        // Too short for a reliable text detection, so the gated "um" must
        // survive; the universal "uhm" is removed regardless.
        let result = post_process_transcription_text(
            "um uhm ok".to_string(),
            &settings,
            &evidence,
            &languages(&["en", "pt"]),
        );

        assert_eq!(evidence, OutputLanguageEvidence::Unknown);
        assert_eq!(result, "um ok");
    }

    #[test]
    fn unknown_evidence_with_confident_text_detection_removes_gated_fillers() {
        let settings = AppSettings {
            selected_language: "auto".to_string(),
            ..Default::default()
        };

        let result = post_process_transcription_text(
            "um so the weather forecast said it would probably rain throughout the whole weekend"
                .to_string(),
            &settings,
            &OutputLanguageEvidence::Unknown,
            &languages(&["en", "pt", "es", "de"]),
        );

        assert_eq!(
            result,
            "so the weather forecast said it would probably rain throughout the whole weekend"
        );
    }

    #[test]
    fn unknown_evidence_with_portuguese_text_preserves_um() {
        let settings = AppSettings {
            selected_language: "auto".to_string(),
            ..Default::default()
        };

        let result = post_process_transcription_text(
            "eu vi um carro na rua ontem de manhã quando fui ao mercado".to_string(),
            &settings,
            &OutputLanguageEvidence::Unknown,
            &languages(&["en", "pt", "es", "de"]),
        );

        assert_eq!(
            result,
            "eu vi um carro na rua ontem de manhã quando fui ao mercado"
        );
    }

    #[test]
    fn model_detected_language_upgrades_unknown_evidence_only() {
        assert_eq!(
            with_model_detected_language(OutputLanguageEvidence::Unknown, Some("en".to_string())),
            OutputLanguageEvidence::ModelDetected("en".to_string())
        );
        assert_eq!(
            with_model_detected_language(OutputLanguageEvidence::Unknown, Some("auto".to_string())),
            OutputLanguageEvidence::Unknown
        );
        assert_eq!(
            with_model_detected_language(OutputLanguageEvidence::Unknown, None),
            OutputLanguageEvidence::Unknown
        );
        assert_eq!(
            with_model_detected_language(
                OutputLanguageEvidence::UserSelected("pt".to_string()),
                Some("en".to_string())
            ),
            OutputLanguageEvidence::UserSelected("pt".to_string())
        );
    }

    #[test]
    fn auto_language_uses_single_language_model_as_evidence() {
        let settings = AppSettings {
            selected_language: "auto".to_string(),
            ..Default::default()
        };

        let evidence =
            resolve_output_language_evidence(&settings, None, &languages(&["en"]), false);

        assert_eq!(
            evidence,
            OutputLanguageEvidence::ModelConstrained("en".to_string())
        );
    }

    #[test]
    fn unsupported_explicit_language_uses_model_fallback_as_evidence() {
        let settings = AppSettings {
            selected_language: "pt".to_string(),
            ..Default::default()
        };

        let evidence = resolve_output_language_evidence(
            &settings,
            Some("en"),
            &languages(&["en", "de"]),
            false,
        );

        assert_eq!(
            evidence,
            OutputLanguageEvidence::ModelConstrained("en".to_string())
        );
    }

    #[test]
    fn ignored_user_language_is_not_output_evidence() {
        let settings = AppSettings {
            // Parakeet V3 ignores language hints and auto-detects even when a
            // selection from the previously active model remains persisted.
            selected_language: "en".to_string(),
            ..Default::default()
        };
        let supported = languages(&["en", "de", "pt"]);

        let evidence = resolve_output_language_evidence(&settings, None, &supported, false);
        assert_eq!(evidence, OutputLanguageEvidence::Unknown);

        let result = post_process_transcription_text(
            "eu vi um carro".to_string(),
            &settings,
            &evidence,
            &supported,
        );
        assert_eq!(result, "eu vi um carro");
    }

    #[test]
    fn unapplied_transcribe_cpp_language_is_not_output_evidence() {
        let settings = AppSettings {
            selected_language: "en".to_string(),
            ..Default::default()
        };
        let supported = languages(&[]);
        let plan = transcribe_cpp_run_plan(false, "en", &supported, false);

        assert_eq!(plan.language, None);
        assert_eq!(
            resolve_output_language_evidence(
                &settings,
                plan.language.as_deref(),
                &supported,
                false,
            ),
            OutputLanguageEvidence::Unknown
        );
    }

    #[test]
    fn translated_output_is_treated_as_english() {
        let settings = AppSettings {
            selected_language: "pt".to_string(),
            ..Default::default()
        };

        let evidence = resolve_output_language_evidence(
            &settings,
            Some("pt"),
            &languages(&["en", "pt"]),
            true,
        );

        assert_eq!(evidence, OutputLanguageEvidence::TranslatedToEnglish);
    }

    #[test]
    fn transcribe_cpp_run_plan_maps_chinese_variants() {
        let plan = transcribe_cpp_run_plan(false, "zh-Hant", &languages(&["zh"]), true);

        assert!(matches!(plan.task, Task::Transcribe));
        assert_eq!(plan.language.as_deref(), Some("zh"));
        assert_eq!(plan.target_language, None);
    }

    #[test]
    fn transcribe_cpp_run_plan_skips_english_translation() {
        let plan = transcribe_cpp_run_plan(true, "en", &languages(&["en", "es"]), true);

        assert!(matches!(plan.task, Task::Transcribe));
        assert_eq!(plan.language.as_deref(), Some("en"));
        assert_eq!(plan.target_language, None);
    }

    #[test]
    fn transcribe_cpp_run_plan_translates_supported_non_english() {
        let plan = transcribe_cpp_run_plan(true, "es", &languages(&["en", "es"]), true);

        assert!(matches!(plan.task, Task::Translate));
        assert_eq!(plan.language.as_deref(), Some("es"));
        assert_eq!(plan.target_language.as_deref(), Some("en"));
    }

    #[test]
    fn transcribe_cpp_run_plan_requires_model_translation_support() {
        let plan = transcribe_cpp_run_plan(true, "es", &languages(&["en", "es"]), false);

        assert!(matches!(plan.task, Task::Transcribe));
        assert_eq!(plan.language.as_deref(), Some("es"));
        assert_eq!(plan.target_language, None);
    }
}

impl Drop for TranscriptionManager {
    fn drop(&mut self) {
        // Skip shutdown unless this is the very last clone. TranscriptionManager
        // is cloned by initiate_model_load() and the watcher thread — those
        // clones dropping must not kill the watcher. The watcher thread holds
        // its own clone, so engine's strong_count is always >= 2 while the
        // watcher is alive. When it reaches 1, only this instance remains
        // and we can safely shut down.
        if Arc::strong_count(&self.engine) > 1 {
            return;
        }

        // Signal the watcher thread to shutdown
        self.shutdown_signal.store(true, Ordering::Relaxed);

        // Wait for the thread to finish gracefully.
        // Use match instead of unwrap to avoid panicking if the mutex is
        // poisoned — a panic inside Drop calls abort().
        let mut guard = match self.watcher_handle.lock() {
            Ok(g) => g,
            Err(e) => {
                warn!("Recovered poisoned watcher_handle mutex during TranscriptionManager drop — a panic occurred earlier this session");
                e.into_inner()
            }
        };
        if let Some(handle) = guard.take() {
            if let Err(e) = handle.join() {
                warn!("Failed to join idle watcher thread: {:?}", e);
            } else {
                debug!("Idle watcher thread joined successfully");
            }
        }
    }
}
