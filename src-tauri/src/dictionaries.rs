//! Domain dictionaries (medical specialties) that extend the user's custom
//! words during transcription.
//!
//! Two kinds exist:
//! - **Bundled** dictionaries live in `src-tauri/dictionaries/` as plain text
//!   (one term per line, `#` for comments) and are embedded at compile time,
//!   so they work fully offline. Users can add their own words on top and
//!   hide bundled words; both edits are stored in settings
//!   ([`crate::settings::DictionaryCustomization`]) — the bundled list itself
//!   stays untouched, so a reset just drops the customization.
//! - **User-created** dictionaries ([`crate::settings::CustomDictionary`])
//!   live entirely in settings and are fully editable.
//!
//! Word order is priority order: for models that take an initial prompt only
//! a prefix of the merged list fits, so the most important terms belong at
//! the top. User-added words sort before bundled ones for the same reason.
//!
//! Some bundled lists are **tiered** (100 / 250 / 500). Because their tiers are
//! prefixes of one another, a tier is a `take(n)` over a single file rather
//! than a file per tier — see [`BuiltinDictionary::levels`]. Only the tiers of
//! *activated* dictionaries are merged into the model context, which is what
//! keeps that context from growing with the bundle.
//!
//! Nothing here is specific to a speech model: this module hands
//! [`full_vocabulary`] a plain, deduplicated list of terms, and the
//! transcription layer decides what to do with it.

use serde::{Deserialize, Serialize};
use specta::Type;
use std::collections::HashSet;
use tauri::AppHandle;

use crate::settings::{self, AppSettings, CustomDictionary, DictionaryCustomization};

/// Which section of the settings page a dictionary belongs to. Purely a
/// grouping hint for the UI; it has no effect on what reaches the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum DictionaryGroup {
    /// Cross-specialty vocabulary that is useful in almost any dictation.
    Core,
    /// A clinical specialty.
    Specialty,
    /// Anatomy, selectable per body region and combinable with any specialty.
    Anatomy,
    /// Drug vocabulary, kept apart from clinical terms: agents, drug classes
    /// and — opt-in only — trade names.
    Medication,
}

/// A compiled-in dictionary. `id` is the stable key stored in settings and
/// used by the frontend for i18n labels (`dictionary.names.<id>`).
struct BuiltinDictionary {
    id: &'static str,
    raw: &'static str,
    group: DictionaryGroup,
    /// Selectable tier sizes, ascending, or empty for an all-or-nothing list.
    ///
    /// A tiered list is ordered by ASR priority and each tier is a *prefix* of
    /// the next, so a tier is `take(n)` over one file rather than a file of its
    /// own. That is what makes 100 ⊂ 250 ⊂ 500 hold by construction instead of
    /// depending on three files staying in sync.
    levels: &'static [u32],
}

/// Modules from the curated German medical vocabulary (release `generated_v2`);
/// ids mirror the manifest's `module_id` with `.` replaced by `_` so they stay
/// flat i18n keys. Order matters: context selection walks the active modules
/// round-robin in this order, so ties go to whatever stands higher here.
///
/// An id that disappears from this table is handled gracefully: it simply stops
/// resolving, so a stale entry left in a user's `active_dictionaries` is skipped
/// rather than breaking the merge.
const BUILTIN: &[BuiltinDictionary] = &[
    BuiltinDictionary {
        id: "core_medical",
        raw: include_str!("../dictionaries/de/core_medical.txt"),
        group: DictionaryGroup::Core,
        levels: &[100, 250],
    },
    BuiltinDictionary {
        id: "internal_cardiology",
        raw: include_str!("../dictionaries/de/internal_cardiology.txt"),
        group: DictionaryGroup::Specialty,
        levels: &[100, 250, 500],
    },
    BuiltinDictionary {
        id: "internal_angiology",
        raw: include_str!("../dictionaries/de/internal_angiology.txt"),
        group: DictionaryGroup::Specialty,
        levels: &[100, 250, 500],
    },
    BuiltinDictionary {
        id: "internal_endocrinology_diabetology",
        raw: include_str!("../dictionaries/de/internal_endocrinology_diabetology.txt"),
        group: DictionaryGroup::Specialty,
        levels: &[100, 250, 500],
    },
    BuiltinDictionary {
        id: "internal_gastro_hepatology",
        raw: include_str!("../dictionaries/de/internal_gastro_hepatology.txt"),
        group: DictionaryGroup::Specialty,
        levels: &[100, 250, 500],
    },
    BuiltinDictionary {
        id: "internal_hematology",
        raw: include_str!("../dictionaries/de/internal_hematology.txt"),
        group: DictionaryGroup::Specialty,
        levels: &[100, 250, 500],
    },
    BuiltinDictionary {
        id: "internal_infectiology",
        raw: include_str!("../dictionaries/de/internal_infectiology.txt"),
        group: DictionaryGroup::Specialty,
        levels: &[100, 250, 500],
    },
    BuiltinDictionary {
        id: "internal_nephrology",
        raw: include_str!("../dictionaries/de/internal_nephrology.txt"),
        group: DictionaryGroup::Specialty,
        levels: &[100, 250, 500],
    },
    BuiltinDictionary {
        id: "internal_oncology",
        raw: include_str!("../dictionaries/de/internal_oncology.txt"),
        group: DictionaryGroup::Specialty,
        levels: &[100, 250, 500],
    },
    BuiltinDictionary {
        id: "internal_pneumology",
        raw: include_str!("../dictionaries/de/internal_pneumology.txt"),
        group: DictionaryGroup::Specialty,
        levels: &[100, 250, 500],
    },
    BuiltinDictionary {
        id: "internal_rheumatology",
        raw: include_str!("../dictionaries/de/internal_rheumatology.txt"),
        group: DictionaryGroup::Specialty,
        levels: &[100, 250, 500],
    },
    BuiltinDictionary {
        id: "neurology",
        raw: include_str!("../dictionaries/de/neurology.txt"),
        group: DictionaryGroup::Specialty,
        levels: &[100, 250, 500],
    },
    BuiltinDictionary {
        id: "radiology",
        raw: include_str!("../dictionaries/de/radiology.txt"),
        group: DictionaryGroup::Specialty,
        levels: &[100, 250, 500],
    },
    BuiltinDictionary {
        id: "surgery_visceral",
        raw: include_str!("../dictionaries/de/surgery_visceral.txt"),
        group: DictionaryGroup::Specialty,
        levels: &[100, 250, 500],
    },
    BuiltinDictionary {
        id: "anatomy_heart_vessels",
        raw: include_str!("../dictionaries/de/anatomy_heart_vessels.txt"),
        group: DictionaryGroup::Anatomy,
        levels: &[100, 250],
    },
    BuiltinDictionary {
        id: "anatomy_abdomen",
        raw: include_str!("../dictionaries/de/anatomy_abdomen.txt"),
        group: DictionaryGroup::Anatomy,
        levels: &[100, 250],
    },
    BuiltinDictionary {
        id: "anatomy_cns",
        raw: include_str!("../dictionaries/de/anatomy_cns.txt"),
        group: DictionaryGroup::Anatomy,
        levels: &[100, 250],
    },
    BuiltinDictionary {
        id: "anatomy_head_neck",
        raw: include_str!("../dictionaries/de/anatomy_head_neck.txt"),
        group: DictionaryGroup::Anatomy,
        levels: &[100, 250],
    },
    BuiltinDictionary {
        id: "anatomy_lower_extremity",
        raw: include_str!("../dictionaries/de/anatomy_lower_extremity.txt"),
        group: DictionaryGroup::Anatomy,
        levels: &[100, 250],
    },
    BuiltinDictionary {
        id: "anatomy_pelvis",
        raw: include_str!("../dictionaries/de/anatomy_pelvis.txt"),
        group: DictionaryGroup::Anatomy,
        levels: &[100, 250],
    },
    BuiltinDictionary {
        id: "anatomy_spine",
        raw: include_str!("../dictionaries/de/anatomy_spine.txt"),
        group: DictionaryGroup::Anatomy,
        levels: &[100, 250],
    },
    BuiltinDictionary {
        id: "anatomy_thorax",
        raw: include_str!("../dictionaries/de/anatomy_thorax.txt"),
        group: DictionaryGroup::Anatomy,
        levels: &[100, 250],
    },
    BuiltinDictionary {
        id: "anatomy_upper_extremity",
        raw: include_str!("../dictionaries/de/anatomy_upper_extremity.txt"),
        group: DictionaryGroup::Anatomy,
        levels: &[100, 250],
    },
    BuiltinDictionary {
        id: "meds_cardiology_generic",
        raw: include_str!("../dictionaries/de/meds_cardiology_generic.txt"),
        group: DictionaryGroup::Medication,
        levels: &[50, 100, 132],
    },
    BuiltinDictionary {
        id: "meds_cardiology_classes",
        raw: include_str!("../dictionaries/de/meds_cardiology_classes.txt"),
        group: DictionaryGroup::Medication,
        levels: &[25, 50, 52],
    },
    // Trade names are opt-in: like every bundled dictionary this one is
    // inactive until the user switches it on (`active_dictionaries` starts
    // empty and nothing auto-activates a builtin).
    BuiltinDictionary {
        id: "meds_cardiology_trade",
        raw: include_str!("../dictionaries/de/meds_cardiology_trade.txt"),
        group: DictionaryGroup::Medication,
        levels: &[25, 50, 60],
    },
    BuiltinDictionary {
        id: "meds_gastro_hepatology_generic",
        raw: include_str!("../dictionaries/de/meds_gastro_hepatology_generic.txt"),
        group: DictionaryGroup::Medication,
        levels: &[25, 50, 100, 145],
    },
    BuiltinDictionary {
        id: "meds_gastro_hepatology_classes",
        raw: include_str!("../dictionaries/de/meds_gastro_hepatology_classes.txt"),
        group: DictionaryGroup::Medication,
        levels: &[25, 46],
    },
    BuiltinDictionary {
        id: "meds_gastro_hepatology_trade",
        raw: include_str!("../dictionaries/de/meds_gastro_hepatology_trade.txt"),
        group: DictionaryGroup::Medication,
        levels: &[],
    },
    BuiltinDictionary {
        id: "meds_neurology_generic",
        raw: include_str!("../dictionaries/de/meds_neurology_generic.txt"),
        group: DictionaryGroup::Medication,
        levels: &[25, 50, 100, 175],
    },
    BuiltinDictionary {
        id: "meds_neurology_classes",
        raw: include_str!("../dictionaries/de/meds_neurology_classes.txt"),
        group: DictionaryGroup::Medication,
        levels: &[25, 44],
    },
    BuiltinDictionary {
        id: "meds_neurology_trade",
        raw: include_str!("../dictionaries/de/meds_neurology_trade.txt"),
        group: DictionaryGroup::Medication,
        levels: &[],
    },
    BuiltinDictionary {
        id: "meds_pneumology_generic",
        raw: include_str!("../dictionaries/de/meds_pneumology_generic.txt"),
        group: DictionaryGroup::Medication,
        levels: &[25, 50, 100, 138],
    },
    BuiltinDictionary {
        id: "meds_pneumology_classes",
        raw: include_str!("../dictionaries/de/meds_pneumology_classes.txt"),
        group: DictionaryGroup::Medication,
        levels: &[25, 41],
    },
    BuiltinDictionary {
        id: "meds_pneumology_trade",
        raw: include_str!("../dictionaries/de/meds_pneumology_trade.txt"),
        group: DictionaryGroup::Medication,
        levels: &[25, 27],
    },
];

/// Dictionary metadata surfaced to the settings UI.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct DictionaryInfo {
    pub id: String,
    /// Display name for user-created dictionaries. `None` for bundled ones —
    /// the frontend resolves those via i18n from the id.
    pub name: Option<String>,
    /// Whether this is a compiled-in dictionary (word list resettable but not
    /// deletable) as opposed to a user-created one.
    pub builtin: bool,
    /// Number of words the dictionary currently contributes (after the tier
    /// cutoff and the user's additions and removals).
    pub word_count: u32,
    /// Whether a bundled dictionary deviates from its shipped word list.
    pub modified: bool,
    /// Which settings section this belongs to.
    pub group: DictionaryGroup,
    /// Selectable tier sizes, ascending; empty for an all-or-nothing list.
    pub levels: Vec<u32>,
    /// The tier currently in use, or `None` when the list has no tiers.
    pub selected_level: Option<u32>,
    /// Whether its whole list takes part in post-correction.
    pub fuzzy_enabled: bool,
    /// Whether it may offer terms as model context.
    pub context_enabled: bool,
    /// How many of its terms the selected tier lets into the context.
    pub context_word_count: u32,
}

/// One word in the dictionary editor. `builtin` distinguishes shipped words
/// (removing hides them) from user-added ones (removing deletes them).
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct DictionaryWord {
    pub word: String,
    pub builtin: bool,
}

fn parse_words(raw: &str) -> impl Iterator<Item = &str> {
    raw.lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
}

fn builtin_by_id(id: &str) -> Option<&'static BuiltinDictionary> {
    BUILTIN.iter().find(|d| d.id == id)
}

fn custom_by_id<'a>(settings: &'a AppSettings, id: &str) -> Option<&'a CustomDictionary> {
    settings.custom_dictionaries.iter().find(|d| d.id == id)
}

/// Case-sensitive containment check; dictionaries are small enough for linear
/// scans everywhere in this module.
fn contains(words: &[String], word: &str) -> bool {
    words.iter().any(|w| w == word)
}

/// The tier a dictionary is set to, and how many terms that takes.
///
/// An unset level means the *smallest* tier rather than the whole list: a
/// module the user just switched on should cost the model as little context as
/// possible until they ask for more. Untiered lists return `None` and are used
/// whole.
fn selected_level(settings: &AppSettings, builtin: &BuiltinDictionary) -> Option<u32> {
    let levels = builtin.levels;
    let first = *levels.first()?;
    let stored = settings.dictionary_levels.get(builtin.id).copied();
    // Ignore a stored level the list no longer offers (tiers can change between
    // releases) rather than silently truncating to something arbitrary.
    Some(match stored {
        Some(level) if levels.contains(&level) => level,
        _ => first,
    })
}

/// The words a dictionary holds, in priority order: user-added words first,
/// then the bundled list minus hidden words. Returns `None` for unknown ids.
///
/// This is the **whole** list — no tier is applied. Tiers exist to keep the
/// model's context small, and post-correction has no such cost, so it compares
/// against everything the dictionary carries. [`context_words`] is the
/// tier-limited view used for the context.
pub fn effective_words(settings: &AppSettings, id: &str) -> Option<Vec<DictionaryWord>> {
    if let Some(builtin) = builtin_by_id(id) {
        let empty = DictionaryCustomization::default();
        let customization = settings.dictionary_customizations.get(id).unwrap_or(&empty);
        let mut words: Vec<DictionaryWord> = customization
            .added_words
            .iter()
            .map(|w| DictionaryWord {
                word: w.clone(),
                builtin: false,
            })
            .collect();
        for word in parse_words(builtin.raw) {
            if !contains(&customization.hidden_words, word) && !words.iter().any(|w| w.word == word)
            {
                words.push(DictionaryWord {
                    word: word.to_string(),
                    builtin: true,
                });
            }
        }
        return Some(words);
    }
    custom_by_id(settings, id).map(|dict| {
        dict.words
            .iter()
            .map(|w| DictionaryWord {
                word: w.clone(),
                builtin: false,
            })
            .collect()
    })
}

/// The words a dictionary may offer as model context: the same list as
/// [`effective_words`], cut to the selected tier.
///
/// The cutoff is applied to the *shipped* list before hiding, so "tier 100"
/// keeps meaning "this module's top 100 terms" — hiding one does not pull the
/// 101st in behind it. User-added words are never cut: someone typed them in
/// because the model got them wrong, which is exactly what context is for.
pub fn context_words(settings: &AppSettings, id: &str) -> Option<Vec<DictionaryWord>> {
    let Some(builtin) = builtin_by_id(id) else {
        return effective_words(settings, id);
    };
    let empty = DictionaryCustomization::default();
    let customization = settings.dictionary_customizations.get(id).unwrap_or(&empty);
    let mut words: Vec<DictionaryWord> = customization
        .added_words
        .iter()
        .map(|w| DictionaryWord {
            word: w.clone(),
            builtin: false,
        })
        .collect();
    let cutoff = selected_level(settings, builtin)
        .map(|level| level as usize)
        .unwrap_or(usize::MAX);
    for word in parse_words(builtin.raw).take(cutoff) {
        if !contains(&customization.hidden_words, word) && !words.iter().any(|w| w.word == word) {
            words.push(DictionaryWord {
                word: word.to_string(),
                builtin: true,
            });
        }
    }
    Some(words)
}

/// The dedup key for a term. Case-insensitive, because the same term differing
/// only in capitalisation is one term. It deliberately stops there: no trimming
/// of hyphens, no folding of umlauts, no stemming — in German medical
/// vocabulary those distinctions separate genuinely different terms rather than
/// spellings of one ("AV-Block" and "AV Block" are not interchangeable).
fn dedup_key(word: &str) -> String {
    word.to_lowercase()
}

/// Every term one dictionary holds, in priority order (earlier line = higher
/// ASR priority). Used by post-correction, which takes the whole list.
fn dictionary_terms(settings: &AppSettings, id: &str) -> Vec<String> {
    effective_words(settings, id)
        .unwrap_or_default()
        .into_iter()
        .map(|entry| entry.word)
        .collect()
}

/// The terms one dictionary may offer as context, cut to its selected tier.
fn dictionary_context_terms(settings: &AppSettings, id: &str) -> Vec<String> {
    context_words(settings, id)
        .unwrap_or_default()
        .into_iter()
        .map(|entry| entry.word)
        .collect()
}

/// Whether a dictionary takes part in post-correction.
pub fn is_fuzzy_enabled(settings: &AppSettings, id: &str) -> bool {
    settings.active_dictionaries.iter().any(|x| x == id)
}

/// Whether a dictionary may offer terms as model context. Independent of
/// post-correction: either, both or neither is a valid combination.
pub fn is_context_enabled(settings: &AppSettings, id: &str) -> bool {
    settings.context_dictionaries.iter().any(|x| x == id)
}

/// **The active vocabulary pool**: the user's personal words plus every term of
/// every *active* dictionary, up to each one's selected tier. Nothing from an
/// inactive dictionary appears here.
///
/// This is the full pool, deduplicated, and it is what downstream fuzzy
/// post-correction works against. It is deliberately *not* what goes into the
/// model's context — see [`context_vocabulary`] for that much smaller slice.
///
/// A term can legitimately belong to several modules (Kardiologie and Anatomie
/// both carry "Aortenisthmus"), so the merge deduplicates; the first spelling
/// encountered wins.
pub fn full_vocabulary(settings: &AppSettings) -> Vec<String> {
    let mut words: Vec<String> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    let mut push = |word: String, words: &mut Vec<String>| {
        if seen.insert(dedup_key(&word)) {
            words.push(word);
        }
    };
    for word in &settings.custom_words {
        push(word.clone(), &mut words);
    }
    for id in &settings.active_dictionaries {
        for word in dictionary_terms(settings, id) {
            push(word, &mut words);
        }
    }
    words
}

/// **The ASR context vocabulary**: at most `budget` terms selected from the
/// active pool, to be handed to a speech model as prompt/context.
///
/// Only dictionaries set to [`DictionaryUsage::ContextAndFuzzy`] take part;
/// one set to `FuzzyOnly` keeps every term in [`full_vocabulary`] but offers
/// none to the model. That setting is a *permission*, not a promise: the
/// budget, the tokenizer measurement and the model's own window still decide
/// how many terms are actually sent.
///
/// The budget is small on purpose (a model's prompt window is finite), so *how*
/// the slots are filled decides whether activating a module has any effect at
/// all. Two rules:
///
/// 1. **Personal words first.** A word the user typed in by hand is almost
///    always there because the model got it wrong, which makes it the single
///    strongest signal available. Personal words therefore take slots before
///    any dictionary does. Edge case: if the user has more personal words than
///    the whole budget, they fill it and no module contributes — the personal
///    list wins outright, and the modules still reach [`full_vocabulary`] for
///    fuzzy correction.
///
/// 2. **Modules share the rest by round-robin**, one term per module per round,
///    each module contributing in its own priority order. Draining module A
///    fully before starting module B would let one 250-term list swallow the
///    entire budget and silently neutralise every other module the user
///    activated — the exact failure this function exists to prevent.
///
/// Shares are *equal*, not weighted by tier size. A tier says how deep a
/// module's candidate pool goes, not how much context it may claim; weighting
/// by tier would hand the biggest list the most slots and recreate the
/// domination problem. A module that runs out early simply stops taking part,
/// and the remaining modules keep drawing until the budget is full.
///
/// Deduplication happens *during* allocation, so a term carried by two modules
/// costs one slot, not two.
///
/// The result is deterministic: module order follows
/// [`AppSettings::active_dictionaries`], which is stable across runs.
pub fn context_vocabulary(settings: &AppSettings, budget: usize) -> Vec<String> {
    let mut context: Vec<String> = Vec::with_capacity(budget.min(256));
    let mut seen: HashSet<String> = HashSet::new();
    if budget == 0 {
        return context;
    }

    for word in &settings.custom_words {
        if context.len() >= budget {
            return context;
        }
        if seen.insert(dedup_key(word)) {
            context.push(word.clone());
        }
    }

    let modules: Vec<Vec<String>> = settings
        .context_dictionaries
        .iter()
        .map(|id| dictionary_context_terms(settings, id))
        .filter(|terms| !terms.is_empty())
        .collect();
    let mut cursors = vec![0usize; modules.len()];

    // One pass per round; a round that places nothing means every module is
    // exhausted (or fully duplicated), so there is nothing left to hand out.
    loop {
        let mut placed_this_round = false;
        for (index, terms) in modules.iter().enumerate() {
            if context.len() >= budget {
                return context;
            }
            while cursors[index] < terms.len() {
                let term = &terms[cursors[index]];
                cursors[index] += 1;
                if seen.insert(dedup_key(term)) {
                    context.push(term.clone());
                    placed_this_round = true;
                    break;
                }
            }
        }
        if !placed_this_round {
            return context;
        }
    }
}

/// The part of the active pool that did *not* make it into the model context,
/// i.e. what downstream fuzzy correction still has to catch.
///
/// Compared case-insensitively rather than by exact string: the two lists
/// deduplicate in different orders, so the same term can survive in `full` with
/// one capitalisation and in `context` with another.
pub fn fuzzy_only_vocabulary(full: &[String], context: &[String]) -> Vec<String> {
    let in_context: HashSet<String> = context.iter().map(|w| dedup_key(w)).collect();
    full.iter()
        .filter(|word| !in_context.contains(&dedup_key(word)))
        .cloned()
        .collect()
}

/// Every term that is *allowed* to be selected as model context: the personal
/// words plus the terms of active dictionaries set to context use,
/// deduplicated.
///
/// This is a ceiling, not a prediction — it ignores the budget and the token
/// window on purpose, because it answers "what could reach the model?", which
/// is the question the settings overview asks. A term carried by two
/// dictionaries is eligible as soon as *one* of them allows context.
pub fn context_eligible_vocabulary(settings: &AppSettings) -> Vec<String> {
    let mut words: Vec<String> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    for word in settings.custom_words.iter().cloned().chain(
        settings
            .context_dictionaries
            .iter()
            .flat_map(|id| dictionary_context_terms(settings, id)),
    ) {
        if seen.insert(dedup_key(&word)) {
            words.push(word);
        }
    }
    words
}

fn info_for(settings: &AppSettings, id: &str) -> Option<DictionaryInfo> {
    let word_count = effective_words(settings, id)?.len() as u32;
    if let Some(builtin) = builtin_by_id(id) {
        let modified = settings
            .dictionary_customizations
            .get(id)
            .map(|c| *c != DictionaryCustomization::default())
            .unwrap_or(false);
        Some(DictionaryInfo {
            id: id.to_string(),
            name: None,
            builtin: true,
            word_count,
            modified,
            group: builtin.group,
            levels: builtin.levels.to_vec(),
            selected_level: selected_level(settings, builtin),
            fuzzy_enabled: is_fuzzy_enabled(settings, id),
            context_enabled: is_context_enabled(settings, id),
            context_word_count: context_words(settings, id).map(|w| w.len()).unwrap_or(0) as u32,
        })
    } else {
        custom_by_id(settings, id).map(|dict| DictionaryInfo {
            id: dict.id.clone(),
            name: Some(dict.name.clone()),
            builtin: false,
            word_count,
            modified: false,
            // A user's own list is whatever they put in it — no tiers.
            group: DictionaryGroup::Specialty,
            levels: Vec::new(),
            selected_level: None,
            fuzzy_enabled: is_fuzzy_enabled(settings, id),
            context_enabled: is_context_enabled(settings, id),
            context_word_count: word_count,
        })
    }
}

fn all_infos(settings: &AppSettings) -> Vec<DictionaryInfo> {
    BUILTIN
        .iter()
        .map(|d| d.id.to_string())
        .chain(settings.custom_dictionaries.iter().map(|d| d.id.clone()))
        .filter_map(|id| info_for(settings, &id))
        .collect()
}

/// Normalize a word the way the frontend does: strip characters that would
/// break prompts, collapse whitespace, trim.
fn normalize_word(word: &str) -> String {
    let cleaned: String = word.replace(['<', '>', '"', '\''], "");
    cleaned.split_whitespace().collect::<Vec<_>>().join(" ")
}

// ---------------------------------------------------------------------------
// Tauri commands
// ---------------------------------------------------------------------------

/// One active dictionary as the overview shows it.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct ActiveDictionarySummary {
    pub id: String,
    /// Display name for user-created dictionaries; bundled ones are named by
    /// the frontend from the id.
    pub name: Option<String>,
    /// Selected tier, for tiered dictionaries. Applies to the context only.
    pub level: Option<u32>,
    /// Terms this dictionary holds (the whole list).
    pub terms: u32,
    /// Terms it may offer as context at the selected tier; 0 when it is not
    /// part of context selection.
    pub context_terms: u32,
    /// Whether its whole list takes part in post-correction.
    pub fuzzy_enabled: bool,
    /// Whether it may offer terms as model context.
    pub context_enabled: bool,
}

/// What the active vocabulary adds up to, for the settings overview.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct VocabularySummary {
    /// Every dictionary taking part in either step, in settings order.
    pub active: Vec<ActiveDictionarySummary>,
    pub personal_words: u32,
    /// Unique terms across both steps — what "active vocabulary" adds up to.
    pub total_terms: u32,
    /// Terms post-correction can draw on: the whole list of every dictionary
    /// switched on for it, plus the personal words. Terms that were sent to
    /// the model stay available for the comparison afterwards.
    pub fuzzy_terms: u32,
    /// Terms *allowed* to be selected as model context — an upper bound, not
    /// what a dictation actually sends.
    pub context_terms: u32,
    /// Of those, the personal words, which are not part of any dictionary in
    /// the list and therefore unaffected by the per-dictionary setting.
    pub context_personal_words: u32,
    /// Terms counted in more than one source, merged into one.
    pub duplicates_merged: u32,
}

pub fn vocabulary_summary(settings: &AppSettings) -> VocabularySummary {
    // Either step counts as "taking part", so a context-only dictionary is
    // listed too — showing only the post-correction side would hide what the
    // model is actually being fed.
    let mut ids: Vec<&String> = settings.active_dictionaries.iter().collect();
    for id in &settings.context_dictionaries {
        if !ids.contains(&id) {
            ids.push(id);
        }
    }
    let active: Vec<ActiveDictionarySummary> = ids
        .into_iter()
        .filter_map(|id| {
            let info = info_for(settings, id)?;
            Some(ActiveDictionarySummary {
                id: info.id,
                name: info.name,
                level: info.selected_level,
                terms: info.word_count,
                context_terms: if info.context_enabled {
                    info.context_word_count
                } else {
                    0
                },
                fuzzy_enabled: info.fuzzy_enabled,
                context_enabled: info.context_enabled,
            })
        })
        .collect();
    let personal_words = settings.custom_words.len() as u32;
    let fuzzy_terms = full_vocabulary(settings).len() as u32;
    let context_pool = context_eligible_vocabulary(settings);
    // The two pools overlap, so the unique total is their merged size rather
    // than a sum — a context-only dictionary adds to it, a shared term does not.
    let mut unique: HashSet<String> = full_vocabulary(settings)
        .iter()
        .map(|w| dedup_key(w))
        .collect();
    unique.extend(context_pool.iter().map(|w| dedup_key(w)));
    let total_terms = unique.len() as u32;
    let counted: u32 = personal_words + active.iter().map(|a| a.terms).sum::<u32>();
    VocabularySummary {
        active,
        personal_words,
        total_terms,
        fuzzy_terms,
        context_terms: context_pool.len() as u32,
        context_personal_words: personal_words,
        duplicates_merged: counted.saturating_sub(total_terms),
    }
}

#[tauri::command]
#[specta::specta]
pub fn get_vocabulary_summary(app: AppHandle) -> VocabularySummary {
    vocabulary_summary(&settings::get_settings(&app))
}

#[tauri::command]
#[specta::specta]
pub fn list_dictionaries(app: AppHandle) -> Vec<DictionaryInfo> {
    all_infos(&settings::get_settings(&app))
}

/// Switch a tiered dictionary to one of its offered tiers.
#[tauri::command]
#[specta::specta]
pub fn set_dictionary_level(app: AppHandle, id: String, level: u32) -> Result<(), String> {
    let builtin =
        builtin_by_id(&id).ok_or_else(|| format!("'{}' is not a bundled dictionary", id))?;
    if !builtin.levels.contains(&level) {
        return Err(format!("'{}' has no level {}", id, level));
    }
    let mut settings = settings::get_settings(&app);
    settings.dictionary_levels.insert(id, level);
    settings::write_settings(&app, settings);
    Ok(())
}

/// Put every dictionary into post-correction, or take every one out.
///
/// The switch does what it says in both directions: on enrols everything that
/// exists (and, while it stays on, anything imported later); off empties the
/// list again. Individual dictionaries can be switched back on afterwards —
/// that leaves the switch in its "some of them" state and does not change
/// whether new imports are enrolled.
///
/// Context selection is untouched either way: that step shapes what the model
/// writes, so it is never changed on someone's behalf.
#[tauri::command]
#[specta::specta]
pub fn set_fuzzy_all_dictionaries(app: AppHandle, enabled: bool) {
    let mut settings = settings::get_settings(&app);
    apply_fuzzy_all(&mut settings, enabled);
    settings::write_settings(&app, settings);
}

/// The command's work without the app handle, so the rule is testable.
pub fn apply_fuzzy_all(settings: &mut AppSettings, enabled: bool) {
    settings.fuzzy_all_dictionaries = enabled;
    if !enabled {
        settings.active_dictionaries.clear();
        return;
    }
    let ids: Vec<String> = BUILTIN
        .iter()
        .map(|d| d.id.to_string())
        .chain(settings.custom_dictionaries.iter().map(|d| d.id.clone()))
        .collect();
    for id in ids {
        if !contains(&settings.active_dictionaries, &id) {
            settings.active_dictionaries.push(id);
        }
    }
}

#[tauri::command]
#[specta::specta]
pub fn get_dictionary_words(app: AppHandle, id: String) -> Result<Vec<DictionaryWord>, String> {
    effective_words(&settings::get_settings(&app), &id)
        .ok_or_else(|| format!("unknown dictionary '{}'", id))
}

#[tauri::command]
#[specta::specta]
pub fn add_dictionary_word(app: AppHandle, id: String, word: String) -> Result<(), String> {
    let word = normalize_word(&word);
    if word.is_empty() || word.len() > 100 {
        return Err("invalid word".to_string());
    }
    let mut settings = settings::get_settings(&app);
    if let Some(builtin) = builtin_by_id(&id) {
        let in_bundled = parse_words(builtin.raw).any(|w| w == word);
        let customization = settings
            .dictionary_customizations
            .entry(id.clone())
            .or_default();
        // Re-adding a hidden bundled word just unhides it; only words the
        // bundled list doesn't carry are tracked as additions.
        customization.hidden_words.retain(|w| w != &word);
        if !in_bundled && !contains(&customization.added_words, &word) {
            customization.added_words.push(word);
        }
        // Drop an empty customization so `modified` stays accurate.
        if *customization == DictionaryCustomization::default() {
            settings.dictionary_customizations.remove(&id);
        }
    } else if let Some(dict) = settings.custom_dictionaries.iter_mut().find(|d| d.id == id) {
        if !contains(&dict.words, &word) {
            dict.words.push(word);
        }
    } else {
        return Err(format!("unknown dictionary '{}'", id));
    }
    settings::write_settings(&app, settings);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn remove_dictionary_word(app: AppHandle, id: String, word: String) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);
    if let Some(builtin) = builtin_by_id(&id) {
        let in_bundled = parse_words(builtin.raw).any(|w| w == word);
        let customization = settings
            .dictionary_customizations
            .entry(id.clone())
            .or_default();
        let was_added = contains(&customization.added_words, &word);
        customization.added_words.retain(|w| w != &word);
        if in_bundled && !contains(&customization.hidden_words, &word) {
            customization.hidden_words.push(word);
        } else if !was_added && !in_bundled {
            return Err("word not found".to_string());
        }
        if *customization == DictionaryCustomization::default() {
            settings.dictionary_customizations.remove(&id);
        }
    } else if let Some(dict) = settings.custom_dictionaries.iter_mut().find(|d| d.id == id) {
        dict.words.retain(|w| w != &word);
    } else {
        return Err(format!("unknown dictionary '{}'", id));
    }
    settings::write_settings(&app, settings);
    Ok(())
}

/// Drop all user edits of a bundled dictionary, restoring the shipped list.
#[tauri::command]
#[specta::specta]
pub fn reset_dictionary(app: AppHandle, id: String) -> Result<(), String> {
    if builtin_by_id(&id).is_none() {
        return Err(format!("'{}' is not a bundled dictionary", id));
    }
    let mut settings = settings::get_settings(&app);
    settings.dictionary_customizations.remove(&id);
    settings::write_settings(&app, settings);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn create_custom_dictionary(app: AppHandle, name: String) -> Result<DictionaryInfo, String> {
    create_custom_dictionary_with_words(&app, name, Vec::new())
}

fn create_custom_dictionary_with_words(
    app: &AppHandle,
    name: String,
    words: Vec<String>,
) -> Result<DictionaryInfo, String> {
    let name = name.trim().to_string();
    if name.is_empty() || name.len() > 80 {
        return Err("invalid name".to_string());
    }
    let mut settings = settings::get_settings(app);
    let id = format!(
        "custom_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or_default()
    );
    settings.custom_dictionaries.push(CustomDictionary {
        id: id.clone(),
        name,
        words,
    });
    // A freshly created dictionary is what the user is about to use, so it
    // joins post-correction — and, while the "all dictionaries" default is on,
    // it must, because that is what the switch promises. It does *not* join
    // context selection on its own: that step is the one that shapes what the
    // model writes, so it stays an explicit choice.
    if !contains(&settings.active_dictionaries, &id) {
        settings.active_dictionaries.push(id.clone());
    }
    let info = info_for(&settings, &id).expect("just inserted");
    settings::write_settings(app, settings);
    Ok(info)
}

#[tauri::command]
#[specta::specta]
pub fn rename_custom_dictionary(app: AppHandle, id: String, name: String) -> Result<(), String> {
    let name = name.trim().to_string();
    if name.is_empty() || name.len() > 80 {
        return Err("invalid name".to_string());
    }
    let mut settings = settings::get_settings(&app);
    let dict = settings
        .custom_dictionaries
        .iter_mut()
        .find(|d| d.id == id)
        .ok_or_else(|| format!("unknown dictionary '{}'", id))?;
    dict.name = name;
    settings::write_settings(&app, settings);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn delete_custom_dictionary(app: AppHandle, id: String) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);
    let before = settings.custom_dictionaries.len();
    settings.custom_dictionaries.retain(|d| d.id != id);
    if settings.custom_dictionaries.len() == before {
        return Err(format!("unknown dictionary '{}'", id));
    }
    settings.active_dictionaries.retain(|a| a != &id);
    settings.context_dictionaries.retain(|a| a != &id);
    settings.dictionary_customizations.remove(&id);
    settings::write_settings(&app, settings);
    Ok(())
}

/// Write the dictionary's current (effective) word list to `path` as plain
/// text — the same format the bundled dictionaries use, so exports can be
/// re-imported anywhere.
#[tauri::command]
#[specta::specta]
pub fn export_dictionary(app: AppHandle, id: String, path: String) -> Result<(), String> {
    let settings = settings::get_settings(&app);
    let words =
        effective_words(&settings, &id).ok_or_else(|| format!("unknown dictionary '{}'", id))?;
    let mut content =
        String::from("# llmedi dictionary export\n# One term per line, '#' starts a comment.\n\n");
    for entry in words {
        content.push_str(&entry.word);
        content.push('\n');
    }
    std::fs::write(&path, content).map_err(|e| format!("could not write '{}': {}", path, e))
}

/// Field separators recognised in a spreadsheet export, in tie-break order:
/// German Excel writes `;`, "tab separated" exports `\t`, and most
/// English-locale tools `,`. Whichever occurs most often wins; on a tie the
/// earlier entry does, because a `;`-separated file whose text fields contain
/// commas is far likelier than the reverse.
const CSV_DELIMITERS: [char; 3] = [';', '\t', ','];

/// First-column labels that mark a spreadsheet's header row rather than a term.
/// Deliberately tiny: no real medical term is spelled like any of these, so a
/// false positive — silently dropping someone's first word — stays unlikely.
const CSV_HEADER_LABELS: &[&str] = &[
    "begriff",
    "begriffe",
    "fachbegriff",
    "fachbegriffe",
    "wort",
    "wörter",
    "woerter",
    "bezeichnung",
    "term",
    "terms",
    "word",
    "words",
];

/// The delimiter a CSV appears to use, or `None` for a single-column file.
fn detect_delimiter(raw: &str) -> Option<char> {
    let mut best: Option<(char, usize)> = None;
    for &delimiter in CSV_DELIMITERS.iter() {
        let count: usize = raw
            .lines()
            .filter(|line| !line.trim_start().starts_with('#'))
            .map(|line| line.matches(delimiter).count())
            .sum();
        // Strict `>` keeps the earlier (higher-priority) delimiter on a tie.
        if count > 0 && best.is_none_or(|(_, best_count)| count > best_count) {
            best = Some((delimiter, count));
        }
    }
    best.map(|(delimiter, _)| delimiter)
}

/// The first column of one CSV record, unwrapping RFC 4180 quoting (`""` is a
/// literal quote) so a quoted field may itself contain the delimiter.
fn first_field(line: &str, delimiter: Option<char>) -> String {
    let mut chars = line.chars().peekable();
    if chars.peek() == Some(&'"') {
        chars.next();
        let mut field = String::new();
        while let Some(c) = chars.next() {
            if c != '"' {
                field.push(c);
            } else if chars.peek() == Some(&'"') {
                chars.next();
                field.push('"');
            } else {
                break;
            }
        }
        return field;
    }
    match delimiter {
        Some(d) => line.split(d).next().unwrap_or("").to_string(),
        None => line.to_string(),
    }
}

/// Object keys a JSON export may carry the term itself under, most specific
/// first. `name` comes last because it is the likeliest to label something
/// other than the term (a module, a source, a category).
const JSON_TERM_KEYS: &[&str] = &[
    "canonical_term",
    "term",
    "begriff",
    "word",
    "wort",
    "text",
    "value",
    "name",
];

/// Object keys a JSON export may nest its array of terms under.
const JSON_LIST_KEYS: &[&str] = &[
    "terms",
    "words",
    "begriffe",
    "woerter",
    "wörter",
    "vocabulary",
    "entries",
    "items",
];

/// Terms from a JSON word list.
///
/// Accepts an array of strings, an array of objects, or an object nesting
/// either under a known key. From an object the term is read from the first
/// matching [`JSON_TERM_KEYS`] entry — deliberately an allowlist rather than
/// "the first string field", because these exports sit next to fields like
/// `term_id`, `category` and `module_name` that would otherwise be imported as
/// vocabulary. An entry yielding no known key is skipped; if that leaves
/// nothing, the caller reports the file as unusable rather than guessing.
fn parse_json_terms(value: &serde_json::Value) -> Vec<String> {
    let array = match value {
        serde_json::Value::Array(items) => Some(items),
        serde_json::Value::Object(map) => JSON_LIST_KEYS
            .iter()
            .find_map(|key| map.get(*key).and_then(|v| v.as_array())),
        _ => None,
    };
    let Some(array) = array else {
        return Vec::new();
    };

    let mut words: Vec<String> = Vec::new();
    for item in array {
        let term = match item {
            serde_json::Value::String(s) => Some(s.as_str()),
            serde_json::Value::Object(map) => JSON_TERM_KEYS
                .iter()
                .find_map(|key| map.get(*key).and_then(|v| v.as_str())),
            _ => None,
        };
        let Some(term) = term else { continue };
        let normalized = normalize_word(term);
        if !normalized.is_empty() && normalized.len() <= 100 && !contains(&words, &normalized) {
            words.push(normalized);
        }
    }
    words
}

/// Terms from an imported file.
///
/// A `.csv` is read as one record per line with only the first column kept —
/// the remaining columns are metadata the ASR prompt has no use for — and a
/// recognised header row is dropped. Anything else is a plain one-term-per-line
/// list, where the whole line is the term (so a term may contain a comma).
fn parse_import(raw: &str, csv: bool) -> Vec<String> {
    // Excel prefixes its UTF-8 exports with a byte-order mark; left in place it
    // would glue itself to the very first term.
    let raw = raw.strip_prefix('\u{feff}').unwrap_or(raw);

    let mut words: Vec<String> = Vec::new();
    if csv {
        let delimiter = detect_delimiter(raw);
        let mut first = true;
        for line in raw.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let field = normalize_word(&first_field(line, delimiter));
            if first {
                first = false;
                if CSV_HEADER_LABELS.contains(&field.to_lowercase().as_str()) {
                    continue;
                }
            }
            if !field.is_empty() && field.len() <= 100 && !contains(&words, &field) {
                words.push(field);
            }
        }
    } else {
        for word in parse_words(raw) {
            let normalized = normalize_word(word);
            if !normalized.is_empty() && normalized.len() <= 100 && !contains(&words, &normalized) {
                words.push(normalized);
            }
        }
    }
    words
}

/// Import a word list as a new user dictionary, named after the file. Accepts a
/// plain `.txt` list, a `.csv` spreadsheet export (first column only) or a
/// `.json` word list. Returns the created dictionary's info.
#[tauri::command]
#[specta::specta]
pub fn import_dictionary(app: AppHandle, path: String) -> Result<DictionaryInfo, String> {
    let extension = std::path::Path::new(&path)
        .extension()
        .map(|e| e.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default();
    let raw = std::fs::read_to_string(&path).map_err(|e| {
        // read_to_string only fails on I/O or non-UTF-8 bytes; the latter is
        // the common one here (Excel's plain "CSV" is often Latin-1), and the
        // io::Error text alone gives the user nothing to act on.
        if e.kind() == std::io::ErrorKind::InvalidData {
            format!(
                "'{}' is not UTF-8 text — re-export it as \"CSV UTF-8\" or plain text",
                path
            )
        } else {
            format!("could not read '{}': {}", path, e)
        }
    })?;

    let words = if extension == "json" {
        let value: serde_json::Value = serde_json::from_str(&raw)
            .map_err(|e| format!("'{}' is not valid JSON: {}", path, e))?;
        let words = parse_json_terms(&value);
        if words.is_empty() {
            // Distinguish "file is empty" from "we could not tell which field
            // holds the term" — the latter is fixable by the user, but only if
            // we say what we looked for.
            return Err(format!(
                "no terms found in '{}': expected a list of strings, or objects with one of: {}",
                path,
                JSON_TERM_KEYS.join(", ")
            ));
        }
        words
    } else {
        parse_import(&raw, extension == "csv")
    };

    if words.is_empty() {
        return Err("no words found in file".to_string());
    }
    let name = std::path::Path::new(&path)
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "Import".to_string());
    create_custom_dictionary_with_words(&app, name, words)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::settings::get_default_settings;

    #[test]
    fn bundled_dictionaries_parse_nonempty() {
        let settings = get_default_settings();
        for builtin in BUILTIN {
            let words = effective_words(&settings, builtin.id).unwrap();
            assert!(!words.is_empty(), "dictionary {} is empty", builtin.id);
            assert!(words.iter().all(|w| w.builtin));
        }
    }

    #[test]
    fn unknown_ids_yield_none_and_are_skipped() {
        let mut settings = get_default_settings();
        assert!(effective_words(&settings, "does_not_exist").is_none());
        settings.active_dictionaries = vec!["does_not_exist".to_string()];
        assert!(full_vocabulary(&settings).is_empty());
    }

    #[test]
    fn personal_words_come_first_and_dedupe() {
        let id = "internal_cardiology";
        // Take the shared term from the list itself, so retiring or reordering a
        // module cannot silently turn this into a test of nothing.
        let shared = shipped(id)[0].to_string();
        let mut settings = with_active(&[id]);
        settings.custom_words = vec![shared.clone(), "MeinWort".to_string()];
        let merged = full_vocabulary(&settings);
        assert_eq!(&merged[..2], &settings.custom_words[..]);
        assert_eq!(merged.iter().filter(|w| **w == shared).count(), 1);
    }

    #[test]
    fn customization_adds_hides_and_resets() {
        let id = "internal_cardiology";
        let hidden = shipped(id)[0].to_string();
        let mut settings = get_default_settings();
        settings.dictionary_customizations.insert(
            id.to_string(),
            DictionaryCustomization {
                added_words: vec!["Zebra-Wort".to_string()],
                hidden_words: vec![hidden.clone()],
            },
        );
        let words = effective_words(&settings, id).unwrap();
        // Added words come first (highest prompt priority).
        assert_eq!(words[0].word, "Zebra-Wort");
        assert!(!words[0].builtin);
        assert!(words.iter().all(|w| w.word != hidden));
        assert!(info_for(&settings, id).unwrap().modified);

        settings.dictionary_customizations.remove(id);
        let words = effective_words(&settings, id).unwrap();
        assert!(words.iter().any(|w| w.word == hidden));
        assert!(!info_for(&settings, id).unwrap().modified);
    }

    #[test]
    fn custom_dictionary_words_are_used() {
        let mut settings = get_default_settings();
        settings.custom_dictionaries.push(CustomDictionary {
            id: "custom_1".to_string(),
            name: "Gynäkologie".to_string(),
            words: vec!["Hysterektomie".to_string()],
        });
        settings.active_dictionaries = vec!["custom_1".to_string()];
        assert_eq!(
            full_vocabulary(&settings),
            vec!["Hysterektomie".to_string()]
        );
        let info = info_for(&settings, "custom_1").unwrap();
        assert_eq!(info.name.as_deref(), Some("Gynäkologie"));
        assert!(!info.builtin);
    }

    /// Full shipped inventory of a bundled list, ignoring tiers.
    fn shipped(id: &str) -> Vec<&'static str> {
        parse_words(builtin_by_id(id).unwrap().raw).collect()
    }

    /// Dictionaries switched on for both steps, which is what most tests here
    /// are about. [`enable`] sets the two switches apart.
    fn with_active(ids: &[&str]) -> AppSettings {
        let mut settings = get_default_settings();
        settings.active_dictionaries = ids.iter().map(|s| s.to_string()).collect();
        settings.context_dictionaries = settings.active_dictionaries.clone();
        settings
    }

    fn has(words: &[String], term: &str) -> bool {
        words.iter().any(|w| dedup_key(w) == dedup_key(term))
    }

    /// Switch a dictionary on for one step, the other, or both.
    fn enable(settings: &mut AppSettings, id: &str, fuzzy: bool, context: bool) {
        settings.active_dictionaries.retain(|x| x != id);
        settings.context_dictionaries.retain(|x| x != id);
        if fuzzy {
            settings.active_dictionaries.push(id.to_string());
        }
        if context {
            settings.context_dictionaries.push(id.to_string());
        }
    }

    /// A dictionary switched off everywhere is in neither pool.
    #[test]
    fn a_dictionary_off_everywhere_reaches_neither_pool() {
        let mut settings = get_default_settings();
        enable(&mut settings, "internal_cardiology", false, false);
        let term = shipped("internal_cardiology")[0];
        assert!(!has(&full_vocabulary(&settings), term));
        assert!(!has(&context_vocabulary(&settings, 120), term));
        assert!(!has(&context_eligible_vocabulary(&settings), term));
    }

    /// Post-correction only: the whole list is compared against the text, and
    /// nothing at all is offered to the model.
    #[test]
    fn post_correction_only_uses_the_whole_list_and_sends_nothing() {
        let id = "internal_cardiology";
        let mut settings = get_default_settings();
        enable(&mut settings, id, true, false);
        let terms = shipped(id);
        let full = full_vocabulary(&settings);
        // The whole list, not the tier: tiers exist for the context alone.
        assert_eq!(full.len(), terms.len());
        assert!(has(&full, terms[terms.len() - 1]));
        assert!(context_vocabulary(&settings, 120).is_empty());
        assert!(context_eligible_vocabulary(&settings).is_empty());
    }

    /// Context only: the model is offered the tier, and the finished text is
    /// *not* compared against this dictionary. All four combinations are valid.
    #[test]
    fn context_only_offers_the_tier_and_stays_out_of_the_correction_pool() {
        let id = "internal_cardiology";
        let mut settings = get_default_settings();
        enable(&mut settings, id, false, true);
        settings.dictionary_levels.insert(id.to_string(), 100);
        assert!(full_vocabulary(&settings).is_empty());
        assert_eq!(context_eligible_vocabulary(&settings).len(), 100);
        assert!(has(&context_vocabulary(&settings, 120), shipped(id)[0]));
    }

    /// Both steps: the tier limits what the model sees, never what correction
    /// compares against.
    #[test]
    fn both_steps_share_a_dictionary_but_not_its_tier() {
        let id = "internal_cardiology";
        let mut settings = get_default_settings();
        enable(&mut settings, id, true, true);
        settings.dictionary_levels.insert(id.to_string(), 100);
        let terms = shipped(id);
        assert_eq!(full_vocabulary(&settings).len(), terms.len());
        assert_eq!(context_eligible_vocabulary(&settings).len(), 100);
        // A term past the tier: corrected afterwards, never sent.
        let beyond = terms[200];
        assert!(has(&full_vocabulary(&settings), beyond));
        assert!(!has(&context_eligible_vocabulary(&settings), beyond));
    }

    /// Strategy D: what was sent stays in the correction pool.
    #[test]
    fn context_terms_remain_in_the_correction_pool() {
        let id = "internal_cardiology";
        let mut settings = get_default_settings();
        enable(&mut settings, id, true, true);
        let sent = context_vocabulary(&settings, 5);
        assert_eq!(sent.len(), 5);
        let full = full_vocabulary(&settings);
        for term in &sent {
            assert!(has(&full, term), "'{}' left the correction pool", term);
        }
    }

    /// Two dictionaries, opposite settings — each pool holds exactly its own.
    #[test]
    fn the_two_steps_are_decided_per_dictionary() {
        let (ctx, fuzzy) = ("internal_cardiology", "core_medical");
        let mut settings = get_default_settings();
        enable(&mut settings, ctx, false, true);
        enable(&mut settings, fuzzy, true, false);
        let context = context_eligible_vocabulary(&settings);
        let full = full_vocabulary(&settings);
        assert!(has(&context, shipped(ctx)[0]));
        assert!(!has(&full, shipped(ctx)[0]));
        assert!(has(&full, shipped(fuzzy)[0]));
        assert!(!has(&context, shipped(fuzzy)[0]));
    }

    /// A term two dictionaries share is eligible as soon as one of them is in
    /// context selection — and is listed once, not twice.
    #[test]
    fn a_shared_term_is_eligible_once_via_any_context_dictionary() {
        let (with_ctx, without) = ("anatomy_heart_vessels", "anatomy_thorax");
        let shared = shipped(with_ctx)
            .into_iter()
            .find(|term| shipped(without).contains(term))
            .expect("the two anatomy modules overlap");
        let mut settings = get_default_settings();
        enable(&mut settings, without, true, false);
        enable(&mut settings, with_ctx, true, true);
        settings.dictionary_levels.insert(with_ctx.to_string(), 250);
        let allowed = context_eligible_vocabulary(&settings);
        assert_eq!(
            allowed
                .iter()
                .filter(|w| dedup_key(w) == dedup_key(shared))
                .count(),
            1,
            "'{}' must be listed once",
            shared
        );
        // Take the permissive one out of context selection and it is gone from
        // the context while staying in the correction pool via the other.
        enable(&mut settings, with_ctx, true, false);
        assert!(!has(&context_eligible_vocabulary(&settings), shared));
        assert!(has(&full_vocabulary(&settings), shared));
    }

    /// Personal words sit outside the list and stay eligible even when no
    /// dictionary feeds the context — which is what the UI tells the user.
    #[test]
    fn personal_words_stay_eligible_without_any_context_dictionary() {
        let mut settings = get_default_settings();
        enable(&mut settings, "internal_cardiology", true, false);
        settings.custom_words = vec!["Blatt-Schmidt-Zeichen".to_string()];
        assert_eq!(
            context_vocabulary(&settings, 120),
            vec!["Blatt-Schmidt-Zeichen".to_string()]
        );
        let summary = vocabulary_summary(&settings);
        assert_eq!(summary.context_terms, 1);
        assert_eq!(summary.context_personal_words, 1);
    }

    /// The overview's numbers, including a context-only dictionary that is not
    /// in the correction pool at all.
    #[test]
    fn summary_counts_each_step_separately() {
        let (ctx, fuzzy) = ("internal_cardiology", "core_medical");
        let mut settings = get_default_settings();
        enable(&mut settings, ctx, false, true);
        enable(&mut settings, fuzzy, true, true);
        settings.dictionary_levels.insert(ctx.to_string(), 100);
        let summary = vocabulary_summary(&settings);

        assert_eq!(summary.fuzzy_terms, full_vocabulary(&settings).len() as u32);
        assert_eq!(
            summary.context_terms,
            context_eligible_vocabulary(&settings).len() as u32
        );
        // Both dictionaries are listed, each with its own two flags.
        let entry = |id: &str| {
            summary
                .active
                .iter()
                .find(|e| e.id == id)
                .unwrap_or_else(|| panic!("{} missing from the overview", id))
                .clone()
        };
        assert!(!entry(ctx).fuzzy_enabled && entry(ctx).context_enabled);
        assert!(entry(fuzzy).fuzzy_enabled && entry(fuzzy).context_enabled);
        assert_eq!(entry(ctx).context_terms, 100);
        assert_eq!(entry(ctx).terms, shipped(ctx).len() as u32);
        // The unique total spans both pools, so it exceeds either of them.
        assert!(summary.total_terms >= summary.fuzzy_terms);
        assert!(summary.total_terms >= summary.context_terms);
    }

    /// The "all dictionaries" default enrols everything that exists now and
    /// leaves context selection — the risky step — untouched.
    #[test]
    fn the_all_switch_enrols_and_empties_post_correction_only() {
        let mut settings = get_default_settings();
        settings.custom_dictionaries.push(CustomDictionary {
            id: "custom_1".to_string(),
            name: "Eigene".to_string(),
            words: vec!["Testwort".to_string()],
        });
        apply_fuzzy_all(&mut settings, true);
        assert!(settings.fuzzy_all_dictionaries);
        assert_eq!(
            settings.active_dictionaries.len(),
            BUILTIN.len() + 1,
            "every dictionary should take part"
        );
        assert!(settings.context_dictionaries.is_empty());

        // Individually switching one off leaves the rest alone…
        enable(&mut settings, "core_medical", false, false);
        assert!(!is_fuzzy_enabled(&settings, "core_medical"));
        assert!(is_fuzzy_enabled(&settings, "internal_cardiology"));
        // …and switching the master off empties the list.
        apply_fuzzy_all(&mut settings, false);
        assert!(!settings.fuzzy_all_dictionaries);
        assert!(settings.active_dictionaries.is_empty());
        assert!(full_vocabulary(&settings).is_empty());
        // Context selection is never touched by this switch.
        enable(&mut settings, "internal_cardiology", false, true);
        apply_fuzzy_all(&mut settings, true);
        assert!(is_context_enabled(&settings, "internal_cardiology"));
        apply_fuzzy_all(&mut settings, false);
        assert!(is_context_enabled(&settings, "internal_cardiology"));
        assert!(!context_eligible_vocabulary(&settings).is_empty());
    }

    #[test]
    fn every_tiered_module_ships_at_least_its_largest_tier() {
        for builtin in BUILTIN.iter().filter(|b| !b.levels.is_empty()) {
            let terms = parse_words(builtin.raw).count();
            let largest = *builtin.levels.last().unwrap() as usize;
            assert_eq!(
                terms, largest,
                "{} ships {} terms but claims a top tier of {}",
                builtin.id, terms, largest
            );
            assert!(
                builtin.levels.windows(2).all(|w| w[0] < w[1]),
                "{} levels are not ascending",
                builtin.id
            );
        }
    }

    #[test]
    fn tiers_are_nested_and_do_not_overreach() {
        let id = "internal_cardiology";
        let full = shipped(id);
        let mut settings = with_active(&[id]);
        let mut previous: Vec<String> = Vec::new();
        for level in [100u32, 250, 500] {
            settings.dictionary_levels.insert(id.to_string(), level);
            let words: Vec<String> = context_words(&settings, id)
                .unwrap()
                .into_iter()
                .map(|w| w.word)
                .collect();
            // Exactly the tier's size — a 100 must not drag in the whole 500.
            assert_eq!(words.len(), level as usize);
            // …and it is the module's top-N, in shipped order.
            assert_eq!(words, full[..level as usize]);
            // 100 ⊂ 250 ⊂ 500.
            assert!(previous.iter().all(|w| words.contains(w)));
            previous = words;
        }
    }

    #[test]
    fn unset_level_uses_the_smallest_tier() {
        let settings = with_active(&["internal_cardiology"]);
        assert!(settings.dictionary_levels.is_empty());
        assert_eq!(
            context_words(&settings, "internal_cardiology")
                .unwrap()
                .len(),
            100
        );
        // The tier bounds the context only; post-correction sees everything.
        assert_eq!(
            effective_words(&settings, "internal_cardiology")
                .unwrap()
                .len(),
            shipped("internal_cardiology").len()
        );
    }

    #[test]
    fn an_untiered_list_has_no_cutoff() {
        // Every module shipped today is tiered, but the untiered path stays
        // supported for lists that have no meaningful cutoff.
        let untiered = BuiltinDictionary {
            id: "test_untiered",
            raw: "Alpha\nBeta\n",
            group: DictionaryGroup::Specialty,
            levels: &[],
        };
        assert_eq!(selected_level(&get_default_settings(), &untiered), None);
    }

    #[test]
    fn unknown_stored_level_falls_back_to_smallest() {
        let mut settings = with_active(&["internal_cardiology"]);
        settings
            .dictionary_levels
            .insert("internal_cardiology".to_string(), 999);
        assert_eq!(
            context_words(&settings, "internal_cardiology")
                .unwrap()
                .len(),
            100
        );
    }

    #[test]
    fn only_active_dictionaries_reach_the_model() {
        let settings = with_active(&["meds_cardiology_generic"]);
        let words = full_vocabulary(&settings);
        assert!(words.contains(&"Sacubitril/Valsartan".to_string()));
        // Nothing from an inactive module leaks in — trade names above all,
        // since those must never load unless explicitly switched on.
        for id in [
            "meds_cardiology_trade",
            "anatomy_heart_vessels",
            "core_medical",
        ] {
            let foreign = shipped(id);
            let leaked: Vec<_> = words
                .iter()
                .filter(|w| {
                    foreign.contains(&w.as_str())
                        && !shipped("meds_cardiology_generic").contains(&w.as_str())
                })
                .collect();
            assert!(leaked.is_empty(), "{} leaked {:?}", id, leaked);
        }
    }

    #[test]
    fn overlapping_modules_contribute_a_term_once() {
        // "Aortenisthmus" is carried by both the cardiology and the anatomy
        // module; the merged context must list it a single time.
        let settings = with_active(&["internal_cardiology", "anatomy_heart_vessels"]);
        let mut settings = settings;
        settings
            .dictionary_levels
            .insert("internal_cardiology".to_string(), 500);
        settings
            .dictionary_levels
            .insert("anatomy_heart_vessels".to_string(), 250);
        let words = full_vocabulary(&settings);
        let mut lowered: Vec<String> = words.iter().map(|w| w.to_lowercase()).collect();
        let before = lowered.len();
        lowered.sort();
        lowered.dedup();
        assert_eq!(before, lowered.len(), "merged context contains duplicates");
        assert_eq!(
            words
                .iter()
                .filter(|w| w.eq_ignore_ascii_case("Aortenisthmus"))
                .count(),
            1
        );
    }

    #[test]
    fn dedupe_is_case_insensitive_but_not_aggressive() {
        let mut settings = get_default_settings();
        settings.custom_words = vec![
            "Stentimplantation".to_string(),
            "stentimplantation".to_string(),
            // Genuinely different terms that a looser normalisation would merge.
            "Aorta".to_string(),
            "Aortenklappe".to_string(),
            "AV-Block".to_string(),
            "AV Block".to_string(),
        ];
        let words = full_vocabulary(&settings);
        assert_eq!(
            words,
            vec![
                "Stentimplantation",
                "Aorta",
                "Aortenklappe",
                "AV-Block",
                "AV Block"
            ]
        );
    }

    #[test]
    fn multiword_terms_and_abbreviations_survive_intact() {
        let mut settings = with_active(&["internal_cardiology", "anatomy_heart_vessels"]);
        settings
            .dictionary_levels
            .insert("internal_cardiology".to_string(), 500);
        settings
            .dictionary_levels
            .insert("anatomy_heart_vessels".to_string(), 250);
        let words = full_vocabulary(&settings);
        for term in [
            "Sinus transversus pericardii",
            "Musculus papillaris anterior",
            "Torsade-de-pointes-Tachykardie",
            "AV-Knoten-Reentrytachykardie",
        ] {
            assert!(words.iter().any(|w| w == term), "missing {}", term);
        }
        // No entry was split on whitespace on the way through.
        for fragment in ["Sinus", "Musculus", "papillaris"] {
            assert!(
                !words.iter().any(|w| w == fragment),
                "split into {}",
                fragment
            );
        }
    }

    #[test]
    fn bundled_lists_carry_no_metadata_columns() {
        // The harness CSV/JSON carry scores next to each term; only the plain
        // term inventory may ship, so no bundled line may look tabular.
        for builtin in BUILTIN {
            for word in parse_words(builtin.raw) {
                assert!(
                    !word.contains(';') && !word.contains('\t') && !word.contains(','),
                    "{} contains a separator: {:?}",
                    builtin.id,
                    word
                );
                assert!(
                    !word.chars().all(|c| c.is_ascii_digit()),
                    "{} contains a bare number: {:?}",
                    builtin.id,
                    word
                );
            }
        }
    }

    #[test]
    fn hiding_a_term_does_not_pull_in_the_next_tier_member() {
        let id = "internal_cardiology";
        let full = shipped(id);
        let mut settings = with_active(&[id]);
        settings.dictionary_levels.insert(id.to_string(), 100);
        settings.dictionary_customizations.insert(
            id.to_string(),
            DictionaryCustomization {
                added_words: Vec::new(),
                hidden_words: vec![full[0].to_string()],
            },
        );
        let words = context_words(&settings, id).unwrap();
        assert_eq!(words.len(), 99);
        assert!(!words.iter().any(|w| w.word == full[100]));
    }

    #[test]
    fn txt_import_keeps_the_whole_line() {
        // No column splitting outside CSV, so a comma stays part of the term.
        let words = parse_import(
            "# Kommentar\nZustand nach, Reanimation\n\nTroponin\n",
            false,
        );
        assert_eq!(words, vec!["Zustand nach, Reanimation", "Troponin"]);
    }

    #[test]
    fn csv_keeps_only_the_first_column() {
        let raw = "Begriff;Fachgebiet\nTroponin;Kardiologie\nDyspnoe;Pneumologie\n";
        assert_eq!(
            parse_import(raw, true),
            vec!["Troponin".to_string(), "Dyspnoe".to_string()]
        );
    }

    #[test]
    fn csv_handles_comma_and_tab_exports() {
        assert_eq!(
            parse_import("Troponin,Kardiologie\n", true),
            vec!["Troponin"]
        );
        assert_eq!(
            parse_import("Troponin\tKardiologie\n", true),
            vec!["Troponin"]
        );
    }

    #[test]
    fn csv_delimiter_tiebreak_prefers_semicolon() {
        // One `;` and one `,` per line: the `,` belongs to the second column's
        // prose, so splitting on `;` is the correct read.
        let raw = "Troponin;Marker, kardial\nDyspnoe;Atemnot, subjektiv\n";
        assert_eq!(parse_import(raw, true), vec!["Troponin", "Dyspnoe"]);
    }

    #[test]
    fn csv_unwraps_quoted_fields() {
        let raw = "\"Zustand nach, Reanimation\";Notfall\n\"Er sagte \"\"ja\"\"\";x\n";
        assert_eq!(
            parse_import(raw, true),
            vec!["Zustand nach, Reanimation", "Er sagte ja"]
        );
    }

    #[test]
    fn csv_drops_a_header_row_but_keeps_real_terms() {
        assert_eq!(parse_import("Wort\nTroponin\n", true), vec!["Troponin"]);
        // Only the *first* row is eligible, and only for known labels.
        assert_eq!(
            parse_import("Troponin\nWort\n", true),
            vec!["Troponin", "Wort"]
        );
    }

    #[test]
    fn csv_single_column_needs_no_delimiter() {
        assert_eq!(
            parse_import("Troponin\nDyspnoe\n", true),
            vec!["Troponin", "Dyspnoe"]
        );
    }

    #[test]
    fn import_strips_excel_byte_order_mark_and_crlf() {
        let words = parse_import("\u{feff}Begriff;X\r\nTroponin;Y\r\n", true);
        assert_eq!(words, vec!["Troponin"]);
    }

    #[test]
    fn csv_dedupes_and_skips_empty_first_columns() {
        let raw = "Troponin;a\n;leer\nTroponin;b\nDyspnoe;c\n";
        assert_eq!(parse_import(raw, true), vec!["Troponin", "Dyspnoe"]);
    }

    /// How many of `context` each module contributed, counted against that
    /// module's *tier-cut* candidate list rather than its full inventory.
    ///
    /// A term carried by two modules is counted for both, so the shares can sum
    /// to slightly more than the budget — the assertions below allow for that
    /// rather than pretending the modules are disjoint.
    fn share_per_module(settings: &AppSettings, context: &[String], ids: &[&str]) -> Vec<usize> {
        let keys: HashSet<String> = context.iter().map(|w| dedup_key(w)).collect();
        ids.iter()
            .map(|id| {
                dictionary_context_terms(settings, id)
                    .iter()
                    .filter(|t| keys.contains(&dedup_key(t)))
                    .count()
            })
            .collect()
    }

    #[test]
    fn single_module_context_is_its_top_terms_in_order() {
        let id = "internal_cardiology";
        let settings = with_active(&[id]);
        let context = context_vocabulary(&settings, 10);
        assert_eq!(context, shipped(id)[..10].to_vec());
    }

    #[test]
    fn every_active_module_reaches_the_context() {
        let ids = [
            "internal_cardiology",
            "anatomy_heart_vessels",
            "meds_cardiology_generic",
        ];
        let settings = with_active(&ids);
        let context = context_vocabulary(&settings, 120);
        assert_eq!(context.len(), 120);
        let shares = share_per_module(&settings, &context, &ids);
        for (id, share) in ids.iter().zip(&shares) {
            assert!(*share > 0, "{} contributed nothing: {:?}", id, shares);
        }
        // Round-robin over three equally deep modules divides evenly.
        assert_eq!(shares, vec![40, 40, 40]);
    }

    #[test]
    fn a_large_first_module_cannot_block_the_rest() {
        // Medical Core at its 250 tier is registered first and is larger than
        // the whole budget — draining it in order would leave nothing over.
        let ids = ["core_medical", "internal_cardiology"];
        let mut settings = with_active(&ids);
        settings
            .dictionary_levels
            .insert("core_medical".to_string(), 250);
        let context = context_vocabulary(&settings, 120);
        let shares = share_per_module(&settings, &context, &ids);
        assert_eq!(
            shares,
            vec![60, 60],
            "core swallowed the budget: {:?}",
            shares
        );
    }

    #[test]
    fn activated_medication_module_is_never_crowded_out() {
        // A 50-term drug list next to two 250-term clinical modules: small, but
        // it must still land terms in the context.
        let ids = [
            "core_medical",
            "internal_cardiology",
            "meds_cardiology_generic",
        ];
        let mut settings = with_active(&ids);
        settings
            .dictionary_levels
            .insert("core_medical".to_string(), 250);
        settings
            .dictionary_levels
            .insert("internal_cardiology".to_string(), 250);
        settings
            .dictionary_levels
            .insert("meds_cardiology_generic".to_string(), 50);
        let context = context_vocabulary(&settings, 120);
        let shares = share_per_module(&settings, &context, &ids);
        assert_eq!(shares, vec![40, 40, 40], "shares: {:?}", shares);
    }

    #[test]
    fn a_module_that_runs_out_yields_its_slots_to_the_others() {
        // The classes list has only 25 terms at its first tier, far fewer than
        // an equal share of the budget; the rest must not go to waste.
        let ids = ["internal_cardiology", "meds_cardiology_classes"];
        let settings = with_active(&ids);
        let context = context_vocabulary(&settings, 120);
        assert_eq!(context.len(), 120);
        let shares = share_per_module(&settings, &context, &ids);
        assert_eq!(shares, vec![95, 25], "shares: {:?}", shares);
    }

    #[test]
    fn tier_bounds_what_may_enter_the_context() {
        let id = "internal_cardiology";
        let full = shipped(id);
        let beyond_100: HashSet<String> = full[100..].iter().map(|t| dedup_key(t)).collect();

        // Tier 100: nothing past rank 100 may appear, even with budget to spare.
        let settings = with_active(&[id]);
        let context = context_vocabulary(&settings, 120);
        assert_eq!(context.len(), 100);
        assert!(!context.iter().any(|w| beyond_100.contains(&dedup_key(w))));

        // Tier 500: the same budget now reaches past rank 100.
        let mut settings = with_active(&[id]);
        settings.dictionary_levels.insert(id.to_string(), 500);
        let context = context_vocabulary(&settings, 120);
        assert_eq!(context.len(), 120);
        assert!(context.iter().any(|w| beyond_100.contains(&dedup_key(w))));
    }

    #[test]
    fn personal_words_take_context_slots_before_any_module() {
        let mut settings = with_active(&["internal_cardiology", "anatomy_heart_vessels"]);
        settings.custom_words = vec![
            "Blatt-Schmidt-Zeichen".to_string(),
            "Frau Dr. Öztürk".to_string(),
        ];
        let context = context_vocabulary(&settings, 10);
        assert_eq!(&context[..2], &settings.custom_words[..]);
        assert_eq!(context.len(), 10);
    }

    #[test]
    fn many_personal_words_may_fill_the_whole_budget() {
        // Documented edge case: the personal list wins outright, and the
        // modules still reach the fuzzy pool.
        let mut settings = with_active(&["internal_cardiology"]);
        settings.custom_words = (0..150).map(|i| format!("Eigenwort{}", i)).collect();
        let context = context_vocabulary(&settings, 120);
        assert_eq!(context.len(), 120);
        assert!(context.iter().all(|w| w.starts_with("Eigenwort")));

        let full = full_vocabulary(&settings);
        let fuzzy = fuzzy_only_vocabulary(&full, &context);
        assert!(fuzzy.iter().any(|w| w == shipped("internal_cardiology")[0]));
    }

    #[test]
    fn a_term_in_two_modules_costs_one_context_slot() {
        // "Aortenisthmus" is carried by cardiology (tier 500) and anatomy.
        let ids = ["internal_cardiology", "anatomy_heart_vessels"];
        let mut settings = with_active(&ids);
        settings
            .dictionary_levels
            .insert("internal_cardiology".to_string(), 500);
        settings
            .dictionary_levels
            .insert("anatomy_heart_vessels".to_string(), 250);
        // "Aortenisthmus" sits at rank 480 in cardiology and 221 in anatomy, so
        // the budget has to be generous enough to reach it at all — here it is
        // large enough to drain both modules, which also makes the context and
        // the pool directly comparable.
        let context = context_vocabulary(&settings, 800);
        let full = full_vocabulary(&settings);
        assert_eq!(context.len(), full.len());
        assert_eq!(
            context.len(),
            500 + 250 - 1,
            "the shared term was not merged"
        );
        let mut keys: Vec<String> = context.iter().map(|w| dedup_key(w)).collect();
        let before = keys.len();
        keys.sort();
        keys.dedup();
        assert_eq!(before, keys.len(), "a term took two context slots");
        assert_eq!(
            context
                .iter()
                .filter(|w| w.eq_ignore_ascii_case("Aortenisthmus"))
                .count(),
            1
        );
    }

    #[test]
    fn terms_cut_from_the_context_stay_in_the_fuzzy_pool() {
        let id = "internal_cardiology";
        let mut settings = with_active(&[id]);
        settings.dictionary_levels.insert(id.to_string(), 500);
        let full = full_vocabulary(&settings);
        let context = context_vocabulary(&settings, 120);
        let fuzzy = fuzzy_only_vocabulary(&full, &context);

        assert_eq!(full.len(), 500);
        assert_eq!(context.len(), 120);
        // The two are a partition of the pool: nothing lost, nothing doubled.
        assert_eq!(fuzzy.len(), full.len() - context.len());
        let context_keys: HashSet<String> = context.iter().map(|w| dedup_key(w)).collect();
        assert!(!fuzzy.iter().any(|w| context_keys.contains(&dedup_key(w))));
        // A term just past the budget is exactly the case fuzzy has to catch.
        assert!(fuzzy.iter().any(|w| w == shipped(id)[120]));
    }

    #[test]
    fn context_never_exceeds_its_budget_and_zero_means_none() {
        let settings = with_active(&["internal_cardiology", "core_medical"]);
        assert!(context_vocabulary(&settings, 0).is_empty());
        for budget in [1, 7, 33] {
            assert_eq!(context_vocabulary(&settings, budget).len(), budget);
        }
    }

    #[test]
    fn context_is_always_a_subset_of_the_pool() {
        let mut settings = with_active(&[
            "core_medical",
            "internal_cardiology",
            "anatomy_heart_vessels",
            "meds_cardiology_generic",
        ]);
        settings.custom_words = vec!["MeinWort".to_string()];
        let pool: HashSet<String> = full_vocabulary(&settings)
            .iter()
            .map(|w| dedup_key(w))
            .collect();
        for word in context_vocabulary(&settings, 120) {
            assert!(pool.contains(&dedup_key(&word)), "{} not in pool", word);
        }
    }

    #[test]
    fn inactive_modules_contribute_no_context() {
        let settings = with_active(&["meds_cardiology_generic"]);
        let context = context_vocabulary(&settings, 120);
        let trade: HashSet<String> = shipped("meds_cardiology_trade")
            .iter()
            .map(|t| dedup_key(t))
            .collect();
        let generic: HashSet<String> = shipped("meds_cardiology_generic")
            .iter()
            .map(|t| dedup_key(t))
            .collect();
        assert!(!context
            .iter()
            .any(|w| trade.contains(&dedup_key(w)) && !generic.contains(&dedup_key(w))));
    }

    fn json_terms(raw: &str) -> Vec<String> {
        parse_json_terms(&serde_json::from_str(raw).unwrap())
    }

    #[test]
    fn json_accepts_a_plain_string_array() {
        assert_eq!(
            json_terms(r#"["Troponin", "Dyspnoe", "Troponin"]"#),
            vec!["Troponin", "Dyspnoe"]
        );
    }

    #[test]
    fn json_reads_the_harness_shape_without_its_metadata() {
        // Exactly the structure of medical_asr_pilot.json: terms nested under
        // "terms", the word itself in "canonical_term", surrounded by fields
        // that must never become vocabulary.
        let raw = r#"{
          "version": "0.1-pilot",
          "terms": [
            {"term_id": "core.medical:001", "module_id": "core.medical",
             "module_name": "Medical Core", "specialty": "fachübergreifend",
             "rank": 1, "tier": 100, "canonical_term": "reduzierter Allgemeinzustand",
             "term_form": "Mehrwortbegriff", "category": "Klinische_Beschreibung",
             "asr_priority_score": 4.0, "notes": ""}
          ]
        }"#;
        let words = json_terms(raw);
        assert_eq!(words, vec!["reduzierter Allgemeinzustand"]);
        for metadata in [
            "core.medical:001",
            "core.medical",
            "Medical Core",
            "fachübergreifend",
            "Mehrwortbegriff",
            "Klinische_Beschreibung",
        ] {
            assert!(
                !words.iter().any(|w| w == metadata),
                "metadata leaked: {}",
                metadata
            );
        }
    }

    #[test]
    fn json_without_a_recognisable_term_key_yields_nothing() {
        // Guessing "the first string field" here would import the id. Better to
        // return nothing and let the caller explain what was expected.
        let raw = r#"[{"id": "x1", "kategorie": "Herz", "score": 4}]"#;
        assert!(json_terms(raw).is_empty());
    }

    #[test]
    fn json_prefers_the_term_over_a_generic_name_field() {
        let raw = r#"[{"name": "Kardiologie", "canonical_term": "Mitralklappeninsuffizienz"}]"#;
        assert_eq!(json_terms(raw), vec!["Mitralklappeninsuffizienz"]);
    }

    #[test]
    fn json_keeps_multiword_terms_whole() {
        let raw = r#"{"woerter": [{"begriff": "Arteria cerebri media"}]}"#;
        assert_eq!(json_terms(raw), vec!["Arteria cerebri media"]);
    }

    #[test]
    fn json_skips_entries_without_a_term_but_keeps_the_rest() {
        let raw = r#"[{"term": "Sepsis"}, {"score": 3}, 42, {"term": "Ileus"}]"#;
        assert_eq!(json_terms(raw), vec!["Sepsis", "Ileus"]);
    }

    #[test]
    fn summary_counts_active_tiers_and_merged_duplicates() {
        let mut settings = with_active(&["internal_cardiology", "anatomy_heart_vessels"]);
        settings
            .dictionary_levels
            .insert("internal_cardiology".to_string(), 500);
        settings
            .dictionary_levels
            .insert("anatomy_heart_vessels".to_string(), 250);
        settings.custom_words = vec!["Aortenisthmus".to_string(), "MeinWort".to_string()];
        let summary = vocabulary_summary(&settings);
        assert_eq!(summary.active.len(), 2);
        assert_eq!(summary.active[0].level, Some(500));
        assert_eq!(summary.active[0].terms, 500);
        assert_eq!(summary.active[1].terms, 250);
        assert_eq!(summary.personal_words, 2);
        // "Aortenisthmus" is personal AND in both modules: counted three
        // times, kept once.
        assert_eq!(summary.duplicates_merged, 2);
        assert_eq!(summary.total_terms, 2 + 500 + 250 - 2);
        assert_eq!(
            summary.total_terms as usize,
            full_vocabulary(&settings).len()
        );
    }

    #[test]
    fn normalize_strips_dangerous_chars() {
        assert_eq!(normalize_word("  <Tro>ponin\"  T “ "), "Troponin T “");
        assert_eq!(normalize_word("a  b"), "a b");
    }
}
