//! Result records and the three output formats: JSON, CSV, readable report.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt::Write as _;

/// One (case × mode × budget) row. Raw and final transcripts are both kept so a
/// later reader can tell whether a win came from the model or the corrector.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ResultRow {
    pub test_id: String,
    pub model: String,
    pub model_path: String,
    pub model_arch: String,
    pub quantization: String,
    pub mode: String,
    pub context_budget: usize,
    pub specialty: String,
    pub difficulty: String,
    pub reference: String,
    pub raw_transcript: String,
    pub final_transcript: String,
    /// The exact string handed to the model, or empty when none was sent.
    pub context: String,
    pub context_terms: usize,
    pub fuzzy_terms: usize,
    pub wer_raw: f64,
    pub wer_final: f64,
    pub target_terms_total: usize,
    pub target_terms_correct_raw: usize,
    pub target_terms_correct_final: usize,
    /// Target terms that were present in the context.
    pub targets_in_context: usize,
    /// …and were then recognised.
    pub targets_in_context_correct: usize,
    pub false_bias_insertions: usize,
    pub exact_match_final: bool,
    /// Terms the term budget selected, before token fitting.
    pub selected_terms: usize,
    /// Selected terms that did not fit the token window (sent to fuzzy instead).
    pub deferred_terms: usize,
    /// Measured token cost of the context sent; `None` if unmeasured.
    pub context_tokens: Option<usize>,
    /// Tokens the context was allowed this run; `None` if the backend gave none.
    pub context_room: Option<usize>,
    /// Whether the token fit was actually measured and checked.
    pub context_verified: bool,
    /// Replacements fuzzy correction made.
    pub fuzzy_replacements: usize,
    /// Spans fuzzy correction declined because two terms were too close.
    pub fuzzy_ambiguous_skips: usize,
    /// Replacements that put in a term the reference does not contain — errors
    /// the corrector itself introduced.
    pub fuzzy_introduced_terms: usize,
    /// Whether correction made this case's WER worse.
    pub fuzzy_worsened: bool,
    pub medications_total: usize,
    pub medications_correct_raw: usize,
    pub medications_correct_final: usize,
    pub negations_total: usize,
    pub negations_kept_raw: usize,
    pub negations_kept_final: usize,
    pub numbers_total: usize,
    pub numbers_kept_raw: usize,
    pub numbers_kept_final: usize,
    pub confusable_insertions_raw: usize,
    pub confusable_insertions_final: usize,
    /// KV cache derived from the library's allocation rule, in MiB. Not measured.
    pub derived_kv_mib: Option<f64>,
    pub audio_seconds: f64,
    pub transcribe_ms: f64,
    pub fuzzy_ms: f64,
    pub latency_ms: f64,
    /// The run with context failed and was repeated without it (as the app does).
    #[serde(default)]
    pub fell_back_without_context: bool,
    /// Error text when transcription failed even without context — in the app
    /// this dictation would have produced nothing. Empty on success.
    #[serde(default)]
    pub run_error: String,
    /// Pool terms in the raw transcript that were not spoken.
    #[serde(default)]
    pub unspoken_terms_raw: Vec<String>,
    /// Pool terms in the final transcript that were not spoken — invented
    /// findings, drugs or diagnoses, whatever put them there.
    #[serde(default)]
    pub unspoken_terms_final: Vec<String>,
}

/// Averages over a set of rows sharing a model, mode and budget.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Aggregate {
    pub mode: String,
    pub context_budget: usize,
    pub cases: usize,
    pub mean_wer_raw: f64,
    pub mean_wer_final: f64,
    pub median_wer_final: f64,
    pub medical_term_accuracy_raw: f64,
    pub medical_term_accuracy_final: f64,
    /// Recognition rate for targets that were in the context.
    pub context_hit_rate: Option<f64>,
    /// …and for targets that were not — the comparison that shows whether
    /// biasing did anything.
    pub non_context_hit_rate: Option<f64>,
    pub false_bias_insertions: usize,
    pub false_bias_rate_per_case: f64,
    pub false_bias_per_100: f64,
    pub exact_sentence_accuracy: f64,
    pub mean_latency_ms: f64,
    pub mean_rtf: Option<f64>,
    pub medication_accuracy: Option<f64>,
    pub negation_preservation: Option<f64>,
    pub number_preservation: Option<f64>,
    pub confusable_insertions: usize,
    pub fuzzy_introduced_terms: usize,
    pub fuzzy_worsened_cases: usize,
    pub fuzzy_ambiguous_skips: usize,
    pub mean_context_tokens: Option<f64>,
    pub deferred_terms: usize,
    pub unverified_runs: usize,
    pub mean_kv_mib: Option<f64>,
    /// Runs repeated without context after the context run failed.
    pub fallbacks: usize,
    /// Runs that produced no text at all.
    pub failed_runs: usize,
    /// Unspoken pool terms in raw / final transcripts, summed over cases.
    pub unspoken_terms_raw: usize,
    pub unspoken_terms_final: usize,
    /// Cases with at least one unspoken pool term in the final transcript.
    pub cases_with_unspoken_final: usize,
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

fn median(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[mid - 1] + values[mid]) / 2.0
    } else {
        values[mid]
    }
}

fn ratio(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

pub fn aggregate(rows: &[ResultRow]) -> Aggregate {
    let mut wer_final: Vec<f64> = rows.iter().map(|r| r.wer_final).collect();
    let targets_total: usize = rows.iter().map(|r| r.target_terms_total).sum();
    let in_context: usize = rows.iter().map(|r| r.targets_in_context).sum();
    let in_context_ok: usize = rows.iter().map(|r| r.targets_in_context_correct).sum();
    let outside_context = targets_total - in_context;
    let outside_context_ok: usize = rows
        .iter()
        .map(|r| r.target_terms_correct_final - r.targets_in_context_correct)
        .sum();
    let false_bias: usize = rows.iter().map(|r| r.false_bias_insertions).sum();
    let rtfs: Vec<f64> = rows
        .iter()
        .filter(|r| r.audio_seconds > 0.0)
        .map(|r| r.latency_ms / 1000.0 / r.audio_seconds)
        .collect();

    Aggregate {
        mode: rows.first().map(|r| r.mode.clone()).unwrap_or_default(),
        context_budget: rows.first().map(|r| r.context_budget).unwrap_or(0),
        cases: rows.len(),
        mean_wer_raw: mean(&rows.iter().map(|r| r.wer_raw).collect::<Vec<_>>()),
        mean_wer_final: mean(&wer_final),
        median_wer_final: median(&mut wer_final),
        medical_term_accuracy_raw: ratio(
            rows.iter().map(|r| r.target_terms_correct_raw).sum(),
            targets_total,
        ),
        medical_term_accuracy_final: ratio(
            rows.iter().map(|r| r.target_terms_correct_final).sum(),
            targets_total,
        ),
        context_hit_rate: (in_context > 0).then(|| ratio(in_context_ok, in_context)),
        non_context_hit_rate: (outside_context > 0)
            .then(|| ratio(outside_context_ok, outside_context)),
        false_bias_insertions: false_bias,
        false_bias_rate_per_case: ratio(false_bias, rows.len()),
        false_bias_per_100: ratio(false_bias, rows.len()) * 100.0,
        exact_sentence_accuracy: ratio(
            rows.iter().filter(|r| r.exact_match_final).count(),
            rows.len(),
        ),
        mean_latency_ms: mean(&rows.iter().map(|r| r.latency_ms).collect::<Vec<_>>()),
        mean_rtf: (!rtfs.is_empty()).then(|| mean(&rtfs)),
        medication_accuracy: group_rate(rows, |r| {
            (r.medications_correct_final, r.medications_total)
        }),
        negation_preservation: group_rate(rows, |r| (r.negations_kept_final, r.negations_total)),
        number_preservation: group_rate(rows, |r| (r.numbers_kept_final, r.numbers_total)),
        confusable_insertions: rows.iter().map(|r| r.confusable_insertions_final).sum(),
        fuzzy_introduced_terms: rows.iter().map(|r| r.fuzzy_introduced_terms).sum(),
        fuzzy_worsened_cases: rows.iter().filter(|r| r.fuzzy_worsened).count(),
        fuzzy_ambiguous_skips: rows.iter().map(|r| r.fuzzy_ambiguous_skips).sum(),
        mean_context_tokens: {
            let measured: Vec<f64> = rows
                .iter()
                .filter_map(|r| r.context_tokens.map(|t| t as f64))
                .collect();
            (!measured.is_empty()).then(|| mean(&measured))
        },
        deferred_terms: rows.iter().map(|r| r.deferred_terms).sum(),
        unverified_runs: rows
            .iter()
            .filter(|r| r.context_terms > 0 && !r.context_verified)
            .count(),
        mean_kv_mib: {
            let kv: Vec<f64> = rows.iter().filter_map(|r| r.derived_kv_mib).collect();
            (!kv.is_empty()).then(|| mean(&kv))
        },
        fallbacks: rows.iter().filter(|r| r.fell_back_without_context).count(),
        failed_runs: rows.iter().filter(|r| !r.run_error.is_empty()).count(),
        unspoken_terms_raw: rows.iter().map(|r| r.unspoken_terms_raw.len()).sum(),
        unspoken_terms_final: rows.iter().map(|r| r.unspoken_terms_final.len()).sum(),
        cases_with_unspoken_final: rows
            .iter()
            .filter(|r| !r.unspoken_terms_final.is_empty())
            .count(),
    }
}

/// Hits over total for one error group, or `None` when the group never
/// occurred — "0 of 0" is not a 0 % result.
fn group_rate(rows: &[ResultRow], pick: impl Fn(&ResultRow) -> (usize, usize)) -> Option<f64> {
    let (hits, total) = rows
        .iter()
        .map(&pick)
        .fold((0, 0), |(h, t), (rh, rt)| (h + rh, t + rt));
    (total > 0).then(|| ratio(hits, total))
}

/// Escape one CSV field per RFC 4180: quote when it contains a delimiter,
/// quote or newline, and double any inner quote.
fn csv_field(value: &str) -> String {
    if value.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
}

pub const CSV_HEADER: &str = "test_id,model,model_arch,quantization,mode,context_budget,specialty,difficulty,reference,raw_transcript,final_transcript,context,selected_terms,context_terms,deferred_terms,context_tokens,context_room,context_verified,fuzzy_terms,wer_raw,wer_final,target_terms_total,target_terms_correct_raw,target_terms_correct_final,targets_in_context,targets_in_context_correct,false_bias_insertions,exact_match_final,fuzzy_replacements,fuzzy_ambiguous_skips,fuzzy_introduced_terms,fuzzy_worsened,medications_total,medications_correct_raw,medications_correct_final,negations_total,negations_kept_raw,negations_kept_final,numbers_total,numbers_kept_raw,numbers_kept_final,confusable_insertions_raw,confusable_insertions_final,derived_kv_mib,audio_seconds,transcribe_ms,fuzzy_ms,latency_ms,fell_back_without_context,run_error,unspoken_terms_raw,unspoken_terms_final";

pub fn to_csv(rows: &[ResultRow]) -> String {
    let mut out = String::from(CSV_HEADER);
    out.push('\n');
    for r in rows {
        let opt = |v: Option<usize>| v.map(|n| n.to_string()).unwrap_or_default();
        let fields = [
            csv_field(&r.test_id),
            csv_field(&r.model),
            csv_field(&r.model_arch),
            csv_field(&r.quantization),
            csv_field(&r.mode),
            r.context_budget.to_string(),
            csv_field(&r.specialty),
            csv_field(&r.difficulty),
            csv_field(&r.reference),
            csv_field(&r.raw_transcript),
            csv_field(&r.final_transcript),
            csv_field(&r.context),
            r.selected_terms.to_string(),
            r.context_terms.to_string(),
            r.deferred_terms.to_string(),
            opt(r.context_tokens),
            opt(r.context_room),
            r.context_verified.to_string(),
            r.fuzzy_terms.to_string(),
            format!("{:.4}", r.wer_raw),
            format!("{:.4}", r.wer_final),
            r.target_terms_total.to_string(),
            r.target_terms_correct_raw.to_string(),
            r.target_terms_correct_final.to_string(),
            r.targets_in_context.to_string(),
            r.targets_in_context_correct.to_string(),
            r.false_bias_insertions.to_string(),
            r.exact_match_final.to_string(),
            r.fuzzy_replacements.to_string(),
            r.fuzzy_ambiguous_skips.to_string(),
            r.fuzzy_introduced_terms.to_string(),
            r.fuzzy_worsened.to_string(),
            r.medications_total.to_string(),
            r.medications_correct_raw.to_string(),
            r.medications_correct_final.to_string(),
            r.negations_total.to_string(),
            r.negations_kept_raw.to_string(),
            r.negations_kept_final.to_string(),
            r.numbers_total.to_string(),
            r.numbers_kept_raw.to_string(),
            r.numbers_kept_final.to_string(),
            r.confusable_insertions_raw.to_string(),
            r.confusable_insertions_final.to_string(),
            r.derived_kv_mib
                .map(|m| format!("{:.1}", m))
                .unwrap_or_default(),
            format!("{:.3}", r.audio_seconds),
            format!("{:.1}", r.transcribe_ms),
            format!("{:.1}", r.fuzzy_ms),
            format!("{:.1}", r.latency_ms),
            r.fell_back_without_context.to_string(),
            csv_field(&r.run_error),
            csv_field(&r.unspoken_terms_raw.join("; ")),
            csv_field(&r.unspoken_terms_final.join("; ")),
        ];
        out.push_str(&fields.join(","));
        out.push('\n');
    }
    out
}

fn pct(value: f64) -> String {
    format!("{:.1} %", value * 100.0)
}

fn optional_pct(value: Option<f64>) -> String {
    value.map(pct).unwrap_or_else(|| "n/a".to_string())
}

/// Human-readable summary: per-mode headline, a budget table per mode, and the
/// C-vs-D comparison the architecture question hinges on.
pub fn render_report(model_label: &str, rows: &[ResultRow]) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "{}\n{}", model_label, "=".repeat(model_label.len()));
    let _ = writeln!(out, "\n{} result rows\n", rows.len());

    // mode -> budget -> rows
    let mut grouped: BTreeMap<String, BTreeMap<usize, Vec<ResultRow>>> = BTreeMap::new();
    for row in rows {
        grouped
            .entry(row.mode.clone())
            .or_default()
            .entry(row.context_budget)
            .or_default()
            .push(row.clone());
    }

    for (mode, budgets) in &grouped {
        // Spell out what each mode actually did, so a report read weeks later
        // does not depend on remembering the letters.
        match crate::dataset::Mode::parse(mode) {
            Some(parsed) => {
                let _ = writeln!(out, "Mode {} — {}", mode, parsed.label());
            }
            None => {
                let _ = writeln!(out, "Mode {}", mode);
            }
        }
        let _ = writeln!(
            out,
            "  {:>6} | {:>7} | {:>10} | {:>9} | {:>9} | {:>6} | {:>8} | {:>6} | {:>6}",
            "Budget",
            "WER",
            "MedTermAcc",
            "FalseBias",
            "ExactSent",
            "RTF",
            "Tokens",
            "Defer",
            "KV MiB"
        );
        let _ = writeln!(out, "  {}", "-".repeat(92));
        for (budget, budget_rows) in budgets {
            let a = aggregate(budget_rows);
            let _ = writeln!(
                out,
                "  {:>6} | {:>7} | {:>10} | {:>9} | {:>9} | {:>6} | {:>8} | {:>6} | {:>6}",
                budget,
                pct(a.mean_wer_final),
                pct(a.medical_term_accuracy_final),
                format!("{:.2}/100", a.false_bias_per_100),
                pct(a.exact_sentence_accuracy),
                a.mean_rtf
                    .map(|r| format!("{:.2}x", r))
                    .unwrap_or_else(|| "n/a".into()),
                a.mean_context_tokens
                    .map(|t| format!("{:.0}", t))
                    .unwrap_or_else(|| "-".into()),
                a.deferred_terms,
                a.mean_kv_mib
                    .map(|m| format!("{:.0}", m))
                    .unwrap_or_else(|| "-".into()),
            );
        }
        // Error groups: the failures that change clinical meaning, reported
        // apart from WER so a small WER gain cannot hide them.
        let _ = writeln!(
            out,
            "  {:>6} | {:>9} | {:>9} | {:>9} | {:>9} | {:>9} | {:>9} | {:>9} | {:>9} | {:>9}",
            "Budget",
            "Medikam.",
            "Verneing.",
            "Zahlen",
            "Verwechs.",
            "Ungespr.",
            "FuzzyErr",
            "Fuzzy↓",
            "Unklar",
            "Fallb/Err"
        );
        for (budget, budget_rows) in budgets {
            let a = aggregate(budget_rows);
            let _ = writeln!(
                out,
                "  {:>6} | {:>9} | {:>9} | {:>9} | {:>9} | {:>9} | {:>9} | {:>9} | {:>9} | {:>9}",
                budget,
                optional_pct(a.medication_accuracy),
                optional_pct(a.negation_preservation),
                optional_pct(a.number_preservation),
                a.confusable_insertions,
                format!("{}→{}", a.unspoken_terms_raw, a.unspoken_terms_final),
                a.fuzzy_introduced_terms,
                a.fuzzy_worsened_cases,
                a.fuzzy_ambiguous_skips,
                format!("{}/{}", a.fallbacks, a.failed_runs),
            );
            if a.unverified_runs > 0 {
                let _ = writeln!(
                    out,
                    "         ! {} run(s) sent context whose token cost could not be verified",
                    a.unverified_runs
                );
            }
        }
        // Context biasing only demonstrably works if targets that were in the
        // context beat those that were not.
        if let Some((_, any_rows)) = budgets.iter().next_back() {
            let a = aggregate(any_rows);
            if a.context_hit_rate.is_some() || a.non_context_hit_rate.is_some() {
                let _ = writeln!(
                    out,
                    "  target terms in context: {}   not in context: {}",
                    optional_pct(a.context_hit_rate),
                    optional_pct(a.non_context_hit_rate)
                );
            }
        }
        let _ = writeln!(out);
    }

    out.push_str(&render_c_vs_d(&grouped));
    out.push_str(
        "\nLegend: Tokens = measured context tokens (model tokenizer); Defer = selected\n\
         terms that did not fit the token window and went to fuzzy correction instead;\n\
         KV MiB = decoder cache derived from the library's allocation rule, not measured.\n\
         Medikam./Verneing./Zahlen = share preserved in the final transcript;\n\
         Verwechs. = similar-sounding terms that appeared although not spoken;\n\
         Ungespr. = dictionary terms in the output that were never spoken, raw→final\n\
         (every pool term checked, not only the listed confounders);\n\
         FuzzyErr = terms fuzzy correction inserted that the reference lacks;\n\
         Fuzzy↓ = cases where correction made WER worse; Unklar = ambiguous spans skipped;\n\
         Fallb/Err = runs repeated without context after failing / runs with no text.\n",
    );
    out
}

/// The open architecture question: does keeping context terms in the fuzzy pool
/// (D) beat removing them (C)?
fn render_c_vs_d(grouped: &BTreeMap<String, BTreeMap<usize, Vec<ResultRow>>>) -> String {
    let (Some(c), Some(d)) = (grouped.get("C"), grouped.get("D")) else {
        return String::from(
            "Mode C vs D\n-----------\nBoth modes are needed for this comparison; \
             run with --modes C,D (or all).\n",
        );
    };
    let mut out = String::from("Mode C vs D — is it worth keeping context terms in fuzzy?\n");
    out.push_str(&"-".repeat(57));
    out.push('\n');
    let _ = writeln!(
        out,
        "  {:>7} | {:>16} | {:>18} | {:>14} | {:>14}",
        "Budget", "ΔWER (D-C)", "ΔMedTermAcc (D-C)", "ΔFalseBias", "ΔUngesprochen"
    );
    let _ = writeln!(out, "  {}", "-".repeat(79));
    for (budget, c_rows) in c {
        let Some(d_rows) = d.get(budget) else {
            continue;
        };
        let (ca, da) = (aggregate(c_rows), aggregate(d_rows));
        // A negative WER delta and a positive accuracy delta favour D.
        let _ = writeln!(
            out,
            "  {:>7} | {:>+15.2}% | {:>+17.2}% | {:>+14} | {:>+14}",
            budget,
            (da.mean_wer_final - ca.mean_wer_final) * 100.0,
            (da.medical_term_accuracy_final - ca.medical_term_accuracy_final) * 100.0,
            da.false_bias_insertions as i64 - ca.false_bias_insertions as i64,
            da.unspoken_terms_final as i64 - ca.unspoken_terms_final as i64,
        );
    }
    out.push_str(
        "\n  Negative ΔWER and positive ΔMedTermAcc favour D. Weigh any gain\n  \
         against ΔFalseBias: accuracy bought with hallucinated terms is not a win.\n",
    );
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(mode: &str, budget: usize) -> ResultRow {
        ResultRow {
            test_id: "t1".into(),
            model: "m".into(),
            model_path: "/m.gguf".into(),
            model_arch: "qwen3_asr".into(),
            quantization: "Q5_K_M".into(),
            mode: mode.into(),
            context_budget: budget,
            specialty: "internal_cardiology".into(),
            difficulty: "normal".into(),
            reference: "eine Aortenklappenstenose".into(),
            raw_transcript: "eine Aortenstenose".into(),
            final_transcript: "eine Aortenklappenstenose".into(),
            context: String::new(),
            context_terms: budget,
            fuzzy_terms: 10,
            wer_raw: 0.5,
            wer_final: 0.0,
            target_terms_total: 1,
            target_terms_correct_raw: 0,
            target_terms_correct_final: 1,
            targets_in_context: 1,
            targets_in_context_correct: 1,
            false_bias_insertions: 0,
            exact_match_final: true,
            selected_terms: budget,
            deferred_terms: 0,
            context_tokens: Some(budget * 8),
            context_room: Some(64_000),
            context_verified: true,
            fuzzy_replacements: 1,
            fuzzy_ambiguous_skips: 0,
            fuzzy_introduced_terms: 0,
            fuzzy_worsened: false,
            medications_total: 1,
            medications_correct_raw: 0,
            medications_correct_final: 1,
            negations_total: 1,
            negations_kept_raw: 1,
            negations_kept_final: 1,
            numbers_total: 0,
            numbers_kept_raw: 0,
            numbers_kept_final: 0,
            confusable_insertions_raw: 0,
            confusable_insertions_final: 0,
            derived_kv_mib: Some(224.0),
            audio_seconds: 2.0,
            transcribe_ms: 400.0,
            fuzzy_ms: 100.0,
            latency_ms: 500.0,
            fell_back_without_context: false,
            run_error: String::new(),
            unspoken_terms_raw: Vec::new(),
            unspoken_terms_final: Vec::new(),
        }
    }

    #[test]
    fn aggregation_averages_the_expected_fields() {
        let mut low = row("C", 120);
        low.wer_final = 0.0;
        let mut high = row("C", 120);
        high.wer_final = 0.4;
        high.exact_match_final = false;
        high.false_bias_insertions = 1;
        let a = aggregate(&[low, high]);

        assert_eq!(a.cases, 2);
        assert!((a.mean_wer_final - 0.2).abs() < 1e-9);
        assert!((a.median_wer_final - 0.2).abs() < 1e-9);
        assert_eq!(a.medical_term_accuracy_final, 1.0);
        assert_eq!(a.false_bias_insertions, 1);
        assert!((a.false_bias_per_100 - 50.0).abs() < 1e-9);
        assert!((a.exact_sentence_accuracy - 0.5).abs() < 1e-9);
        // 500 ms over 2 s of audio.
        assert!((a.mean_rtf.unwrap() - 0.25).abs() < 1e-9);
    }

    #[test]
    fn context_hit_rate_is_none_without_any_covered_target() {
        let mut r = row("A", 0);
        r.targets_in_context = 0;
        r.targets_in_context_correct = 0;
        let a = aggregate(&[r]);
        assert_eq!(a.context_hit_rate, None);
        // The one target was outside the context and was recognised.
        assert_eq!(a.non_context_hit_rate, Some(1.0));
    }

    #[test]
    fn csv_has_a_header_and_one_line_per_row() {
        let csv = to_csv(&[row("C", 120), row("D", 120)]);
        let lines: Vec<&str> = csv.lines().collect();
        assert_eq!(lines.len(), 3);
        assert_eq!(lines[0], CSV_HEADER);
        assert_eq!(lines[0].split(',').count(), lines[1].split(',').count());
    }

    #[test]
    fn csv_escapes_transcripts_containing_commas_and_quotes() {
        let mut r = row("C", 120);
        r.reference = "Aorta, Klappe".into();
        r.raw_transcript = "er sagte \"ja\"".into();
        let csv = to_csv(&[r]);
        assert!(csv.contains("\"Aorta, Klappe\""));
        assert!(csv.contains("\"er sagte \"\"ja\"\"\""));
        // The escaped comma must not add a column.
        let header_cols = CSV_HEADER.split(',').count();
        let parsed = csv.lines().nth(1).unwrap();
        let mut cols = 1;
        let mut in_quotes = false;
        for c in parsed.chars() {
            match c {
                '"' => in_quotes = !in_quotes,
                ',' if !in_quotes => cols += 1,
                _ => {}
            }
        }
        assert_eq!(cols, header_cols);
    }

    #[test]
    fn json_round_trips() {
        let rows = vec![row("C", 120)];
        let json = serde_json::to_string(&rows).unwrap();
        let back: Vec<ResultRow> = serde_json::from_str(&json).unwrap();
        assert_eq!(back[0].test_id, "t1");
        assert_eq!(back[0].mode, "C");
    }

    #[test]
    fn report_shows_a_budget_table_and_the_c_vs_d_comparison() {
        let rows = vec![row("C", 60), row("C", 120), row("D", 60), row("D", 120)];
        let report = render_report("Qwen3-ASR", &rows);
        assert!(report.contains("Mode C"));
        assert!(report.contains("Mode D"));
        assert!(report.contains("Budget"));
        assert!(report.contains("Mode C vs D"));
        assert!(report.contains("ΔWER"));
    }

    #[test]
    fn error_groups_are_scored_apart_and_absent_groups_are_not_zero() {
        let mut lost_negation = row("C", 120);
        lost_negation.negations_kept_final = 0;
        lost_negation.fuzzy_introduced_terms = 1;
        lost_negation.fuzzy_worsened = true;
        let a = aggregate(&[row("C", 120), lost_negation]);
        assert_eq!(a.negation_preservation, Some(0.5));
        assert_eq!(a.medication_accuracy, Some(1.0));
        // No numbers were spoken: "n/a", not 0 %.
        assert_eq!(a.number_preservation, None);
        assert_eq!(a.fuzzy_introduced_terms, 1);
        assert_eq!(a.fuzzy_worsened_cases, 1);
        let report = render_report("m", &[row("C", 120)]);
        assert!(report.contains("Verneing."));
        assert!(report.contains("Legend"));
    }

    #[test]
    fn failures_fallbacks_and_unspoken_terms_are_counted() {
        let mut fell_back = row("C", 400);
        fell_back.fell_back_without_context = true;
        fell_back.unspoken_terms_final = vec!["Sacubitril/Valsartan".into()];
        let mut failed = row("C", 400);
        failed.run_error = "output truncated".into();
        let a = aggregate(&[row("C", 400), fell_back, failed]);
        assert_eq!(a.fallbacks, 1);
        assert_eq!(a.failed_runs, 1);
        assert_eq!(a.unspoken_terms_final, 1);
        assert_eq!(a.cases_with_unspoken_final, 1);
        // Rows written before these fields existed still load.
        let mut old: serde_json::Value = serde_json::to_value(row("C", 60)).unwrap();
        let object = old.as_object_mut().unwrap();
        object.remove("run_error");
        object.remove("unspoken_terms_final");
        let back: ResultRow = serde_json::from_value(old).unwrap();
        assert!(back.run_error.is_empty());
    }

    #[test]
    fn c_vs_d_section_explains_itself_when_a_mode_is_missing() {
        let report = render_report("m", &[row("A", 0)]);
        assert!(report.contains("--modes C,D"));
    }
}
