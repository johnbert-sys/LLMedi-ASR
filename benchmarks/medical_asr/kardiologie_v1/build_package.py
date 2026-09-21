#!/usr/bin/env python3
"""Build the cardiology recording package from one source table.

Produces, next to this script:
  diktate.jsonl      benchmark dataset for HUMAN recordings (audio/<id>.wav)
  AUFNAHME.md        what to read, and how to record it
and, with --synthetic, a development-only dataset voiced by the local macOS
TTS (never a substitute for human speakers):
  synthetic/dataset.jsonl + synthetic/audio/<id>.wav

Every annotated term is validated against the bundled dictionaries, and every
spoken annotation against its own reference text, so the package cannot drift
from the vocabulary the app actually ships. All patient details are invented.

Usage:
  python3 build_package.py              # validate + write dataset and guide
  python3 build_package.py --synthetic  # also voice it with `say` (local only)
"""

import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DICT_DIR = os.path.join(HERE, "..", "..", "..", "src-tauri", "dictionaries", "de")

# (id, class, reference, targets, medications, negations, numbers,
#  confusable_terms, negative_terms)
# class: normal | hard | medication | negation | confounder
CASES = [
    # --- normale Arztbriefsprache ------------------------------------------
    ("kard_n01", "normal", "Bei Aufnahme zeigte sich eine hochgradige Aortenklappenstenose.",
     ["Aortenklappenstenose"], [], [], [], [], []),
    ("kard_n02", "normal", "In der Echokardiographie fand sich eine Mitralklappeninsuffizienz.",
     ["Echokardiographie", "Mitralklappeninsuffizienz"], [], [], [], [], []),
    ("kard_n03", "normal", "Es besteht ein paroxysmales Vorhofflimmern.",
     ["paroxysmales Vorhofflimmern"], [], [], [], ["persistierendes Vorhofflimmern"], []),
    ("kard_n04", "normal", "Herr Weber wurde bei NSTEMI zur Koronarangiographie aufgenommen.",
     ["NSTEMI", "Koronarangiographie"], [], [], [], ["STEMI"], []),
    ("kard_n05", "normal", "Anschließend erfolgte eine Stentimplantation.",
     ["Stentimplantation"], [], [], [], [], []),
    ("kard_n06", "normal", "Im EKG zeigte sich ein AV-Block ersten Grades.",
     ["EKG", "AV-Block ersten Grades"], [], [], [], ["AV-Block zweiten Grades"], []),
    ("kard_n07", "normal", "Bekannt ist eine chronische Herzinsuffizienz.",
     ["chronische Herzinsuffizienz"], [], [], [], [], []),
    ("kard_n08", "normal", "Nach elektrischer Kardioversion bestand wieder Sinusrhythmus.",
     ["Kardioversion"], [], [], [], [], []),
    ("kard_n09", "normal", "Bekannt ist eine arterielle Hypertonie.",
     ["arterielle Hypertonie"], [], [], [], ["arterielle Hypotonie"], []),
    ("kard_n10", "normal", "Wir empfehlen die elektive Katheterablation.",
     ["Katheterablation"], [], [], [], [], []),
    ("kard_n11", "normal", "Es besteht eine Herzinsuffizienz mit erhaltener Ejektionsfraktion.",
     ["Herzinsuffizienz mit erhaltener Ejektionsfraktion"], [], [], [],
     ["Herzinsuffizienz mit reduzierter Ejektionsfraktion"], []),
    ("kard_n12", "normal", "Frau Schneider erhielt eine CT-Koronarangiographie.",
     ["CT-Koronarangiographie"], [], [], [], [], []),
    ("kard_n13", "normal", "Die Echokardiographie zeigte einen Perikarderguss.",
     ["Echokardiographie", "Perikarderguss"], [], [], [], [], []),
    ("kard_n14", "normal", "Es liegt ein persistierendes Vorhofflimmern vor.",
     ["persistierendes Vorhofflimmern"], [], [], [], ["paroxysmales Vorhofflimmern"], []),

    # --- ASR-schwierige Begriffe ---------------------------------------------
    ("kard_h01", "hard", "Es erfolgte eine transkatheter Aortenklappenimplantation.",
     ["transkatheter Aortenklappenimplantation"], [], [], [], [], []),
    ("kard_h02", "hard", "Im Langzeit-EKG zeigte sich eine AV-Knoten-Reentrytachykardie.",
     ["Langzeit-EKG", "AV-Knoten-Reentrytachykardie"], [], [], [], ["AV-Reentrytachykardie"], []),
    ("kard_h03", "hard", "Es besteht der Verdacht auf ein Tachykardie-Bradykardie-Syndrom.",
     ["Tachykardie-Bradykardie-Syndrom"], [], [], [], [], []),
    ("kard_h04", "hard", "Echokardiographisch zeigte sich eine hypertroph-obstruktive Kardiomyopathie.",
     ["hypertroph-obstruktive Kardiomyopathie"], [], [], [], [], []),
    ("kard_h05", "hard", "Bekannt ist ein Wolff-Parkinson-White-Syndrom.",
     ["Wolff-Parkinson-White-Syndrom"], [], [], [], [], []),
    ("kard_h06", "hard", "Im Monitoring trat eine Torsade-de-pointes-Tachykardie auf.",
     ["Torsade-de-pointes-Tachykardie"], [], [], [], [], []),
    ("kard_h07", "hard", "Geplant ist eine transkatheter Edge-to-Edge-Reparatur.",
     ["transkatheter Edge-to-Edge-Reparatur"], [], [], [], [], []),
    ("kard_h08", "hard", "Der Sinus transversus pericardii war unauffällig.",
     ["Sinus transversus pericardii"], [], [], [], [], []),
    ("kard_h09", "hard", "Es fand sich eine arrhythmogene rechtsventrikuläre Kardiomyopathie.",
     ["arrhythmogene rechtsventrikuläre Kardiomyopathie"], [], [], [], [], []),
    ("kard_h10", "hard", "Der CHA2DS2-VASc-Score beträgt vier Punkte.",
     ["CHA2DS2-VASc-Score"], [], [], [], [], []),

    # --- Medikamente, Dosen und Zahlen ---------------------------------------
    ("kard_m01", "medication", "Wir beginnen mit Apixaban 5 mg zweimal täglich.",
     [], ["Apixaban"], [], ["5 mg"], ["Rivaroxaban"], []),
    ("kard_m02", "medication", "Bisoprolol 2,5 mg morgens wird fortgeführt.",
     [], ["Bisoprolol"], [], ["2,5 mg"], ["Metoprolol"], []),
    ("kard_m03", "medication", "Umstellung auf Sacubitril/Valsartan 24/26 mg zweimal täglich.",
     [], ["Sacubitril/Valsartan"], [], ["24/26 mg"], [], []),
    ("kard_m04", "medication", "Torasemid 10 mg morgens und Ramipril 5 mg abends.",
     [], ["Torasemid", "Ramipril"], [], ["10 mg", "5 mg"], ["Furosemid"], []),
    ("kard_m05", "medication", "Atorvastatin 40 mg zur Nacht.",
     [], ["Atorvastatin"], [], ["40 mg"], ["Rosuvastatin"], []),
    ("kard_m06", "medication", "Rivaroxaban wurde auf Edoxaban 60 mg umgestellt.",
     [], ["Rivaroxaban", "Edoxaban"], [], ["60 mg"], ["Apixaban"], []),
    ("kard_m07", "medication", "Candesartan 8 mg wird pausiert.",
     [], ["Candesartan"], [], ["8 mg"], [], []),

    # --- Verneinungen ---------------------------------------------------------
    ("kard_g01", "negation", "Kein Perikarderguss.",
     ["Perikarderguss"], [], ["Kein Perikarderguss"], [], [], []),
    ("kard_g02", "negation", "Es fand sich keine Aortenklappenstenose.",
     ["Aortenklappenstenose"], [], ["keine Aortenklappenstenose"], [], [], []),
    ("kard_g03", "negation", "Blutdruck 135/85 mmHg, keine orthostatische Hypotonie.",
     ["orthostatische Hypotonie"], [], ["keine orthostatische Hypotonie"], ["135/85 mmHg"], [], []),
    ("kard_g04", "negation", "Ohne Hinweis auf eine Mitralklappeninsuffizienz.",
     ["Mitralklappeninsuffizienz"], [], ["Ohne Hinweis"], [], [], []),
    ("kard_g05", "negation", "Kein Anhalt für ein Wolff-Parkinson-White-Syndrom.",
     ["Wolff-Parkinson-White-Syndrom"], [], ["Kein Anhalt"], [], [], []),

    # --- Confounder: aktiver Kontext enthält Begriffe, die NICHT gesagt werden
    ("kard_c01", "confounder", "Die linksventrikuläre Funktion war erhalten.",
     [], [], [], [], [], ["Tachykardie-Bradykardie-Syndrom", "Sacubitril/Valsartan"]),
    ("kard_c02", "confounder", "Der Patient ist beschwerdefrei und kardial kompensiert.",
     [], [], [], [], [], ["NSTEMI", "Herzinsuffizienz mit reduzierter Ejektionsfraktion"]),
    ("kard_c03", "confounder", "Im EKG Sinusrhythmus ohne Erregungsrückbildungsstörungen.",
     [], [], [], [], [], ["AV-Block dritten Grades", "Torsade-de-pointes-Tachykardie"]),
    ("kard_c04", "confounder", "Bekannt ist eine arterielle Hypotonie.",
     ["arterielle Hypotonie"], [], [], [], ["arterielle Hypertonie"], []),
    ("kard_c05", "confounder", "Die Belastbarkeit ist unverändert gut.",
     [], [], [], [], [], ["paroxysmales Vorhofflimmern", "Apixaban"]),
]

STRIP = ".,;:!?\"„“()"


def tokens(text):
    return [w.strip(STRIP).lower() for w in text.split() if w.strip(STRIP)]


def contains(text, phrase):
    hay, needle = tokens(text), tokens(phrase)
    return bool(needle) and any(
        hay[i:i + len(needle)] == needle for i in range(len(hay) - len(needle) + 1))


def load_dictionary_terms():
    terms = set()
    for name in os.listdir(DICT_DIR):
        with open(os.path.join(DICT_DIR, name), encoding="utf-8") as f:
            terms.update(l.strip() for l in f if l.strip() and not l.startswith("#"))
    return terms


def validate(cases, terms):
    problems, ids = [], set()
    for (cid, klass, ref, targets, meds, negs, nums, confusable, negative) in cases:
        if cid in ids:
            problems.append(f"{cid}: duplicate id")
        ids.add(cid)
        for t in targets + meds + confusable + negative:
            if t not in terms:
                problems.append(f"{cid}: '{t}' is not in the bundled dictionaries")
        for t in targets + meds + negs + nums:
            if not contains(ref, t):
                problems.append(f"{cid}: spoken annotation '{t}' is missing from its reference")
        for t in confusable + negative:
            if contains(ref, t):
                problems.append(f"{cid}: unspoken term '{t}' appears in its reference")
    return problems


def record(case, audio):
    cid, klass, ref, targets, meds, negs, nums, confusable, negative = case
    return {
        "id": cid, "audio": audio, "reference": ref, "specialty": "internal_cardiology",
        "difficulty": klass, "target_terms": targets, "medications": meds,
        "negations": negs, "numbers": nums, "confusable_terms": confusable,
        "negative_terms": negative,
    }


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_guide(path):
    lines = [
        "# Aufnahmepaket Kardiologie v1",
        "",
        f"{len(CASES)} kurze Diktatsätze für den Medical-ASR-Benchmark. Alle Personen- und",
        "Patientendaten sind **erfunden**. Bitte keine echten Patientendaten ergänzen.",
        "",
        "## So nehmen Sie auf",
        "",
        "1. **Sprechen Sie wie beim echten Diktat** — normales Tempo, normale",
        "   Betonung. Nicht überdeutlich artikulieren, keine Pausen zwischen Silben.",
        "2. Lesen Sie den Satz **genau so, wie er dasteht**. Zahlen und Einheiten",
        "   sprechen Sie so, wie Sie es im Diktat tun würden (z. B. „zwei Komma fünf",
        "   Milligramm“). Satzzeichen nicht mitsprechen.",
        "3. **Ein Satz pro Datei**, Dateiname = ID, z. B. `audio/kard_n01.wav`.",
        "4. **Format: WAV, 16 kHz, Mono.** Am einfachsten mit llmedi selbst: Diktat",
        "   aufnehmen, die WAV liegt dann unter",
        "   `~/Library/Application Support/com.llmedi.app/recordings/`.",
        "   Andere Aufnahmen umwandeln mit",
        "   `afconvert -f WAVE -d LEI16@16000 -c 1 eingabe.m4a audio/kard_n01.wav`.",
        "5. Ruhige Umgebung, aber das eigene Diktiermikrofon — kein Studio.",
        "6. Versprecher: Datei neu aufnehmen, nicht korrigierend weitersprechen.",
        "",
        "Ideal sind **mehrere Sprecherinnen und Sprecher**. Legen Sie dafür je Person",
        "einen Unterordner an (`audio_sprecherA/`) und kopieren Sie `diktate.jsonl`",
        "mit angepassten Pfaden — so bleiben die Ergebnisse je Stimme vergleichbar.",
        "",
        "## Warum diese Sätze",
        "",
        "| Klasse | Anzahl | prüft |",
        "|---|---|---|",
    ]
    counts = {}
    for c in CASES:
        counts[c[1]] = counts.get(c[1], 0) + 1
    purpose = {
        "normal": "typische Arztbriefsprache",
        "hard": "lange, seltene, schwer erkennbare Fachbegriffe",
        "medication": "Wirkstoffnamen, Dosen und Zahlen; ähnlich klingende Wirkstoffe",
        "negation": "Verneinungen — ein Verlust kehrt den Befund um",
        "confounder": "Kontextbegriffe, die NICHT gesagt werden — erkennt Halluzinationen",
    }
    for k in ["normal", "hard", "medication", "negation", "confounder"]:
        lines.append(f"| {k} | {counts.get(k, 0)} | {purpose[k]} |")
    lines += ["", "## Die Sätze", ""]
    for c in CASES:
        lines.append(f"- `{c[0]}` — {c[2]}")
    lines += [
        "",
        "## Danach",
        "",
        "```bash",
        "cd src-tauri",
        "cargo run --release --bin medical-asr-benchmark -- \\",
        "  --model <pfad-zum-Qwen3-ASR.gguf> \\",
        "  --dataset ../benchmarks/medical_asr/kardiologie_v1/diktate.jsonl \\",
        "  --config  ../benchmarks/medical_asr/config.json \\",
        "  --out-dir ../benchmarks/medical_asr/results",
        "```",
        "",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def synthesize(voice="Anna"):
    out_dir = os.path.join(HERE, "synthetic")
    audio_dir = os.path.join(out_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)
    rows = []
    for case in CASES:
        aiff = os.path.join(audio_dir, case[0] + ".aiff")
        wav = os.path.join(audio_dir, case[0] + ".wav")
        subprocess.run(["say", "-v", voice, "-o", aiff, case[2]], check=True)
        subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", aiff, wav],
                       check=True)
        os.remove(aiff)
        rows.append(record(case, "audio/" + case[0] + ".wav"))
    write_jsonl(os.path.join(out_dir, "dataset.jsonl"), rows)
    with open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(
            "# Synthetische Entwicklungsaudios — NICHT für Entscheidungen\n\n"
            f"Erzeugt lokal mit macOS `say -v {voice}` aus denselben Texten wie\n"
            "`../diktate.jsonl`. Nichts davon hat das Gerät verlassen.\n\n"
            "Synthetische Sprache ist sauberer und gleichförmiger als echtes Diktat\n"
            "und liest Abkürzungen, Zahlen und Fremdwörter anders vor. Diese Daten\n"
            "prüfen nur, ob die Pipeline Ende zu Ende läuft und wie sich Modi und\n"
            "Budgets *grob* verhalten. Ein Standardbudget oder eine Fuzzy-Strategie\n"
            "wird damit **nicht** gewählt — dafür braucht es menschliche Sprecher.\n")
    return len(rows)


def main():
    terms = load_dictionary_terms()
    problems = validate(CASES, terms)
    if problems:
        print("Package invalid:", *problems, sep="\n  ")
        sys.exit(1)
    os.makedirs(os.path.join(HERE, "audio"), exist_ok=True)
    write_jsonl(os.path.join(HERE, "diktate.jsonl"),
                [record(c, "audio/" + c[0] + ".wav") for c in CASES])
    write_guide(os.path.join(HERE, "AUFNAHME.md"))
    print(f"{len(CASES)} cases valid against {len(terms)} dictionary terms; "
          "diktate.jsonl and AUFNAHME.md written.")
    if "--synthetic" in sys.argv:
        n = synthesize()
        print(f"{n} synthetic clips written to synthetic/ (development only).")


if __name__ == "__main__":
    main()
