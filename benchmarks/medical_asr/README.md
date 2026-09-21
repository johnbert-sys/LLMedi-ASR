# Medical ASR Benchmark

## Zweck

Misst, wie stark **Context Biasing** und die **Fuzzy-Nachkorrektur** die
Erkennung medizinischer Fachbegriffe verbessern — und was sie kosten.

Die zentralen offenen Fragen:

- Wirkt Context Biasing bei unseren Fachbegriffen überhaupt?
- Welches Context-Budget ist optimal (60? 120? 240?)
- Ist es besser, Context-Begriffe **zusätzlich** in der Fuzzy-Korrektur zu
  behalten (Modus D) oder sie dort zu entfernen (Modus C, heutiger Zustand)?
- Fügt zu viel Context Begriffe ein, die nie gesprochen wurden?

Der Harness ist **eine Hülle um die echte Pipeline**, keine zweite
ASR-Implementierung. Modulauswahl, Round-Robin, Stufen, Deduplizierung,
Context-Rendering und Fuzzy-Korrektur stammen vollständig aus dem
Produktionscode der App.

## Dataset erstellen

Audio kommt nach `audio/`, Metadaten als eine JSON-Zeile pro Fall in
`dataset.jsonl`:

```json
{
  "id": "cardio_001",
  "audio": "audio/cardio_001.wav",
  "reference": "Bei Aufnahme zeigte sich eine hochgradige Aortenklappenstenose.",
  "specialty": "internal_cardiology",
  "target_terms": ["Aortenklappenstenose"],
  "negative_terms": [],
  "difficulty": "normal"
}
```

| Feld               | Bedeutung                                                                                                   |
| ------------------ | ----------------------------------------------------------------------------------------------------------- |
| `id`               | eindeutig, taucht in allen Ergebnissen auf                                                                  |
| `audio`            | Pfad **relativ zur `dataset.jsonl`**                                                                        |
| `reference`        | was tatsächlich gesprochen wurde                                                                            |
| `specialty`        | Fachgebiet, für die Gruppierung                                                                             |
| `target_terms`     | gesprochene Fachbegriffe, die im Transkript stehen müssen                                                   |
| `negative_terms`   | **nicht** gesprochene Begriffe, die im Context stehen könnten                                               |
| `difficulty`       | `normal`, `hard`, `medication`, `negation` oder `confounder`                                                |
| `medications`      | gesprochene Wirkstoffnamen (eigene Fehlergruppe)                                                            |
| `negations`        | gesprochene Verneinungen, die wörtlich erhalten bleiben müssen („kein Perikarderguss")                      |
| `numbers`          | gesprochene Zahlen/Dosen, wörtlich zu erhalten („2,5 mg", „135/85 mmHg")                                    |
| `confusable_terms` | ähnlich klingende, **nicht** gesprochene Begriffe („Mitralklappeninsuffizienz" bei gesprochener „…stenose") |

Die letzten vier Felder sind optional. Ein fertiges Aufnahmepaket für die
Kardiologie liegt in `kardiologie_v1/` (Anleitung: `AUFNAHME.md`).

**Audioformat: 16 kHz, Mono, WAV.** Genau das, was die App selbst aufnimmt — du
kannst eigene Diktate direkt aus
`~/Library/Application Support/com.llmedi.app/recordings/` übernehmen. Anderes
Material vorher konvertieren:

```bash
afconvert -f WAVE -d LEI16@16000 -c 1 eingabe.m4a audio/cardio_001.wav
```

(`afconvert` gehört zu macOS; `ffmpeg -i eingabe.wav -ar 16000 -ac 1 …` geht
ebenso, falls installiert.)

### Die drei Testklassen

**A — Normale Arztbriefsprache.** Realistische Sätze mit typischem Vokabular
(Aortenklappenstenose, Vorhofflimmern, Echokardiographie).

**B — ASR-schwierige Begriffe.** Lang, selten oder phonetisch heikel
(AV-Knoten-Reentrytachykardie, transkatheter Aortenklappenimplantation).
`"difficulty": "hard"`.

**C — Confounder.** Der aktive Context enthält schwierige Begriffe, die im Audio
**nicht** vorkommen. Diese in `negative_terms` eintragen. Diese Klasse ist die
wichtigste Absicherung: Sie zeigt, ob das Modell Begriffe aus dem Context
halluziniert. `"difficulty": "confounder"`.

## Wörterbücher konfigurieren

`config.json` legt fest, womit gemessen wird — dieselben Felder wie in den
App-Einstellungen:

```json
{
  "active_dictionaries": ["core_medical", "internal_cardiology"],
  "dictionary_levels": { "internal_cardiology": 250 },
  "context_dictionaries": ["internal_cardiology"],
  "custom_words": []
}
```

`active_dictionaries` sind die Wörterbücher der Nachkorrektur — dort zählt immer
die **ganze** Wortliste, Stufen spielen keine Rolle. `context_dictionaries` sind
die, die dem Modell Kontext anbieten dürfen; dort begrenzt `dictionary_levels`
die Auswahl. Fehlt `context_dictionaries`, gilt dieselbe Liste wie bei
`active_dictionaries`; eine leere Liste misst die Nachkorrektur allein.

`word_correction_threshold` ist optional. Fehlt es, gilt **der Standardwert der
App** (0.18) — der Benchmark misst dann genau die Korrektur, die die App macht.

## Benchmark starten

```bash
cargo run --release --bin medical-asr-benchmark -- \
  --model ~/.cache/huggingface/hub/models--handy-computer--Qwen3-ASR-1.7B-gguf/snapshots/*/Qwen3-ASR-1.7B-Q5_K_M.gguf
```

Nützliche Optionen:

| Option                         | Standard                               |
| ------------------------------ | -------------------------------------- |
| `--dataset`                    | `benchmarks/medical_asr/dataset.jsonl` |
| `--config`                     | `benchmarks/medical_asr/config.json`   |
| `--modes`                      | `A,B,C,D`                              |
| `--context-budgets`            | `0,30,60,120,240,400`                  |
| `--out-dir`                    | `benchmarks/medical_asr/results`       |
| `--specialty` / `--difficulty` | ohne Filter                            |

`--rescore <ergebnis.json>` bewertet einen gespeicherten Lauf mit den aktuellen
Metriken neu, ohne erneut zu transkribieren (schreibt `…-rescored.*`).

Modellpfad und Architektur sind nicht fest verdrahtet: Architektur und
Context-Fähigkeit werden **aus dem Modell selbst** gelesen.

## Modi

| Modus | Recognition Context  | Fuzzy-Korrektur                             |
| ----- | -------------------- | ------------------------------------------- |
| **A** | —                    | —                                           |
| **B** | —                    | kompletter Pool                             |
| **C** | `context_vocabulary` | Pool **minus** Context                      |
| **D** | `context_vocabulary` | **kompletter** Pool — das Verhalten der App |

A und B ignorieren das Budget und laufen deshalb nur einmal statt einmal pro
Budget.

Wie in der App wird die Auswahl vor dem Senden **mit dem Tokenizer des Modells
gemessen und ins Kontextfenster eingepasst**. Begriffe, die nicht passen, werden
zurückgestellt — in Modus C gehen sie dann in die Nachkorrektur, weil das Modell
sie nicht gesehen hat. Der Report zeigt Tokenverbrauch und Zurückgestelltes je
Budget. **C gegen D** ist die eigentliche Architekturfrage; der Report enthält
dafür einen eigenen Vergleichsblock.

## Budgets

Das Budget begrenzt, wie viele Begriffe als Context an das Modell gehen. Es wird
unverändert an die Produktionsfunktion `context_vocabulary(settings, budget)`
weitergereicht — der Benchmark wählt **nicht** selbst aus.

`0` bedeutet: kein Context. Die Stufen der Module (100/250/500) bestimmen
weiterhin den Kandidatenpool; das Budget entscheidet, wie viele davon es in den
Prompt schaffen.

## Ergebnisse

Pro Lauf entstehen in `--out-dir` drei Dateien mit Modellnamen und Zeitstempel:

- **`.json`** — alle Rohdaten je Testfall, inklusive Raw- und Final-Transkript
  sowie dem tatsächlich gesendeten Context-String
- **`.csv`** — eine Zeile pro Testfall × Konfiguration, für eigene Auswertungen
- **`.txt`** — der lesbare Aggregatreport (erscheint auch auf der Konsole)

Raw und Final werden getrennt gespeichert, damit nachvollziehbar bleibt, **wo**
eine Verbesserung entstand: im Modell oder in der Nachkorrektur.

## Gemessene Größen

Je Testfall und Konfiguration: WER roh und korrigiert, Trefferquote der
Zielbegriffe, False-Bias-Einfügungen, **ungesprochene Wörterbuchbegriffe**
(jeder Begriff des aktiven Pools, der in der Ausgabe steht, aber nicht in der
Referenz — roh und final, mit Namen im JSON/CSV; Beugungsendungen bis drei
Buchstaben zählen als gesprochen), **Fehlergruppen** (Medikamente,
Verneinungen, Zahlen, ähnlich klingende Begriffe), **durch Fuzzy verursachte
Fehler** (ersetzte Begriffe, die die Referenz nicht enthält; Fälle, in denen die
Korrektur die WER verschlechtert), übersprungene mehrdeutige Stellen, gesendeter
Kontext samt Tokenverbrauch, Laufzeit, Echtzeitfaktor und der daraus abgeleitete
KV-Cache-Bedarf (nach der Allokationsregel der Bibliothek berechnet, **nicht
gemessen**).

**Fehlgeschlagene Läufe werden nicht übersprungen.** Scheitert ein Lauf mit
Kontext (z. B. „output truncated"), wiederholt der Benchmark ihn — genau wie
die App — ohne Kontext und gibt dann den ganzen Pool in die Nachkorrektur
(`fell_back_without_context`). Scheitert auch das, steht die Zeile mit leerem
Transkript und `run_error` im Ergebnis (WER 100 %): In der App wäre dieses
Diktat verloren gewesen. Übersprungen wird nur unlesbares Audio.

## Interpretation

**Hohe Medical Term Accuracy allein reicht nicht.** Ein Modell, das aus einem
großen Context munter Fachbegriffe einstreut, erreicht hohe Trefferquoten und
produziert gleichzeitig Befunde, die nie diktiert wurden — im klinischen Kontext
der gefährlichere Fehler.

Lies deshalb immer beide Spalten zusammen:

- **MedTermAcc** steigt mit dem Budget → Biasing wirkt.
- **FalseBias** steigt mit → das Budget ist zu groß.

Das optimale Budget ist das größte, bei dem die Trefferquote noch steigt und die
False-Bias-Rate noch flach bleibt.

Zwei weitere Hilfen im Report:

- **`target terms in context` vs. `not in context`** — wirkt Biasing wirklich?
  Nur wenn die erste Zahl deutlich höher liegt, hat der Context etwas bewirkt
  und nicht einfach das Modell den Begriff ohnehin gekonnt.
- **ΔWER / ΔMedTermAcc / ΔFalseBias (D−C)** — negatives ΔWER und positives
  ΔMedTermAcc sprechen für D. Ein Gewinn, der mit zusätzlichen False-Bias-Fällen
  erkauft ist, zählt nicht.

## Determinismus

Die Auswahllogik (Round-Robin, Stufen, Deduplizierung) ist vollständig
deterministisch: gleiche Einstellungen ergeben exakt denselben Context-String.

Die Dekodierung von `transcribe.cpp` wird mit den Standardparametern
aufgerufen; deren Determinismus ist von der Bibliothek und dem Backend (Metal,
CPU) abhängig und wurde hier nicht verifiziert. Laufzeiten schwanken ohnehin.
Wiederhole knappe Vergleiche daher besser mehrfach.
