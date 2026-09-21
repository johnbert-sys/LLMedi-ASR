# Aufnahmepaket Kardiologie v1

41 kurze Diktatsätze für den Medical-ASR-Benchmark. Alle Personen- und
Patientendaten sind **erfunden**. Bitte keine echten Patientendaten ergänzen.

## So nehmen Sie auf

1. **Sprechen Sie wie beim echten Diktat** — normales Tempo, normale
   Betonung. Nicht überdeutlich artikulieren, keine Pausen zwischen Silben.
2. Lesen Sie den Satz **genau so, wie er dasteht**. Zahlen und Einheiten
   sprechen Sie so, wie Sie es im Diktat tun würden (z. B. „zwei Komma fünf
   Milligramm“). Satzzeichen nicht mitsprechen.
3. **Ein Satz pro Datei**, Dateiname = ID, z. B. `audio/kard_n01.wav`.
4. **Format: WAV, 16 kHz, Mono.** Am einfachsten mit llmedi selbst: Diktat
   aufnehmen, die WAV liegt dann unter
   `~/Library/Application Support/com.llmedi.app/recordings/`.
   Andere Aufnahmen umwandeln mit
   `afconvert -f WAVE -d LEI16@16000 -c 1 eingabe.m4a audio/kard_n01.wav`.
5. Ruhige Umgebung, aber das eigene Diktiermikrofon — kein Studio.
6. Versprecher: Datei neu aufnehmen, nicht korrigierend weitersprechen.

Ideal sind **mehrere Sprecherinnen und Sprecher**. Legen Sie dafür je Person
einen Unterordner an (`audio_sprecherA/`) und kopieren Sie `diktate.jsonl`
mit angepassten Pfaden — so bleiben die Ergebnisse je Stimme vergleichbar.

## Warum diese Sätze

| Klasse     | Anzahl | prüft                                                              |
| ---------- | ------ | ------------------------------------------------------------------ |
| normal     | 14     | typische Arztbriefsprache                                          |
| hard       | 10     | lange, seltene, schwer erkennbare Fachbegriffe                     |
| medication | 7      | Wirkstoffnamen, Dosen und Zahlen; ähnlich klingende Wirkstoffe     |
| negation   | 5      | Verneinungen — ein Verlust kehrt den Befund um                     |
| confounder | 5      | Kontextbegriffe, die NICHT gesagt werden — erkennt Halluzinationen |

## Die Sätze

- `kard_n01` — Bei Aufnahme zeigte sich eine hochgradige Aortenklappenstenose.
- `kard_n02` — In der Echokardiographie fand sich eine Mitralklappeninsuffizienz.
- `kard_n03` — Es besteht ein paroxysmales Vorhofflimmern.
- `kard_n04` — Herr Weber wurde bei NSTEMI zur Koronarangiographie aufgenommen.
- `kard_n05` — Anschließend erfolgte eine Stentimplantation.
- `kard_n06` — Im EKG zeigte sich ein AV-Block ersten Grades.
- `kard_n07` — Bekannt ist eine chronische Herzinsuffizienz.
- `kard_n08` — Nach elektrischer Kardioversion bestand wieder Sinusrhythmus.
- `kard_n09` — Bekannt ist eine arterielle Hypertonie.
- `kard_n10` — Wir empfehlen die elektive Katheterablation.
- `kard_n11` — Es besteht eine Herzinsuffizienz mit erhaltener Ejektionsfraktion.
- `kard_n12` — Frau Schneider erhielt eine CT-Koronarangiographie.
- `kard_n13` — Die Echokardiographie zeigte einen Perikarderguss.
- `kard_n14` — Es liegt ein persistierendes Vorhofflimmern vor.
- `kard_h01` — Es erfolgte eine transkatheter Aortenklappenimplantation.
- `kard_h02` — Im Langzeit-EKG zeigte sich eine AV-Knoten-Reentrytachykardie.
- `kard_h03` — Es besteht der Verdacht auf ein Tachykardie-Bradykardie-Syndrom.
- `kard_h04` — Echokardiographisch zeigte sich eine hypertroph-obstruktive Kardiomyopathie.
- `kard_h05` — Bekannt ist ein Wolff-Parkinson-White-Syndrom.
- `kard_h06` — Im Monitoring trat eine Torsade-de-pointes-Tachykardie auf.
- `kard_h07` — Geplant ist eine transkatheter Edge-to-Edge-Reparatur.
- `kard_h08` — Der Sinus transversus pericardii war unauffällig.
- `kard_h09` — Es fand sich eine arrhythmogene rechtsventrikuläre Kardiomyopathie.
- `kard_h10` — Der CHA2DS2-VASc-Score beträgt vier Punkte.
- `kard_m01` — Wir beginnen mit Apixaban 5 mg zweimal täglich.
- `kard_m02` — Bisoprolol 2,5 mg morgens wird fortgeführt.
- `kard_m03` — Umstellung auf Sacubitril/Valsartan 24/26 mg zweimal täglich.
- `kard_m04` — Torasemid 10 mg morgens und Ramipril 5 mg abends.
- `kard_m05` — Atorvastatin 40 mg zur Nacht.
- `kard_m06` — Rivaroxaban wurde auf Edoxaban 60 mg umgestellt.
- `kard_m07` — Candesartan 8 mg wird pausiert.
- `kard_g01` — Kein Perikarderguss.
- `kard_g02` — Es fand sich keine Aortenklappenstenose.
- `kard_g03` — Blutdruck 135/85 mmHg, keine orthostatische Hypotonie.
- `kard_g04` — Ohne Hinweis auf eine Mitralklappeninsuffizienz.
- `kard_g05` — Kein Anhalt für ein Wolff-Parkinson-White-Syndrom.
- `kard_c01` — Die linksventrikuläre Funktion war erhalten.
- `kard_c02` — Der Patient ist beschwerdefrei und kardial kompensiert.
- `kard_c03` — Im EKG Sinusrhythmus ohne Erregungsrückbildungsstörungen.
- `kard_c04` — Bekannt ist eine arterielle Hypotonie.
- `kard_c05` — Die Belastbarkeit ist unverändert gut.

## Danach

```bash
cd src-tauri
cargo run --release --bin medical-asr-benchmark -- \
  --model <pfad-zum-Qwen3-ASR.gguf> \
  --dataset ../benchmarks/medical_asr/kardiologie_v1/diktate.jsonl \
  --config  ../benchmarks/medical_asr/config.json \
  --out-dir ../benchmarks/medical_asr/results
```
