# llmedi — Pilotstatus der Wörterbuch-Pipeline

Stand: 2026-09-19. Dieses Dokument trennt strikt vier Stufen:

| Stufe                            | Bedeutung                                                 |
| -------------------------------- | --------------------------------------------------------- |
| **implementiert**                | Code existiert und ist im Build aktiv                     |
| **technisch getestet**           | automatisierte Tests / Messungen an Modell und Bibliothek |
| **mit echten Diktaten gemessen** | Benchmark mit menschlichen Sprechern ausgewertet          |
| **freigegeben**                  | für einen definierten Einsatzbereich abgenommen           |

**Aktuell ist nichts „mit echten Diktaten gemessen" und nichts „freigegeben".**
Erfolgreiche Builds und Unit-Tests belegen keine Erkennungsqualität.

## Verarbeitungskette (vom Wörterbuch zum Text)

```
aktivierte Wörterbücher + Stufen + eigene Wörter
  │ dictionaries::full_vocabulary()         → Kandidatenpool (dedupliziert)
  │ dictionaries::context_vocabulary(N)      → Auswahl: eigene Wörter zuerst,
  │                                            dann Module reihum nach Priorität
  │ context_window::fit_context()            → Tokenmessung mit dem Tokenizer des
  │                                            Modells; was nicht passt, wird
  │                                            zurückgestellt (nicht verworfen)
  │ transcription::apply_context_to_run_options()
  │   ├ Qwen3-ASR:  RunOptions::context   (Recognition Context, PR #144)
  │   └ Whisper:    initial_prompt          (Whisper-Run-Extension)
  ▼
Transkription (transcribe-cpp, Fork-Commit 986757bc)
  │ transcription::run_with_context_fallback() → scheitert der Lauf mit
  │                                            Kontext, einmal ohne Kontext
  │                                            wiederholen; dann geht der ganze
  │                                            Pool in die Nachkorrektur
  │ audio_toolkit::correct_with_vocabulary() → Fuzzy-Nachkorrektur gegen den
  │                                            ganzen aktiven Pool, auch gegen
  │                                            bereits gesendete Begriffe
  ▼
ausgegebener Text
```

Jeder Lauf protokolliert (Info-Level, ohne Diktattext): Poolgröße, ausgewählt,
gesendet, zurückgestellt, Tokenverbrauch, Tokenraum, Korrekturen. Die
Einstellungsseite „Wörterbuch" hat zwei Bereiche; der Kontextbereich zeigt, wie
viele Begriffe überhaupt als Kontext ausgewählt werden dürfen und was das letzte
Diktat davon gesendet hat.

## Stand je Baustein

| Baustein                                         | Stufe              | Beleg                                                                                              |
| ------------------------------------------------ | ------------------ | -------------------------------------------------------------------------------------------------- |
| Qwen Recognition Context                         | technisch getestet | nativer Smoke-Test gegen Qwen3-ASR-1.7B: `Feature::Context` gemeldet, Lauf mit Kontext erfolgreich |
| Fuzzy-Korrektur (Umlaute, Sicherheit)            | technisch getestet | 20 Gefahrenfälle: vorher 9 unsicher, jetzt 0; 21 deutsche Regressionstests                         |
| Token-Einpassung                                 | technisch getestet | echte Fenster gemessen (s. u.); 8 Tests                                                            |
| Rückfall ohne Kontext                            | technisch getestet | synthetisch: Diktat, das mit Kontext verloren ging („output truncated"), liefert jetzt Text        |
| Übersicht in der Oberfläche                      | implementiert      | zeigt auch, wenn das letzte Diktat ohne Kontext wiederholt wurde                                   |
| Getrennte Schalter für Kontext und Nachkorrektur | implementiert      | zwei Bereiche, Stufen nur beim Kontext, Alle-Schalter, Migration; 14 Tests                         |
| Kontextbudget 120 / Fuzzy-Strategie C vs. D      | **offen**          | nur synthetische Daten                                                                             |

## Mitgelieferte Wörterbücher (Stand 2026-09-20)

35 Module aus der kuratierten Sammlung `generated_v2`, zusammen 9 909 Einträge
(8 912 eindeutig; 884 Begriffe kommen in mehreren Modulen vor und kosten im
Kontext nur einen Platz):

| Gruppe          | Module                                                                                                                                                                                     | Begriffe |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------- |
| Grundwortschatz | core_medical                                                                                                                                                                               | 250      |
| Fachgebiete     | 12 (Kardiologie, Angiologie, Endokrinologie, Gastroenterologie, Hämatologie, Infektiologie, Nephrologie, Onkologie, Pneumologie, Rheumatologie, Neurologie, Radiologie, Viszeralchirurgie) | je 500   |
| Anatomie        | 9 Körperregionen                                                                                                                                                                           | je 250   |
| Medikamente     | 12 (Wirkstoffe, Wirkstoffklassen, Handelsnamen für vier Fächer)                                                                                                                            | 24–175   |

Stufen bleiben Präfixe einer Datei (`take(n)`), Handelsnamen bleiben opt-in,
nichts wird automatisch aktiviert. Gemessen mit allen Modulen auf der höchsten
Stufe: Nachkorrektur 0,33 s je 130-Wort-Diktat (Release, M5); die 41
Referenzdiktate blieben dabei unverändert (`tests/vocabulary_safety_probe.rs`).
Bei 35 aktiven Modulen und Budget 120 entfallen im Kontext nur noch etwa
3–4 Begriffe je Modul — ein Grund mehr, nur fachlich passende Module zu
aktivieren.

## Gemessene Grenzen (Tokenizer und Bibliothek, nicht geschätzt)

|                     | Qwen3-ASR-1.7B Q5_K_M                                | Whisper large-v3-turbo                |
| ------------------- | ---------------------------------------------------- | ------------------------------------- |
| Kontextgrenze       | geteiltes Fenster 65 536 Tokens                      | Prompt max. 223 Tokens, vorne gekappt |
| Ausgabe-Reserve     | 256 Tokens fest (längere Texte werden abgeschnitten) | —                                     |
| Audio               | 12,5 Tokens/s, max. ~87 min ohne Kontext             | 30-s-Fenster                          |
| 120 Kontextbegriffe | 947 Tokens                                           | 995 Tokens → nur 23 passen            |

Die frühere Annahme „~2 Tokens pro Begriff" war um Faktor 4 falsch.

## Behobene Fehler im Produktionspfad

1. **Fuzzy-Korrektur vertauschte Befunde** — `Mitralklappenstenose` →
   `Mitralklappeninsuffizienz`, „Herzinsuffizienz mit erhaltener EF" → „…mit
   reduzierter EF", `Tropen` → `Troponin` (Soundex-Bonus bzw. langer
   Normabstand). Jetzt: kein Soundex, wortweiser Abgleich, Mehrdeutigkeit → keine
   Ersetzung.
2. **Fuzzy-Korrektur löschte Verneinungen** — `kein Aortenklappenstenose` →
   `Aortenklappenstenose`. Jetzt: Verneinungen und Dosiseinheiten nach Zahlen
   werden nie angetastet oder verschluckt.
3. **124 von 1244 Begriffen (alle mit ä/ö/ü/ß) wurden nie korrigiert**, die
   längsten Begriffe waren durch 50-Zeichen- und 3-Wort-Grenzen unerreichbar.
4. **Whisper verlor Begriffe unbemerkt** — die Bibliothek kappte den Prompt
   vorne (persönliche Wörter zuerst weg), die App nahm die gekappten Begriffe
   trotzdem aus der Nachkorrektur. Jetzt: Einpassung vor dem Senden, Rest geht
   in die Nachkorrektur.
5. **Ein Diktat konnte durch den eigenen Kontext verloren gehen** — bei großem
   Kontext brach Qwen mit „output truncated" ab, die App lieferte keinen Text.
   Jetzt: automatischer Rückfall ohne Kontext (App und Benchmark identisch).
6. **Fuzzy-Fehler aus dem synthetischen Lauf** — `CT-Koronarangiographie`,
   `echokardiographisch`, `Punktion`→`Funktion`, `des`→`DES`, buchstabiertes
   `N S T E M I` — je mit Regressionstest behoben.
7. **Benchmark maß eine andere Korrektur als die App** — Schwelle 0.8 statt
   0.18. Jetzt an den App-Standard gebunden.

## Synthetischer Entwicklungslauf (keine Entscheidungsgrundlage)

41 Kardiologie-Sätze, eine synthetische Stimme (`say -v Anna`), Qwen3-ASR-1.7B
Q5_K_M, 574 Läufe, kein Lauf verloren. Ergebnis:
`results/synthetic/Qwen3-ASR-1.7B-Q5_K_M-1789834584-rescored.txt`.

| Konfiguration | WER   | Zielbegriffe | ungesprochene Begriffe (roh→final) |
| ------------- | ----- | ------------ | ---------------------------------- |
| A ohne alles  | 23,0% | 54,3%        | 0→0                                |
| B nur Fuzzy   | 20,7% | 62,9%        | 0→0                                |
| C, Budget 30  | 11,0% | 80,0%        | 2→2                                |
| C, Budget 120 | 13,5% | 80,0%        | 0→1                                |
| D, Budget 120 | 11,1% | 82,9%        | 0→1                                |
| D, Budget 400 | 8,9%  | 97,1%        | 1→1                                |

Was sich daraus **ablesen** lässt:

- Die Nachkorrektur fügte in keinem Lauf einen Begriff ein, der nicht
  gesprochen wurde (FuzzyErr 0; vor den Korrekturen 1–4 je Konfiguration).
- **Kontext erzeugt Halluzinationen des Modells**, schon im Rohtext: bei
  Budget 30/60 wurde gesprochenes „Atorvastatin" bzw. „Candesartan" zu
  „Sacubitril/Valsartan"; „Sinusrhythmus" wurde zu „Sinus Tachykardie"
  (60/120) bzw. „Sinus transversus pericardii" (240/400). Ohne Kontext trat das
  nie auf. Das ist das wichtigste Risiko für den Einsatz und hängt nicht
  monoton vom Budget ab.
- Bei „Sinus Tachykardie" fügt die Nachkorrektur die Halluzination zum
  Fachbegriff „Sinustachykardie" zusammen — sie erfindet nichts, macht die
  Modellhalluzination aber glaubwürdiger.
- D lag synthetisch in allen Budgets ≥ 60 vor C, ohne zusätzliche
  ungesprochene Begriffe.

Was sich **nicht** ablesen lässt: ob das bei menschlichen Sprechern, mit
Hintergrundgeräusch, Dialekt oder schnellem Diktat genauso ist. Eine Stimme,
ein Satz je Befund, kaum Zahlen.

## Zwei getrennte Schritte je Wörterbuch (seit 2026-09-20)

Die Wörterbuchseite hat zwei Bereiche, und jedes Wörterbuch wird für beide
Schritte getrennt geschaltet. Alle vier Kombinationen sind erlaubt.

| Schalter         | Nachkorrektur                                  | Modellkontext                    |
| ---------------- | ---------------------------------------------- | -------------------------------- |
| Nachkorrektur an | **komplette** Wortliste, ohne Stufenbegrenzung | —                                |
| Kontext an       | —                                              | Begriffe bis zur gewählten Stufe |
| beide an         | komplette Liste                                | Begriffe bis zur gewählten Stufe |
| beide aus        | nichts                                         | nichts                           |

**Umfangsstufen gelten nur noch für den Kontext.** Die Nachkorrektur kostet
keine Rechenzeit im Modell und vergleicht deshalb immer gegen die ganze Liste.

Der Schalter „Alle Wörterbücher für die Nachkorrektur" wirkt in beide
Richtungen: An schaltet alle ein (und nimmt künftig importierte automatisch
dazu), Aus nimmt alle wieder heraus. Sind einzelne abgewählt, zeigt er einen
Teilzustand samt „x von y aktiv"; ein Klick schaltet dann wieder alle ein. Er
wirkt **nur** auf die Nachkorrektur — der Kontext bleibt eine bewusste
Einzelentscheidung, weil er das Erkennungsergebnis beeinflusst.

Der Kontextbereich zeigt oben nur die Wörterbücher, die tatsächlich Kontext
liefern; alle übrigen liegen hinter „Weitere Wörterbücher hinzufügen". Dazu die
Aktion „Kontext leeren" samt Rückgängig. Der Kasten darüber nennt bewusst keine
Obergrenze, sondern den **Kandidatenpool** — was davon wirklich gesendet wird,
steht in der Zeile zum letzten Diktat, weil nur diese Zahl gemessen ist.

Migration (Schema 3): Was bisher aktiviert war, ist weiter in der Nachkorrektur;
für den Kontext übernimmt die Migration dieselbe Liste, abzüglich der
Wörterbücher, die kurzzeitig auf „nur Nachkorrektur" standen. Aktivierungen,
Stufen und eigene Wortänderungen bleiben erhalten.

**Änderung gegenüber Strategie C:** Die App nutzt Strategie D — gesendete
Kontextbegriffe bleiben für die Nachkorrektur verfügbar, weil ein Begriff im
Prompt nicht bedeutet, dass das Modell ihn geschrieben hat. Das stützt sich auf
synthetische Läufe und ist **nicht klinisch validiert**; C bleibt im Benchmark
als Vergleich.

Laufzeit: Mit allen 35 Wörterbüchern in der Nachkorrektur umfasst der Pool
8 912 eindeutige Begriffe; gemessen 0,33 s je 130-Wort-Diktat (Release, M5). Da
die Stufen hier nicht mehr begrenzen, ist das der Normalfall, sobald der
Alle-Schalter an ist.

## Offene Entscheidungen

- **Standardbudget** (derzeit 120 Begriffe, konservativ, keine Modellgrenze).
- **Fuzzy-Strategie**: C (Pool minus Kontext, heutiger Stand) oder D (ganzer Pool).
- **Medikamente im Kontext?** Die einzigen Wirkstoff-Verwechslungen kamen aus
  dem Kontext. Die neue Verwendungseinstellung erlaubt es jetzt, Medikamenten-
  Wörterbücher auf „Nur Nachkorrektur" zu stellen; ob das die bessere
  Voreinstellung ist, entscheiden erst menschliche Aufnahmen.
- **Voreinstellungen der beiden Schalter** (derzeit: Nachkorrektur nur für
  ausgewählte Wörterbücher, Alle-Schalter aus; Kontext wie bisher aktiviert).

Alle drei werden erst mit menschlichen Aufnahmen aus
`kardiologie_v1/diktate.jsonl` entschieden. Synthetische Ergebnisse dienen nur
der Entwicklung.

## Bekannte Grenzen

- Qwen schneidet Ausgaben über 256 Tokens ab (~150 Wörter je Diktat).
- Zahlen im Referenztext stehen als Ziffern; spricht das Modell sie aus
  („zwei Komma fünf"), zählt der Benchmark das bewusst als Fehler.
- Flexionsregeln sind heuristisch; ein fehlerhaft abgeschnittenes Wort, das
  zufällig wie eine Beugungsendung aussieht, wird nicht korrigiert (sicherer
  Fehler: bleibt stehen).
- Die sicherere Nachkorrektur lässt Schreibvarianten stehen, die die alte
  noch korrigierte: `hypertrof-obstruktive`, `Sinustransversusperikardie`
  (zusammengeschrieben, mehrere Abweichungen). Bewusst: lieber ein sichtbarer
  Tippfehler als ein falscher Befund.
- Der Fork `yuku-contrib/transcribe.cpp` ist ein ungemergter Upstream-PR.
- Das Wörterbuch-UI ist nur auf Deutsch und Englisch übersetzt; die übrigen
  22 Sprachen fallen auf Englisch zurück (`bun run check:translations` schlägt
  deshalb fehl).
