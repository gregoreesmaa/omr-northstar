# OMR-NorthStar Pilot Fixture Pack (`pilot_dev_v0_1`)

Canary string: `OMR_NS_V1_CANARY_DO_NOT_TRAIN`  
License: `CC0-1.0` (Public Domain Dedication)

## Purpose

`pilot_dev_v0_1` is a self-contained, 100% CC0 runnable fixture pack for end-to-end smoke testing of the OMR-NorthStar evaluation pipeline (`omr_northstar.pilot` and `omr_northstar.scoring`). It contains **12 valid MusicXML 3.1 ground-truth scores** (`gt/ns-pilot-001.musicxml` – `gt/ns-pilot-012.musicxml`) paired with a schema-compliant `manifest.json` (`schemas/page_metadata.schema.json`).

## Coverage Matrix (4 Domains × 6 Tiers)

| Page ID | Work ID | Domain | Tier | Visual | Anchor ID | Protocol | Engraver / Writer | Musical Content |
|---|---|---|---|---|---|---|---|---|
| `ns-pilot-001` | `pilot-w01` | `CDR` | `T0` | `E` | `anchor-t0-melody` | `B` | LilyPond 2.24 | Monophonic C-major melody |
| `ns-pilot-002` | `pilot-w02` | `CDR` | `T1` | `E` | `anchor-t1-piano` | `A` | Dorico 5 | 2-staff grand-staff piano with chords, dynamics & slurs |
| `ns-pilot-003` | `pilot-w02` | `CDR` | `T1` | `E` | `anchor-t1-piano` | `A` | Verovio 4.3 | Same grand-staff anchor in 2nd engraver (Protocol A) |
| `ns-pilot-004` | `pilot-w03` | `ALS` | `T2` | `M` | `anchor-t2-counterpoint` | `B` | Breitkopf & Haertel (1898) | 2-voice counterpoint + lyrics + alto-clef viola |
| `ns-pilot-005` | `pilot-w04` | `ALS` | `T3` | `M` | — | — | Edition Peters (1904) | 4-staff chamber score with Bb Clarinet transposition & 2-bar multi-measure rest |
| `ns-pilot-006` | `pilot-w05` | `ALS` | `T0` | `E` | — | — | Baerenreiter (1952) | **Negative control**: blank staff (0 notes, stress family `2A`) |
| `ns-pilot-007` | `pilot-w01` | `RWP` | `T0` | `M` | `anchor-t0-melody` | `B` | LilyPond 2.24 | Paired phone-photo capture of `ns-pilot-001` |
| `ns-pilot-008` | `pilot-w03` | `RWP` | `T2` | `H` | `anchor-t2-counterpoint` | `B` | Breitkopf & Haertel (1898) | Paired phone-photo capture of `ns-pilot-004` |
| `ns-pilot-009` | `pilot-w06` | `RWP` | `T4` | `H` | — | — | Schott Music | Asymmetric `7/8` meter + triplet tuplet + mid-measure clef & key change |
| `ns-pilot-010` | `pilot-w02` | `HWM` | `T1` | `H` | `anchor-t1-piano` | `C` | `writer-pilot-01` | Paired handwritten manuscript of `anchor-t1-piano` |
| `ns-pilot-011` | `pilot-w07` | `HWM` | `T5` | `B` | — | — | `writer-pilot-02` | Ostinato trap (bar-4 `G#` mutation) + counterfactual accidentals (`1A`, `1B`) |
| `ns-pilot-012` | `pilot-w08` | `HWM` | `T4` | `B` | — | — | `writer-pilot-03` | Handwritten `7/8` meter + triplet tuplet + mid-measure clef & key change |

## Governance Notes

- **Paired Isolation Protocols**:
  - Protocol A (`anchor-t1-piano`, `CDR`↔`CDR`): Dorico 5 vs. Verovio 4.3 yields `delta_engraver`.
  - Protocol B (`anchor-t0-melody`, `anchor-t2-counterpoint`, `CDR`/`ALS`↔`RWP`): yields `delta_photo`.
  - Protocol C (`anchor-t1-piano`, `CDR`↔`HWM`): yields `delta_handwriting`.
- **Engraver & Writer Diversity (Pilot Scale)**: 7 distinct engravers/publishers and 3 distinct manuscript writers are represented across the 12 pages. Because Protocol B pairs an identical print across `CDR`/`ALS` and `RWP`, two engravers appear twice (`2/12 = 16.7%`, a pilot-pack scale artifact; full `dev-v1`/`test-v1` releases enforce `<= 15%` per engraver).
- **Negative Control**: `ns-pilot-006` is an explicit empty/blank-staff score (`0` notes) verifying that systems do not hallucinate notes on empty staves.

## Running Baselines

```python
from omr_northstar.pilot import run_oracle_baseline, run_degraded_baseline

oracle_report = run_oracle_baseline("pilot_dev_v0_1")
degraded_report = run_degraded_baseline("pilot_dev_v0_1", seed=42)
```
