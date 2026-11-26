# 01 — Architecture & Governance Specification

Existing OMR benchmarks fail as a North Star for four reasons: they are single-domain (clean renders or one archival corpus), they saturate on monophonic/simple homophonic music while hiding polyphonic and orchestral collapse, they substitute synthetic noise for real photography and real handwriting, and they score with proxy token-edit rates that hide catastrophic structural errors.

**OMR-NorthStar (`OMR-NS`)** is a unified, domain-complete, multi-year benchmark engineered to give crisp signal to early systems while remaining near-impossible to saturate at the frontier.

---

## 1. Tiered Difficulty Ladder (`T0` → `T5`)

Musical and engraving complexity is defined **orthogonally to visual domain**. The core suite comprises **720 pages** split at an exact 1:2 ratio per cell into **240 Public Dev (`dev-v1`)** and **480 Sealed Private Test (`test-v1`)**, plus an **80-page Annual Fresh Holdout (`fresh-YYYY`)**.

| Tier | Name | Share | Dev / Test Pages | Musical & Engraving Specification | Diagnostic Role |
|---|---|---:|---:|---|---|
| **T0** | **Atomic Fundamentals** | 10% | 24 / 48 | Single-staff monophony, 1 voice, treble/bass clefs, `<= 4` sharps/flats, simple/compound meters (`2/4`–`6/8`), whole→16th, basic dots/ties/accidentals, 4–6 clean systems/page. Zero chords, tuplets, grace notes, or lyrics. | Sanity floor. Early systems get immediate pitch/rhythm/clef/layout signal; frontier systems should exceed 97%+. |
| **T1** | **Standard Solo & Homophony** | 20% | 48 / 96 | 2-staff keyboard grand staff or solo + simple accompaniment; 1–2 voices/staff; chords `<= 4` notes; standard articulations, dynamics, hairpins, slurs, `8va`/`8vb`, repeats/voltas, triplets, simple grace notes, basic tempo/expression text. | Core everyday repertoire (piano, hymnals, pop/classical sheets). Measures practical product viability. |
| **T2** | **Polyphony, Vocal & Chamber** | 25% | 60 / 120 | True 3–4 voice counterpoint across 2–4 staves (fugues, Romantic/Impressionist piano with cross-staff beaming/voicing, SATB choral with multi-verse lyrics + melismas, string quartets, voice+piano). Alto/tenor clefs, mid-system clef/key/meter changes, nested tuplets, cue notes, chord symbols/figured bass, pedal marks. | Separates sequence memorizers from true structural parsers (voice splitting, cross-staff routing, lyric alignment). |
| **T3** | **Large Ensemble & Orchestral** | 20% | 48 / 96 | 8–24+ staves/system (symphonic, wind band, opera). French-score dynamic staff hiding (system size changes page-to-page), transposing instruments, multi-measure rests, `divisi`/`a2`/`unis.`, rehearsal marks, small rastral sizes, complex brace/bracket hierarchies, heavy vertical collisions. | Stress-tests high-res layout parsing, instrument identity tracking across changing systems, and dense vertical alignment. |
| **T4** | **Virtuosic & Contemporary** | 15% | 36 / 72 | 3-staff piano, polymeters, irrational/nested tuplets (`11:8`, `5:4` in `3:2`), microtonal accidentals, feathered beaming, tone clusters, ossia staves, unmeasured cadenzas, jazz/pop slash + rhythmic notation, guitar TAB+staff hybrids, historical C-clefs. | Breaks rigid-grammar / closed-vocabulary systems; tests open-set symbol and rhythmic generalization. |
| **T5** | **Grandmaster & Adversarial** | 10% | 24 / 48 | Extreme compound musical + physical/semantic traps: composer working autographs with **cross-outs, paste-overs, and marginalia**; **heavily annotated performer scores** (dense pencil fingerings, bowings, cuts, highlighter — scored on canonical-score extraction vs annotation separation); **extreme capture** (double-page open-book gutter warp, clips/tears occluding edges, harsh shadows/glare); 18th/19th-c. hand-punched copperplate & miniature pocket scores. | Guarantees multi-year headroom; requires human-level visual-musical reasoning and world knowledge. |

---

## 2. The 4-Domain × 6-Tier Matrix

### The 4 Visual Domains

1. **Clean Digital Renders (`CDR`)**: Born-digital exports across **`>= 6` distinct engraving engines and music fonts** (e.g., MuseScore/Leland, LilyPond/Emmentaler, Dorico/Bravura, Finale/Maestro, Sibelius/Opus, Verovio/Petaluma) spanning varied page geometries and DPIs (150–300 DPI). Never single-engraver.
2. **Archival & Library Scans (`ALS`)**: Real flatbed/overhead scans of historical and modern print editions (lithography, copperplate, pocket scores, varied publishers) with natural paper aging, foxing, bleed-through, skew, and binding shadows.
3. **Real-World Phone Photos (`RWP`)**: **100% real physical camera captures** (zero synthetic degradation). Captured across diverse smartphones, lighting (overhead, window sidelight, dim rehearsal room), and supports (piano rack, wire stand, desk, handheld).
4. **Handwritten Manuscripts (`HWM`)**: **100% real human handwriting** (zero synthetic wobble fonts), stratified into:
   - *HWM-a*: Fair-copy archival manuscripts (legible historical copyists/composers).
   - *HWM-b*: Modern commissioned human-written pages (diverse musicians, pen/pencil on staff paper, scanned & photographed).
   - *HWM-c*: Hurried working autographs & sketches (concentrated in `T3`–`T5`).

### Allocation Matrix (`Dev / Test` pages per cell; zero empty cells)

| Tier | `CDR` (Render) | `ALS` (Scan) | `RWP` (Photo) | `HWM` (Handwritten) | Tier Total (`Dev / Test`) |
|---|---:|---:|---:|---:|---:|
| **T0** | 6 / 12 | 6 / 12 | 6 / 12 | 6 / 12 | **24 / 48** (10%) |
| **T1** | 12 / 24 | 12 / 24 | 12 / 24 | 12 / 24 | **48 / 96** (20%) |
| **T2** | 15 / 30 | 15 / 30 | 15 / 30 | 15 / 30 | **60 / 120** (25%) |
| **T3** | 12 / 24 | 14 / 28 | 12 / 24 | 10 / 20 | **48 / 96** (20%) |
| **T4** | 10 / 20 | 10 / 20 | 8 / 16 | 8 / 16 | **36 / 72** (15%) |
| **T5** | 2 / 4 | 6 / 12 | 8 / 16 | 8 / 16 | **24 / 48** (10%) |
| **Domain Total** | **57 / 114** (23.8%) | **63 / 126** (26.3%) | **61 / 122** (25.4%) | **59 / 118** (24.6%) | **240 / 480** (**720 Total**) |

*Note on `T5 × CDR`*: Represents extreme born-digital adversarial engraving (micro-dense contemporary scores, extreme collisions, polymetric/graphical complexity), while `T5` weight shifts deliberately toward `ALS`/`RWP`/`HWM` where real-world physical and semantic adversity peaks.

### Built-In Paired-Control Slices (Confounder-Free Diagnostics)

Embedded inside the matrix are three exact paired subsets sharing identical musical ground truth (`paired_anchor_id`):

- **Engraver-Invariance Pairings (`CDR`)**: 20 Dev / 40 Test pages rendered identically across 3 different engravers → isolates font/layout overfitting (`Δ_engraver`).
- **Capture-Robustness Pairings (`CDR`/`ALS` ↔ `RWP`)**: 30 Dev / 60 Test `RWP` pages are real phone photos of the exact physical pages present in `CDR`/`ALS` → isolates pure camera/lighting/geometry degradation (`Δ_photo`).
- **Handwriting-Gap Pairings (`CDR` ↔ `HWM-b`)**: 15 Dev / 30 Test modern handwritten pages copied by humans from exact `CDR` scores → isolates pure handwriting legibility from musical complexity (`Δ_handwriting`).

---

## 3. Anti-Contamination, Licensing & Governance Protocol

### 3.1 Three-Tier Split & Evaluation Security

- **`dev-v1` (240 pages, Public)**: Open images, canonical ground truth, and open-source evaluator. Tagged with explicit canary strings (`OMR_NS_V1_CANARY_DO_NOT_TRAIN`) and machine-readable exclusion manifests. Used for local iteration and error analysis; strictly prohibited in training/validation.
- **`test-v1` (480 pages, Sealed Private)**: Ground truth is **never** released publicly. Official scores require submitting a containerized model (Docker/OCI) run in a network-isolated evaluation sandbox (no internet, no raw image exfiltration, rate-limited submissions).
- **`fresh-YYYY` (80 pages/year, Rolling Blind Holdout)**: Newly photographed physical scores, newly commissioned human manuscripts, and newly engraved unpublished pieces never previously on the web. Evaluated annually on top leaderboard models; a **`> 3.0`-point drop** from `test-v1` to `fresh-YYYY` triggers a public contamination flag.

### 3.2 Zero-Leakage Disjointness Rules

1. **Work-Level Disjointness**: If any page of a musical work appears in `test-v1`, **zero** pages of that work (in any edition, arrangement, scan, or manuscript) may appear in `dev-v1` or official training corpora.
2. **Writer/Copyist Disjointness (`HWM`)**: Every modern writer and historical copyist in `test-v1` is disjoint from `dev-v1`.
3. **Device/Venue Disjointness (`RWP`)**: `>= 50%` of smartphone models and capture environments in `test-v1` are unseen in `dev-v1`.
4. **Blind Scrubbing Index**: Releases ship a privacy-preserving index of image perceptual hashes (`pHash`/`dHash`) + symbolic pitch-interval/IOI 8-gram hashes so external teams can automatically decontaminate their training sets against `test-v1` without ever seeing `test-v1`.

### 3.3 Ground-Truth Provenance & Permissive Licensing

- **Zero-OMR-Bootstrapped GT**: No non-`CDR` ground truth may be produced by unverified OMR output. Every `ALS`/`RWP`/`HWM` page undergoes **double-blind human engraving** by two trained musicians, automated structural diffing, audio-playback audit, and third-party senior engraver adjudication.
- **Canonical Equivalence Layer**: Ground truth is stored in canonical MusicXML 4.0 + MEI paired with a deterministic canonicalizer that normalizes semantically equivalent representations (voice ID permutations, default stem directions, equivalent rest subdivisions) so valid engraving differences are not falsely penalized.
- **Clean Licensing Only**: Public-domain works (verified PD) or CC0/CC-BY-4.0 newly commissioned works; 100% permissive image rights (PD archival scans, in-house CC-BY-4.0 photos, rights-transferred manuscripts). **Zero `CC-BY-NC` or scraped proprietary publisher editions.**

### 3.4 Versioning & Anti-Saturation Governance

- Strict semantic versioning (`v1.0` frozen; GT errata only increment `v1.0.x` with full changelog and automatic historical re-scoring).
- **Mandatory Escalation Trigger**: Once any verified system exceeds **92.0 `NS`** on `test-v1` (or `> 95%` across `T0`–`T3`), `v2.0` is triggered, retiring saturated `T0` weight into expanded `T4`–`T5` slices.

---

## 4. 4-Layer Diagnostic Reporting Format

Every benchmark run emits one canonical `report.json` (validated against `schemas/report.schema.json`) and a 4-layer executive/forensic dashboard:

1. **Layer 1 — Headline North Star Score (`NS`, 0–100)**: Composite geometric-mean + worst-stratum-floor score penalized by `CFR_max` (exact math in `docs/04-metrics-and-human-utility.md`), reported with **95% work-clustered bootstrap CIs** (1,000 resamples), **Median Weighted Fixes Per Page (`wFPP`)**, **Net Time Saved Ratio (`NTSR`)**, **Negative-Utility Page Rate (`NUPR`)**, and **Catastrophic Failure Rate (`CFR`)**.
2. **Layer 2 — 6×4 Matrix Heatmap & Paired Robustness Deltas**: Full per-`(Tier, Domain)` cell table of `Q` and `wFPP`, plus confounder-free deltas `Δ_engraver`, `Δ_photo`, and `Δ_handwriting`.
3. **Layer 3 — Surgical 8-Bucket Error Taxonomy (% of Total Edit Cost)**:
   1. Layout & System Segmentation
   2. Global Context / State Drift (raw + oracle-context-corrected)
   3. Pitch & Accidental Errors
   4. Rhythm & Meter Errors
   5. Polyphony & Voice/Staff Routing
   6. Engraving & Expressive Symbols
   7. Hallucination / Degeneration / Truncation
   8. Annotation Contamination (`T5`)
4. **Layer 4 — Actionable Forensic Pack & Regression Diff**: Worst-10 & Best-10 page roster stratified by domain, efficiency telemetry (median & P95 latency, peak memory, parameter count), and pairwise `compare(A, B)` permutation significance (`p < 0.05`).
