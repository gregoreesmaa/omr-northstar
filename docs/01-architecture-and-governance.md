# 01 — Architecture & Governance Specification

Existing OMR benchmarks fail as a North Star for four reasons: they are single-domain (clean renders or one archival corpus), they saturate on monophonic/simple homophonic music while hiding polyphonic and orchestral collapse, they substitute synthetic noise for real photography and real handwriting, and they score with proxy token-edit rates that hide catastrophic structural errors.

**OMR-NorthStar (`OMR-NS`)** is a unified, domain-complete, multi-year benchmark engineered to give crisp signal to early systems while remaining near-impossible to saturate at the frontier.

---

## 1. Canonical Corpus Size & Three Orthogonal Axes

### 1.1 Canonical Corpus Composition
Every `v1.x` release of `OMR-NS` comprises three disjoint page pools:
- **720 Core Music Pages** — Drawn from **~360 distinct musical works** (1–3 pages/work) and split at an exact 1:2 ratio per `(Tier, Domain)` cell into **240 Public Dev (`dev-v1`)** and **480 Sealed Private Test (`test-v1`)**.
- **80 Pages/Year Rolling Fresh Holdout (`fresh-YYYY`)** — Newly engraved, photographed, and commissioned blind pages added annually to detect training-set contamination.
- **40 Negative-Control Pages (`16 dev-v1 / 24 test-v1`)** — Out-of-distribution non-music pages (title pages, tables of contents, text libretti, blank manuscript staff paper, non-music tables/diagrams) plus framing/orientation/two-page-spread controls. Pages with `n_gt = 0` are excluded from the 720-page `Q(P)` average and enforced via the mandatory **Negative-Control Gate (`NC-Gate`)** (see `docs/04-metrics-and-human-utility.md` §2.4).
- **Optional Evaluation-Only Non-Commercial Research Track (`research_nc_eval_only`)** — BYOD manifest pointers to `CC-BY-NC` handwritten academic corpora (e.g., CVC-MUSCIMA / MUSCIMA++), strictly eval-only and never redistributed commercially, providing a guaranteed zero-leakage handwritten evaluation track for commercial/permissive models (see §5.3).

### 1.2 Three Orthogonal Classification Axes
To eliminate taxonomic collisions across specifications and metadata schemas, every core page is indexed by **three strictly orthogonal axes**:

1. **Musical Complexity Tier (`T0`–`T5`, 6 tiers):** Pure musical and engraving complexity, independent of image quality. The 15 fine-grained musicological repertoire strata (`T1.1`–`T3.7` in `docs/02-notational-taxonomy.md`) map cleanly onto `T0`–`T5`.
2. **Visual Domain (`CDR`, `ALS`, `RWP`, `HWM`, 4 domains):** Capture and production modality.
3. **Visual Severity (`V1`–`V4`, 4 levels):** Quantitative optical, geometric, photometric, and calligraphic difficulty — **`V1` Easy (`E`)**, **`V2` Medium (`M`)**, **`V3` Hard (`H`)**, and **`V4` Brutal (`B`)** (encoded in `page_metadata.schema.json` as `visual_tier ∈ {"E", "M", "H", "B"}`; see `docs/03-visual-and-manuscript-domains.md`).

---

## 2. Musical Complexity Ladder (`T0` → `T5`)

| Tier | Name | Corpus Page Share | `NS_tier` Weight | Dev / Test Pages | Repertoire Strata Mapping (`docs/02`) | Musical & Engraving Specification | Diagnostic Role |
|---|---|---:|---:|---:|---|---|---|
| **T0** | **Atomic Fundamentals** | 10% | 5% | 24 / 48 | `T1.1` (basic), `T1.4` (simple) | Single-staff monophony, 1 voice, treble/bass clefs, `<= 4` sharps/flats, simple/compound meters (`2/4`–`6/8`), whole→16th, basic dots/ties/accidentals, 4–6 clean systems/page. Zero chords, tuplets, grace notes, or lyrics. | Sanity floor. Early systems get immediate pitch/rhythm/clef/layout signal; frontier systems should exceed 97%+. |
| **T1** | **Standard Solo & Homophony** | 20% | 15% | 48 / 96 | `T1.1` (interm.), `T1.2`, `T1.3`, `T1.4` | 2-staff keyboard grand staff or solo + simple accompaniment; 1–2 voices/staff; chords `<= 4` notes; standard articulations, dynamics, hairpins, slurs, `8va`/`8vb`, repeats/voltas, triplets, simple grace notes, basic lyrics/chords/tempo text. | Core everyday repertoire (piano, hymnals, lead sheets, classical sheets). Measures practical product viability. |
| **T2** | **Polyphony, Vocal & Chamber** | 25% | 25% | 60 / 120 | `T2.1`, `T2.2`, `T2.3`, `T2.4` (5–8 staves) | True 3–4 voice counterpoint across 2–6 staves (fugues, Romantic/Impressionist piano with cross-staff beaming/voicing, SATB choral with multi-verse lyrics + melismas, string quartets, voice+piano, 3-staff organ/accordion). Alto/tenor clefs, mid-system clef/key/meter changes, nested tuplets, cue notes, chord symbols/figured bass, pedal marks. | Separates sequence memorizers from true structural parsers (voice splitting, cross-staff routing, lyric alignment). |
| **T3** | **Large Ensemble & Orchestral** | 20% | 25% | 48 / 96 | `T2.4` (9–12 staves), `T3.2` (16–24 staves), `T3.4` (Big Band) | 8–24+ staves/system (symphonic, wind band, Big Band, opera). French-score dynamic staff hiding (system size changes page-to-page), transposing instruments, multi-measure rests, `divisi`/`a2`/`unis.`, rehearsal marks, small rastral sizes, complex brace/bracket hierarchies, heavy vertical collisions. | Stress-tests high-res layout parsing, instrument identity tracking across changing systems, and dense vertical alignment. |
| **T4** | **Virtuosic & Contemporary** | 15% | 18% | 36 / 72 | `T3.1`, `T3.3`, `T3.4` (complex jazz/NNS), `T3.5`, `T3.6`, `T3.7` | 3–4 staff piano, harp pedaling, polymeters, irrational/nested tuplets (`11:8`, `5:4` in `3:2`), microtonal accidentals, feathered beaming, tone clusters, ossia staves, unmeasured cadenzas, jazz/pop slash + Nashville charts, guitar TAB+staff hybrids, unpitched percussion, Baroque figured bass & historical C-clefs. | Breaks rigid-grammar / closed-vocabulary systems; tests open-set symbol and rhythmic generalization. |
| **T5** | **Grandmaster & Adversarial** | 10% | 12% | 24 / 48 | Compound across `T3.1`–`T3.7` + Pillar 1–3 Traps | Extreme compound musical + physical/semantic traps: 28–32+ staff miniature orchestral scores; composer working autographs with **cross-outs, paste-overs, and marginalia**; **heavily annotated performer scores** (`HWM-Hybrid`: dense pencil fingerings, bowings, cuts, highlighter); **extreme capture** (open-book gutter warp, clips/tears, harsh shadows/glare); 18th/19th-c. hand-punched copperplate. | Guarantees multi-year headroom; requires human-level visual-musical reasoning and world knowledge. |

*Note on Page Share vs. `NS_tier` Weight:* `T0` holds 10% of corpus pages for statistical precision on the sanity floor, but carries only 5% weight in `NS_tier` so trivial pages cannot pad frontier scores, while `T3`–`T5` carry elevated scoring weight (`25% + 18% + 12% = 55%`).

---

## 3. The 4-Domain × 6-Tier Core Matrix

### 3.1 The 4 Visual Domains

1. **Clean Digital Renders (`CDR`)**: Born-digital exports across **`>= 6` distinct engraving engines and music fonts** (e.g., MuseScore/Leland, LilyPond/Emmentaler, Dorico/Bravura & Petaluma, Finale/Maestro, Sibelius/Opus, Verovio/Leipzig) spanning varied page geometries and DPIs (96–300 DPI). Never single-engraver.
2. **Archival & Library Scans (`ALS`)**: Real flatbed/overhead scans of historical and modern print editions (movable type, copperplate, lithography, offset, microfilm, pocket scores, varied publishers) with natural paper aging, foxing, bleed-through, skew, and binding shadows.
3. **Real-World Phone Photos (`RWP`)**: **100% real physical camera captures** (zero synthetic degradation). Captured across diverse smartphones, lighting (overhead, window sidelight, dim rehearsal room), and supports (piano rack, wire stand, desk, handheld).
4. **Handwritten Manuscripts (`HWM`)**: **100% real human handwriting** (zero synthetic wobble fonts), stratified into four sub-profiles:
   - *`HWM-a`*: Fair-copy archival manuscripts (legible historical copyists/composers).
   - *`HWM-b`*: Modern commissioned human-written pages (diverse musicians, pen/pencil on staff paper, scanned & photographed).
   - *`HWM-c`*: Hurried working autographs & sketches (concentrated in `T3`–`T5`).
   - *`HWM-Hybrid`*: Printed scores overlaid with heavy handwritten performer annotations (pencil fingerings, bowings, cuts, highlighter, taped patch measures; concentrated in `T2`–`T5`).

### 3.2 Allocation Matrix (`Dev / Test` pages per cell; zero empty cells)

| Tier | `CDR` (Render) | `ALS` (Scan) | `RWP` (Photo) | `HWM` (Handwritten) | Tier Total (`Dev / Test`) |
|---|---:|---:|---:|---:|---:|
| **T0** | 6 / 12 | 6 / 12 | 6 / 12 | 6 / 12 | **24 / 48** (10%) |
| **T1** | 12 / 24 | 12 / 24 | 12 / 24 | 12 / 24 | **48 / 96** (20%) |
| **T2** | 15 / 30 | 15 / 30 | 15 / 30 | 15 / 30 | **60 / 120** (25%) |
| **T3** | 12 / 24 | 14 / 28 | 12 / 24 | 10 / 20 | **48 / 96** (20%) |
| **T4** | 10 / 20 | 10 / 20 | 8 / 16 | 8 / 16 | **36 / 72** (15%) |
| **T5** | 2 / 4 | 6 / 12 | 8 / 16 | 8 / 16 | **24 / 48** (10%) |
| **Domain Total** | **57 / 114** (23.8%) | **63 / 126** (26.3%) | **61 / 122** (25.4%) | **59 / 118** (24.6%) | **240 / 480** (**720 Core Total**) |

*Note on `T5 × CDR`*: Represents extreme born-digital adversarial engraving (micro-dense contemporary scores, extreme collisions, polymetric/graphical complexity), while `T5` weight shifts deliberately toward `ALS`/`RWP`/`HWM` where real-world physical and semantic adversity peaks.

### 3.3 Built-In Paired-Capture Controlled Isolation Protocols (A–E)

Embedded inside the corpus are **5 paired-capture protocols** linking identical musical ground truth via `paired_anchor_id` (full optical specifications in `docs/03-visual-and-manuscript-domains.md` §6):

- **Protocol A — Digital-to-Physical Capture Ladder (`CDR → Print → Scan → Phone`):** 30 Dev / 60 Test `RWP` pages are real physical captures of exact `CDR`/`ALS` anchor pages across an 8-step ladder (`A0`–`A7`), isolating pure camera/lighting/geometry/codec drop (`Δ_photo`).
- **Protocol B — Multi-Engraver & Scale Invariance (`CDR`):** 20 Dev / 40 Test `CDR` pages rendered identically across `>= 3` (up to 5) engraver/font stacks, 3 staff-space scales, and 2 raster modes, isolating font/layout overfitting (`Δ_engraver`) and minimum legible staff-space cliff $s_{\text{cliff}}$.
- **Protocol C — Archival Multi-Digitization (`ALS` + `RWP`):** Same physical historical library volume captured via 600 DPI color cradle scan, legacy 1-bit microfilm, and reading-room handheld phone photo.
- **Protocol D — Handwritten & Hybrid Isolation (`HWM` + `RWP`):** Clean vs. pencil/highlighter-annotated physical pairs (`D1`), 15 Dev / 30 Test modern human-copied `CDR ↔ HWM-b` calligraphic pairs (`D2`, isolating `Δ_handwriting`), and manuscript scan vs. phone photo pairs (`D3`).
- **Protocol E — Factorized Single-Stressor Sweeps:** Controlled 5-step single-axis cliff probes (scale $s$, keystone $\theta$, blur $\sigma$, EV gradient, bleed-through $\alpha$, occlusion area) on a 10-page anchor set.

---

## 4. Input Contract: Single-Page, Multi-Page, Formats & EXIF/DPI Rules

Full container execution details are specified in [`docs/06-submission-and-harness.md`](docs/06-submission-and-harness.md). At the architectural level:

1. **Single-Page vs. Multi-Page Sequences:**
   - Standard benchmark items are delivered as **single-page units** (`page_id`).
   - Multi-page continuity items (Stress-Test Family `3C` and multi-page works of 2–4 consecutive pages) are delivered as **explicitly ordered page sequences** (`work_id` + `page_index_in_work = 1..k`) in an ordered input directory/manifest, so systems can propagate cross-page state (instrument identities, transpositions, ties, slurs, hairpins, `8va`/`Ped.`, French-score staff reappearance) without guessing page order.
   - Single-page-only systems may still run by emitting independent per-page outputs, which are evaluated normally on single-page metrics and scored strictly on cross-page continuity (`S_cont` and `3C`).
2. **Supported Input File Formats:**
   - **PNG** (lossless 1-bit, 8-bit grayscale, or 24-bit RGB), **JPEG** (camera/recompressed captures), **TIFF** (single-page or multi-page lossless/CCITT archival scans), and **PDF** (optional born-digital/archival wrapper; the sealed harness always provides canonical rasterized PNG/TIFF/JPEG inputs alongside any PDF wrapper so systems never depend on proprietary PDF libraries).
3. **DPI & EXIF Rules:**
   - **Zero Oracle Metadata:** Ground-truth DPI, staff-space height, domain tags (`CDR`/`ALS`/`RWP`/`HWM`), complexity tier, visual severity, staff counts, and instrumentation are **never** provided to the model at inference time.
   - **EXIF Orientation Normalization:** For all standard corpus pages, EXIF orientation tags are applied and stripped prior to delivery so pixels are stored upright. **Exception — Family `3B` Orientation Traps:** Intentional `90°`, `180°`, and `270°` rotations are baked directly into the pixel array with neutral/absent EXIF headers so systems must perform genuine visual orientation detection rather than reading camera metadata.

---

## 5. Anti-Contamination, Licensing & Governance Protocol

### 5.1 Three-Tier Split & Evaluation Security

- **`dev-v1` (240 core pages + 16 negative controls, Public):** Open images, canonical ground truth, and open-source evaluator. Tagged with explicit canary strings (`OMR_NS_V1_CANARY_DO_NOT_TRAIN`) and machine-readable exclusion manifests. Used for local iteration and error analysis; strictly prohibited in training/validation.
- **`test-v1` (480 core pages + 24 negative controls, Sealed Private):** Ground truth is **never** released publicly. Official scores require submitting a containerized model (Docker/OCI; see `docs/06-submission-and-harness.md`) run in a network-isolated evaluation sandbox (no internet, no raw image exfiltration, rate-limited submissions).
- **`fresh-YYYY` (80 pages/year, Rolling Blind Holdout):** Newly photographed physical scores, newly commissioned human manuscripts, and newly engraved unpublished pieces never previously on the web. Evaluated annually on top leaderboard models; a **`> 3.0`-point drop** from `test-v1` to `fresh-YYYY` triggers a public contamination flag.

### 5.2 Zero-Leakage Disjointness Rules

1. **Work-Level Disjointness:** If any page of a musical work appears in `test-v1`, **zero** pages of that work (in any edition, arrangement, scan, or manuscript) may appear in `dev-v1` or official training corpora.
2. **Writer/Copyist Disjointness (`HWM`):** Every modern writer and historical copyist in `test-v1` is disjoint from `dev-v1`.
3. **Device/Venue Disjointness (`RWP`):** `>= 50%` of smartphone models and capture environments in `test-v1` are unseen in `dev-v1`.
4. **Blind Scrubbing Index:** Releases ship a privacy-preserving index of image perceptual hashes (`pHash`/`dHash`) + symbolic pitch-interval/IOI 8-gram hashes so external teams can automatically decontaminate their training sets against `test-v1` without ever seeing `test-v1`.

### 5.3 Ground-Truth Provenance & Permissive Licensing

- **Zero-OMR-Bootstrapped GT:** No non-`CDR` ground truth may be produced by unverified OMR output. Every `ALS`/`RWP`/`HWM` page undergoes **double-blind human engraving** by two trained musicians, automated structural diffing, audio-playback audit, and third-party senior engraver adjudication.
- **Canonical Equivalence Layer:** Ground truth is stored in canonical MusicXML 4.0 + MEI paired with a deterministic canonicalizer that normalizes semantically equivalent representations (voice ID permutations, default stem directions, equivalent rest subdivisions) so valid engraving differences are not falsely penalized.
- **Clean Two-Track Licensing Governance:**
  1. **Permissive Core Track (`dev-v1`, `test-v1`, `fresh-YYYY`):** 100% permissive rights only — verified Public Domain (`Public-Domain`), `CC0-1.0`, or `CC-BY-4.0` for both music and images (PD archival scans, in-house CC-BY-4.0 photos, rights-transferred manuscripts). **Zero `CC-BY-NC` or scraped proprietary publisher editions** in the core track.
  2. **Evaluation-Only Non-Commercial Research Track (`research_nc_eval_only`):** An optional companion track referencing `CC-BY-NC` academic corpora (e.g., handwritten datasets such as CVC-MUSCIMA / MUSCIMA++) strictly via **Bring-Your-Own-Data (BYOD) manifest pointers** (checksums, canonical splits, and alignment metadata — never redistributed commercially and strictly prohibited from training). **Key governance advantage:** because commercial and permissive OMR models never train on `CC-BY-NC` corpora, these datasets serve as a **guaranteed zero-leakage, high-diversity handwritten evaluation suite** without removing scarce permissive handwritten pages from training sets. Results are reported separately on the `research_nc_eval_only` track and never mixed into the permissive `test-v1` headline `NS`.

### 5.4 Versioning & Anti-Saturation Governance

- Strict semantic versioning (`v1.0` frozen; GT errata only increment `v1.0.x` with full changelog and automatic historical re-scoring).
- **Mandatory Escalation Trigger:** Once any verified system exceeds **92.0 `NS`** on `test-v1` (or `> 95%` across `T0`–`T3`), `v2.0` is triggered, retiring saturated `T0` weight into expanded `T4`–`T5` slices.

---

## 6. Canonical Scoring Summary & 4-Layer Diagnostic Reporting Format

All scoring follows the single canonical `40 / 25 / 20 / 15` per-page quality formula and cross-domain `NS` composite defined in `docs/04-metrics-and-human-utility.md`:

$$Q_{\text{raw}}(P) = 100 \cdot \Big( 0.40\,S_{\text{note}}(P) + 0.25\,S_{\text{struct}}(P) + 0.20\,S_{\text{expr}}(P) + 0.15\,\max\big(0,\, \text{NTSR}(P)\big) \Big)$$

$$NS = \Big( 0.75\,M_{\text{geom}} + 0.25\,\min_{s \in \mathcal{D}} Q_s \Big) \cdot \Big(1 - 0.50\,\max_{s \in \mathcal{D}} \text{CFR}_s\Big)$$

accompanied by the official **6-Tier Weighted North Star (`NS_tier`)** (`T0: 5%, T1: 15%, T2: 25%, T3: 25%, T4: 18%, T5: 12%`) and the mandatory **Negative-Control Gate (`NC-Gate`)**.

Every benchmark run emits one canonical `report.json` (validated against `schemas/report.schema.json`) and a 4-layer executive/forensic dashboard:

1. **Layer 1 — Headline North Star Score (`NS`, 0–100) & Companion `NS_tier`:** Composite cross-domain `NS` with **95% work-clustered bootstrap CIs** (1,000 resamples), companion `NS_tier`, `NC-Gate` pass/fail status, **Median / P90 / P99 Weighted Fixes Per Page (`wFPP`)**, **Net Time Saved Ratio (`NTSR`)**, **Negative-Utility Page Rate (`NUPR`)**, and **Catastrophic Failure Rate (`CFR`)**.
2. **Layer 2 — 6×4 Matrix Heatmap & Paired Robustness Deltas:** Full per-`(Tier, Domain)` cell table of `Q` and `wFPP`, plus confounder-free deltas `Δ_engraver`, `Δ_photo`, and `Δ_handwriting`, and the 9-family Adversarial Scorecard.
3. **Layer 3 — Surgical 8-Bucket Error Taxonomy (% of Total Edit Cost):**
   1. Layout & System Segmentation
   2. Global Context / State Drift (raw + oracle-context-corrected)
   3. Pitch & Accidental Errors
   4. Rhythm & Meter Errors
   5. Polyphony & Voice/Staff Routing
   6. Engraving & Expressive Symbols
   7. Hallucination / Degeneration / Truncation
   8. Annotation Contamination (`T5` / `HWM-Hybrid`)
4. **Layer 4 — Actionable Forensic Pack & Regression Diff:** Worst-10 & Best-10 page roster stratified by domain, efficiency telemetry (median & P95 latency, peak memory, parameter count), and pairwise `compare(A, B)` work-clustered permutation significance (`p < 0.05`).
