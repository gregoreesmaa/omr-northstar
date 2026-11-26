# OMR-NorthStar (`OMR-NS`)

**A Unified, Model-Agnostic North Star Benchmark for Optical Music Recognition**

Existing OMR benchmarks fail as a North Star for four reasons: they are single-domain (clean renders or one archival corpus), they saturate on monophonic/simple homophonic music while hiding polyphonic and orchestral collapse, they substitute synthetic noise for real photography and real handwriting, and they score with proxy token-edit rates that hide catastrophic structural errors.

**OMR-NorthStar (`OMR-NS`)** is an independent, model-agnostic benchmark specification, JSON Schema suite, and deterministic reference evaluator designed to measure true end-to-end Optical Music Recognition across three strictly orthogonal axes.

---

## Canonical Corpus Size & Three Orthogonal Axes

### Canonical Corpus Size
- **720 Core Pages** — Split at an exact 1:2 ratio per cell into **240 Public Dev (`dev-v1`)** and **480 Sealed Private Test (`test-v1`)**, drawn from **~360 distinct musical works** (1–3 pages/work) with strict work-, writer-, and device-level disjointness.
- **80 Pages/Year Rolling Fresh Holdout (`fresh-YYYY`)** — Newly captured/engraved blind holdout to detect training-set contamination over time.
- **40 Negative-Control Pages (`16 dev-v1 / 24 test-v1`)** — Out-of-distribution non-music pages (title pages, text/libretti, blank manuscript staff paper, tables/graphics) plus framing/orientation/spread controls, scored via a dedicated **Negative-Control Gate (`NC-Gate`)** so `n_gt = 0` pages never divide by zero in `Q(P)`.
- **Optional Evaluation-Only Non-Commercial Research Track (`research_nc_eval_only`)** — Alongside the 100% permissive Core Track (`Public-Domain` / `CC0-1.0` / `CC-BY-4.0`), `OMR-NS` supports an evaluation-only BYOD manifest track pointing to non-commercial handwritten corpora (e.g., `CC-BY-NC` datasets such as CVC-MUSCIMA / MUSCIMA++). Because commercial and permissive OMR models never train on `CC-BY-NC` data, these works act as a **guaranteed zero-leakage, high-diversity handwritten evaluation benchmark** without sacrificing scarce permissive handwritten data from training sets (eval-only, never trained on, never redistributed commercially).

### Three Orthogonal Classification Axes (Never Colliding)
1. **Musical Complexity Tier (`T0`–`T5`, 6 tiers):** Pure notational and engraving complexity, mapped cleanly from 15 fine-grained musicological repertoire strata (`T1.1`–`T3.7`, see [`docs/02-notational-taxonomy.md`](docs/02-notational-taxonomy.md)).
2. **Visual Domain (`CDR`, `ALS`, `RWP`, `HWM`, 4 domains):**
   - **Clean Digital Renders (`CDR`)** — Born-digital exports across `>= 6` distinct engraving engines and music fonts (e.g., Dorico/Bravura, MuseScore/Leland, LilyPond/Emmentaler, Finale/Maestro, Sibelius/Opus, Verovio/Leipzig, plus jazz/handwritten-style digital fonts).
   - **Archival & Library Scans (`ALS`)** — Real flatbed/overhead scans of 17th–20th century engraved plates, movable type, lithography, offset, microfilm, and aged library prints with foxing, bleed-through, gutter shadows, and stamps.
   - **Real-World Phone Photos (`RWP`)** — 100% real physical smartphone captures across uncontrolled lighting, perspective tilt, open-book gutter curl, plastic-sleeve glare, folds, photocopies, and messaging-app recompression.
   - **Handwritten Manuscripts (`HWM`)** — 100% real human handwriting spanning archival fair copies (`HWM-a`), modern commissioned human manuscripts (`HWM-b`), hurried composer drafts with cross-outs/paste-overs (`HWM-c`), and heavily annotated printed scores (`HWM-Hybrid`).
3. **Visual Severity (`V1`–`V4`, 4 levels):** Quantitative physical/optical difficulty independent of musical complexity — **`V1` Easy (`E`)**, **`V2` Medium (`M`)**, **`V3` Hard (`H`)**, and **`V4` Brutal (`B`)** (schema enum codes `E`/`M`/`H`/`B`; see [`docs/03-visual-and-manuscript-domains.md`](docs/03-visual-and-manuscript-domains.md)).

---

## The 6-Tier (`T0`–`T5`) × 4-Domain Core Matrix (720 Pages)

| Tier | Name | Page Share | `NS_tier` Weight | `CDR` (Render) | `ALS` (Scan) | `RWP` (Photo) | `HWM` (Handwritten) | Tier Total (`Dev / Test`) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **T0** | **Atomic Fundamentals** | 10% | 5% | 6 / 12 | 6 / 12 | 6 / 12 | 6 / 12 | **24 / 48** |
| **T1** | **Standard Solo & Homophony** | 20% | 15% | 12 / 24 | 12 / 24 | 12 / 24 | 12 / 24 | **48 / 96** |
| **T2** | **Polyphony, Vocal & Chamber** | 25% | 25% | 15 / 30 | 15 / 30 | 15 / 30 | 15 / 30 | **60 / 120** |
| **T3** | **Large Ensemble & Orchestral** | 20% | 25% | 12 / 24 | 14 / 28 | 12 / 24 | 10 / 20 | **48 / 96** |
| **T4** | **Virtuosic & Contemporary** | 15% | 18% | 10 / 20 | 10 / 20 | 8 / 16 | 8 / 16 | **36 / 72** |
| **T5** | **Grandmaster & Adversarial** | 10% | 12% | 2 / 4 | 6 / 12 | 8 / 16 | 8 / 16 | **24 / 48** |
| **Total** | **Core Matrix** | **100%** | **100%** | **57 / 114** | **63 / 126** | **61 / 122** | **59 / 118** | **240 / 480 (720)** |

*(Plus **80 pages/year `fresh-YYYY`** and **40 negative-control pages** `16 dev-v1 / 24 test-v1` evaluated via `NC-Gate`.)*

---

## Composite North Star Score (`NS`) & Companion Tier-Weighted `NS_tier`

Every scored music page is evaluated on its **Canonical Evaluation Representation (CER)** via two-stage Needleman–Wunsch measure alignment and Hungarian voice-bijection matching across four canonical axes (`40 / 25 / 20 / 15`):

$$Q_{\text{raw}}(P) = 100 \cdot \Big( 0.40\,S_{\text{note}}(P) + 0.25\,S_{\text{struct}}(P) + 0.20\,S_{\text{expr}}(P) + 0.15\,\max\big(0,\, \text{NTSR}(P)\big) \Big)$$

$$Q(P) = Q_{\text{raw}}(P) \cdot \mathbb{I}[\text{parseable} \land \neg\text{empty}] \cdot (0.85)^{k_{\text{dropped\_sys}}} \cdot \text{LoopPen}(P)$$

where $\text{NTSR}(P) = 1 - \frac{\text{wFPP}(P)}{\text{TranscribeCost}(P)}$ is the **Net Time Saved Ratio** derived from root-cause-factored **Weighted Fixes Per Page (`wFPP`)** capped at rational musician re-entry cost.

### Headline Cross-Domain North Star Score (`NS`)
To prevent clean-only specialists from gaming arithmetic averages, the four domain stratum scores $Q_s$ ($s \in \{\text{CDR}, \text{ALS}, \text{RWP}, \text{HWM}\}$, work-level macro-averages) are combined via a **Geometric Mean + Worst-Stratum Floor** penalized by the **Worst-Stratum Catastrophic Failure Rate ($\text{CFR}_{\max}$)**:

$$M_{\text{geom}} = \left(\prod_{s \in \mathcal{D}} \max(Q_s,\, 0.1)\right)^{1/4}$$

$$NS = \Big( 0.75\,M_{\text{geom}} + 0.25\,\min_{s \in \mathcal{D}} Q_s \Big) \cdot \Big(1 - 0.50\,\max_{s \in \mathcal{D}} \text{CFR}_s\Big)$$

### Official Companion Tier-Weighted North Star Score (`NS_tier`)
To track musical-complexity headroom independently of visual domain, `OMR-NS` reports the companion **6-Tier Weighted North Star (`NS_tier`)** using complexity weights $w_t = \{T0: 5\%,\; T1: 15\%,\; T2: 25\%,\; T3: 25\%,\; T4: 18\%,\; T5: 12\%\}$:

$$M_{\text{tier,geom}} = \prod_{t \in \mathcal{T}} \max(Q_t,\, 0.1)^{w_t}, \qquad NS_{\text{tier}} = \Big( 0.75\,M_{\text{tier,geom}} + 0.25\,\min_{t \in \mathcal{T}} Q_t \Big) \cdot \Big(1 - 0.50\,\max_{t \in \mathcal{T}} \text{CFR}_t\Big)$$

### Negative-Control Gate (`NC-Gate`)
The 40 negative-control pages (`n_gt = 0` non-music/blank-staff pages and spread/orientation controls) are excluded from the 720-page `Q(P)` average (preventing `0/0` division in `NTSR`) and enforced as a mandatory leaderboard gate: verified submissions must achieve `0%` Non-Music Hallucination Rate (`NC_FP = 0`) and `100%` two-page spread ordering accuracy (violations trigger an `[NC-FAIL]` leaderboard flag; see [`docs/04-metrics-and-human-utility.md`](docs/04-metrics-and-human-utility.md)).

**Illustrative `NS` Calibration:**
- *Clean-only specialist* `(98, 92, 35, 10)`, `CFR_max = 0.60` $\implies M_{\text{geom}} = 42.15,\; NS = \mathbf{23.88}$
- *Printed+Photo strong, HW weak* `(95, 88, 80, 30)`, `CFR_max = 0.25` $\implies M_{\text{geom}} = 66.93,\; NS = \mathbf{50.48}$
- *Balanced universal system* `(90, 84, 76, 68)`, `CFR_max = 0.04` $\implies M_{\text{geom}} = 79.06,\; NS = \mathbf{74.77}$

---

## Public Baseline Results (v0.1, 125 pages)

The unpacked 125-page v0.1 pilot evaluation pack (125 PNG page images + 46 canonical MusicXML ground-truth files + `manifest.json`, 100% permissive `CC0-1.0` / `PDM-1.0` / `CC-BY-4.0`) is hosted publicly on Hugging Face:

- **Dataset:** [huggingface.co/datasets/gregoreesmaa/omr-northstar-v0.1](https://huggingface.co/datasets/gregoreesmaa/omr-northstar-v0.1) — **License:** `CC-BY-4.0` overall (100% permissive Core Track: 97 `CC0-1.0`, 20 `CC0-1.0`/`PDM-1.0`, 8 `CC-BY-4.0`; zero `CC-BY-NC`; full per-stratum provenance & attribution in the dataset `LICENSE` and `manifest.json`).

```bash
huggingface-cli download gregoreesmaa/omr-northstar-v0.1 --repo-type dataset --local-dir ./omr-northstar-v0.1
```

*(Note: In v0.1, `ALS` and `RWP` are pilot degradations derived from the `CDR` anchors pending full physical flatbed scanning and smartphone capture for v1.0; `HWM` contains real historical handwritten manuscripts.)*

All public baselines are evaluated **zero-shot with out-of-the-box defaults** in offline isolated execution on the 125-page v0.1 pilot pack (**31 CDR + 31 ALS + 31 RWP + 28 HWM + 4 Pillar 3B Negative Controls**: 2 blank staff pages + 2 text program-notes pages), scored with the canonicalized `omr-northstar` CER reference evaluator (`89c698c`).

| System | Version / License | NS Score (95% CI) | Tier-W NS | CDR Q (CFR) | ALS Q (CFR) | RWP Q (CFR) | HWM Q (CFR) | `S_note` | `S_expr` | `mean_NTSR` | `NMHR` |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Oracle GT** *(ceiling)* | Ground Truth | **99.53** `[99.32, 99.68]` | 99.69 | 99.70 (0.0%) | 99.66 (0.0%) | 99.66 (0.0%) | 99.34 (0.0%) | 1.0000 | 1.0000 | +1.000 | 0.0% |
| **homr** | `0.7.1` / AGPL-3.0 | **14.67** `[12.02, 17.78]` | 25.24 | 60.76 (31.4%) | 35.65 (67.7%) | 69.21 (19.4%) | 8.56 (92.9%) | 0.5037 | 0.2545 | +0.324 | 0.0% |
| **Audiveris** | `5.11.0` / AGPL-3.0 | **2.81** `[1.49, 3.80]` | 11.90 | 72.07 (22.9%) | 2.15 (100.0%) | 33.60 (67.7%) | 0.55 (100.0%) | 0.3379 | 0.1896 | +0.161 | 0.0% |
| **Qwen2.5-VL-7B-Instruct** *(VLM)* | `7B` / Apache-2.0 | **1.57** `[0.98, 2.02]` | 3.49 | 13.22 (85.7%) | 1.73 (100.0%) | 1.56 (100.0%) | 5.01 (100.0%) | 0.0336 | 0.0401 | -9.509 | 0.0% |
| **SMT++** (`smt-fp-grandstaff`) | `10.9M` / MIT | **1.26** `[0.58, 1.92]` | 2.55 | 11.73 (97.1%) | 4.02 (90.3%) | 5.32 (93.6%) | 0.43 (100.0%) | 0.0635 | 0.0537 | -0.351 | 0.0% |

**Methodology & Notes:**
- **Unified Single-Pipeline Protocol:** Each system receives raw page images only (no domain tags, oracle routing, or per-stratum tuning). Classical/specialized OMR baselines (`homr`, `Audiveris`) ran in offline Docker/CPU isolation; the neural GPU baselines — zero-shot general VLM `Qwen2.5-VL-7B-Instruct` (deterministic BF16 greedy ABC prompting + `music21` conversion) and full-page end-to-end OMR `SMT++` (`PRAIG/smt-fp-grandstaff`, a printed piano GrandStaff `**kern` checkpoint + `music21` conversion, evaluated zero-shot across the full multi-instrumentation pack) — ran on a single NVIDIA L4 GPU (24 GB VRAM).
- **Cross-Domain Bottleneck:** Classical and modern print-oriented OMR systems achieve strong clean-engraving performance (`CDR` `60.8`–`72.1`), but suffer severe catastrophic failure rates (`CFR`) on archival scans (`ALS`) and real handwritten manuscripts (`HWM`), driving composite `NS` sharply down as designed. Meanwhile, zero-shot general VLMs (`Qwen2.5-VL-7B-Instruct`) emit parseable notation on `81.6%` of pages but suffer severe pitch/rhythm hallucination loops (`S_note` `0.0336`, `mean_NTSR` `-9.509`), while single-domain neural specialists (`SMT++`, trained on printed piano GrandStaff) achieve stronger note fidelity on readable printed pages (`S_note` `0.0635`, `mean_NTSR` `-0.351`) but fail `**kern` parsing on `70.4%` of non-GrandStaff, degraded, and handwritten pages (`HWM` `0.43`, `100.0%` CFR).
- **Negative-Control Gate (`NC-Gate`):** All evaluated baselines (`homr`, `Audiveris`, `Qwen2.5-VL-7B-Instruct`, `SMT++`) pass all 4/4 Pillar 3B negative controls (`0.0%` Non-Music Hallucination Rate).
- **Upcoming Public Baselines:** `oemer 0.1.8` is currently running and will be added to this table upon completion.

---

## 5 Paired-Capture Controlled Isolation Protocols

Single-image averages conflate visual degradation with musical complexity. `OMR-NS` embeds five paired protocols sharing identical musical ground truth (`paired_anchor_id`):

- **Protocol A — Digital-to-Physical Capture Ladder (`CDR → Print → Scan → Phone`):** 8-step ladder (`A0` clean render through `A7` messaging-app-recompressed 4× photocopy phone photo) isolating pure camera/optics/codec drop (`Δ_photo`).
- **Protocol B — Multi-Engraver & Scale Invariance (`CDR`):** Identical scores across 5 engraver/font stacks, 3 staff-space scales ($s \in \{18, 9, 5\}$ px), and 2 raster modes (`Δ_engraver` and $s_{\text{cliff}}$).
- **Protocol C — Archival Multi-Digitization (`ALS` + `RWP`):** 600 DPI color cradle scan vs. legacy 1-bit microfilm vs. handheld reading-room phone photo of the exact same physical volume.
- **Protocol D — Handwritten & Hybrid Isolation (`HWM` + `RWP`):** Clean vs. pencil/highlighter-annotated physical pairs (`D1`), 3-writer calligraphic variance pairs (`D2`, `Δ_handwriting`), and manuscript flatbed scan vs. phone photo pairs (`D3`).
- **Protocol E — Factorized Single-Stressor Sweeps:** Controlled 5-step single-axis cliff probes for scale $s$, keystone $\theta$, blur $\sigma$, illumination gradient (EV), bleed-through $\alpha$, and localized occlusion area.

---

## 9 Adversarial Stress-Test Families (Anti-Saturation)

To prevent aggregate F1 from hiding rare, high-leverage failures, `OMR-NS` evaluates 9 stress-test families across 3 pillars:

1. **1A — Repetition, Copy-Loop & "Spot-the-Diff" Ostinato Traps** (Trap-Site Recall `TSR >= 95%`, Exact Bar-Count `100%`, Copy-Loop `0%`)
2. **1B — Language-Model Prior & "Anti-Prior" Traps** (Counterfactual Classics `PHR <= 2%`, Atonal/12-Tone, Polymeter & Irrational Tuplets)
3. **1C — Long-Context & Dense-Page Exhaustion** (24–32 staff orchestral & 2,000–4,500 symbol solo pages; `Q4/Q1 >= 0.95`, zero dropped systems)
4. **2A — Dense Graphical Collisions & Extreme Voice Packing** (cluster chords, 5–8 ledger lines, 3–4 voices/staff, graphic overlap storms)
5. **2B — Semantic & Structural Edge Cases** (cue notes, ossias, mid-bar clef/key/time changes, non-standard barlines, double-sided bleed-through)
6. **2C — Mixed-Domain & Heavily Annotated Score Traps** (pencil fingerings, bowings, highlighter, crossed-out bars, taped patch measures)
7. **3A — Paired Counterfactuals** (exact-score 5-view domain-drop matrix)
8. **3B — Out-of-Distribution (OOD) & Negative Controls** (non-music text/blank staff rejection `FP = 0`, 90°/180°/270° rotation invariance, 2-page spread ordering)
9. **3C — Multi-Page Continuity & Long-Horizon Coherence** (cross-page ties/slurs/hairpins/ottavas, French-score staff-identity tracking `>= 98%`)

---

## Specification Documents & Schemas

- [`docs/01-architecture-and-governance.md`](docs/01-architecture-and-governance.md) — Canonical 720-page + 80-fresh + 40-negative-control architecture, 3 orthogonal axes, 6×4 matrix, multi-page input contract, zero-leakage disjointness, blind pHash/8-gram scrubbing, semantic versioning & 92.0 escalation trigger.
- [`docs/02-notational-taxonomy.md`](docs/02-notational-taxonomy.md) — Complete musicological repertoire taxonomy (`T1.1`–`T3.7`) mapped to Complexity Tiers `T0`–`T5`, phenomenon-level edge-case quotas, early-music scope boundaries, non-Latin lyric scripts, organ/accordion/harp/Nashville extensions, and `<= 15%` per-engraver diversity cap.
- [`docs/03-visual-and-manuscript-domains.md`](docs/03-visual-and-manuscript-domains.md) — Quantitative physical grounding, `>= 98%` human-legibility floor, 4 domains × 4 Visual Severity levels (`V1`–`V4`), and Paired Protocols A–E.
- [`docs/04-metrics-and-human-utility.md`](docs/04-metrics-and-human-utility.md) — CER data model, 2-stage alignment, 4-axis scoring, `wFPP` / `NTSR` / `CFR`, Negative-Control Gate (`NC-Gate`), 3-bucket equivalence rules, canonical `NS`, and companion `NS_tier`.
- [`docs/05-adversarial-stress-tests.md`](docs/05-adversarial-stress-tests.md) — All 9 adversarial stress-test families and Private Parametric Trap Generator protocol.
- [`docs/06-submission-and-harness.md`](docs/06-submission-and-harness.md) — Sealed `test-v1` OCI/Docker container submission contract (`--input-dir`, `--output-dir`, network isolation, timeouts, zero oracle routing, multi-page & EXIF/DPI rules) and reproducible `omr-northstar` CLI workflow.
- [`schemas/page_metadata.schema.json`](schemas/page_metadata.schema.json) — JSON Schema (Draft 2020-12) for per-page physical, notational, provenance, and pairing metadata.
- [`schemas/report.schema.json`](schemas/report.schema.json) — JSON Schema (Draft 2020-12) for the canonical 4-layer `report.json` output.

---

## Quickstart (Reference Scorer, CLI & Tests)

`omr_northstar` is a zero-external-dependency Python 3.10+ reference package.

```bash
# Run unit tests
python -m unittest discover -s tests -v

# Score a page-level results JSON file and print the 4-layer leaderboard scorecard
python -m omr_northstar score path/to/results.json

# Emit canonical report.json (with per-page scores for paired permutation compare)
python -m omr_northstar score path/to/results.json --json-out report_a.json --include-per-page

# Compare two system report.json files (A vs B) with work-clustered permutation significance
python -m omr_northstar compare report_a.json report_b.json --n-perm 1000 --seed 42

# Validate a page metadata manifest against required fields, enums, and licensing rules
python -m omr_northstar validate-manifest path/to/dev_manifest.json

# Audit dev/test split disjointness (work_id, writer_id, capture_device_id)
python -m omr_northstar validate-splits path/to/dev_manifest.json path/to/test_manifest.json

# Validate a canonical report.json against the OMR-NS report schema
python -m omr_northstar validate-report report_a.json
```

---

## Mandatory Anti-Gaming Rules

1. **Single Unified Pipeline (Zero Oracle Routing):** Systems receive only raw page image(s) (single pages or ordered 2–4 page sequences). Ground-truth domain tags, visual severity, staff counts, instrumentation, or manual per-domain model switching are prohibited (internal learned routing is permitted).
2. **Work, Writer & Device Disjointness:** Zero work-level, edition-level, or manuscript-hand overlap between train/dev and sealed test splits.
3. **Full-Matrix Mandatory Entry & Anti-Abstention:** Missing strata score `0`; refusing to output on hard pages scores `0` and triggers `CFR`.
4. **Negative-Control Gate (`NC-Gate`):** Hallucinating notes on `n_gt = 0` non-music or blank-staff pages (`NC_FP > 0`) disqualifies a submission from Verified Leaderboard Status.
5. **Rolling Fresh Holdout Audit:** A `> 3.0`-point drop from `test-v1` to `fresh-YYYY` triggers a public contamination flag.
6. **Mandatory Escalation Trigger:** When any verified system exceeds `92.0 NS` on `test-v1`, `v2.0` is triggered, retiring saturated `T0` weight into expanded `T4`–`T5` slices.

---

## License

- **Code, Reference Scorer, CLI, Tests, and JSON Schemas:** [MIT License](LICENSE)
- **Specifications and Documentation (`README.md`, `docs/`):** [CC-BY-4.0](LICENSE)
- **Test Data (`pilot_dev_v0_1/` & [Hugging Face v0.1 Pilot Pack](https://huggingface.co/datasets/gregoreesmaa/omr-northstar-v0.1)):** [CC-BY-4.0](LICENSE) overall (`CC0-1.0` / `PDM-1.0` / `CC-BY-4.0`, 100% permissive Core Track, zero `CC-BY-NC`; per-page licenses and attributions in the dataset `LICENSE` and `manifest.json`)
