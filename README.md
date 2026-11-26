# OMR-NorthStar (`OMR-NS`)

**A Unified, Model-Agnostic North Star Benchmark for Optical Music Recognition**

Existing OMR benchmarks fail as a North Star for four reasons: they are single-domain (clean renders or one archival corpus), they saturate on monophonic/simple homophonic music while hiding polyphonic and orchestral collapse, they substitute synthetic noise for real photography and real handwriting, and they score with proxy token-edit rates that hide catastrophic structural errors.

**OMR-NorthStar (`OMR-NS`)** is an independent, model-agnostic benchmark specification, JSON Schema suite, and deterministic reference evaluator designed to measure true end-to-end Optical Music Recognition across all four real-world visual domains:

1. **Clean Digital Renders (`CDR`)** — Born-digital exports across `>= 6` distinct engraving engines and music fonts (e.g., Dorico/Bravura, MuseScore/Leland, LilyPond/Emmentaler, Finale/Maestro, Sibelius/Opus, Verovio/Leipzig, plus jazz/handwritten-style digital fonts).
2. **Archival & Library Scans (`ALS`)** — Real flatbed/overhead scans of 17th–20th century engraved plates, movable type, lithography, offset, microfilm, and aged library prints with foxing, bleed-through, gutter shadows, and stamps.
3. **Real-World Phone Photos (`RWP`)** — 100% real physical smartphone captures across uncontrolled lighting, perspective tilt, open-book gutter curl, plastic-sleeve glare, folds, photocopies, and messaging-app recompression.
4. **Handwritten Manuscripts (`HWM`)** — 100% real human handwriting spanning archival fair copies, modern commissioned human manuscripts, hurried composer drafts with cross-outs/paste-overs, and heavily annotated printed scores (`HWM-Hybrid`).

---

## The 6-Tier (`T0`–`T5`) × 4-Domain Matrix

Musical/engraving complexity is stratified orthogonally to visual domain across **720 core pages** split at an exact 1:2 ratio per cell into **240 Public Dev (`dev-v1`)** and **480 Sealed Private Test (`test-v1`)**, plus an **80-page Annual Fresh Holdout (`fresh-YYYY`)** to detect contamination over time.

| Tier | Name | Share | `CDR` (Render) | `ALS` (Scan) | `RWP` (Photo) | `HWM` (Handwritten) | Tier Total (`Dev / Test`) |
|---|---|---:|---:|---:|---:|---:|---:|
| **T0** | **Atomic Fundamentals** | 10% | 6 / 12 | 6 / 12 | 6 / 12 | 6 / 12 | **24 / 48** |
| **T1** | **Standard Solo & Homophony** | 20% | 12 / 24 | 12 / 24 | 12 / 24 | 12 / 24 | **48 / 96** |
| **T2** | **Polyphony, Vocal & Chamber** | 25% | 15 / 30 | 15 / 30 | 15 / 30 | 15 / 30 | **60 / 120** |
| **T3** | **Large Ensemble & Orchestral** | 20% | 12 / 24 | 14 / 28 | 12 / 24 | 10 / 20 | **48 / 96** |
| **T4** | **Virtuosic & Contemporary** | 15% | 10 / 20 | 10 / 20 | 8 / 16 | 8 / 16 | **36 / 72** |
| **T5** | **Grandmaster & Adversarial** | 10% | 2 / 4 | 6 / 12 | 8 / 16 | 8 / 16 | **24 / 48** |
| **Total** | **All Tiers** | **100%** | **57 / 114** | **63 / 126** | **61 / 122** | **59 / 118** | **240 / 480 (720)** |

---

## Composite North Star Score (`NS`)

Every page is evaluated on its **Canonical Evaluation Representation (CER)** via two-stage Needleman–Wunsch measure alignment and Hungarian voice-bijection matching across four axes:

$$Q_{\text{raw}}(P) = 100 \cdot \Big( 0.40\,S_{\text{note}}(P) + 0.25\,S_{\text{struct}}(P) + 0.20\,S_{\text{expr}}(P) + 0.15\,\max\big(0,\, \text{NTSR}(P)\big) \Big)$$

$$Q(P) = Q_{\text{raw}}(P) \cdot \mathbb{I}[\text{parseable} \land \neg\text{empty}] \cdot (0.85)^{k_{\text{dropped\_sys}}} \cdot \text{LoopPen}(P)$$

where $\text{NTSR}(P) = 1 - \frac{\text{wFPP}(P)}{\text{TranscribeCost}(P)}$ is the **Net Time Saved Ratio** derived from root-cause-factored **Weighted Fixes Per Page (`wFPP`)** capped at rational musician re-entry cost.

To prevent clean-only specialists from gaming arithmetic averages, the four domain stratum scores $Q_s$ ($s \in \{\text{CDR}, \text{ALS}, \text{RWP}, \text{HWM}\}$) are combined via a **Geometric Mean + Worst-Stratum Floor** penalized by the **Worst-Stratum Catastrophic Failure Rate ($\text{CFR}_{\max}$)**:

$$M_{\text{geom}} = \left(\prod_{s \in \mathcal{D}} \max(Q_s,\, 0.1)\right)^{1/4}$$

$$NS = \Big( 0.75\,M_{\text{geom}} + 0.25\,\min_{s \in \mathcal{D}} Q_s \Big) \cdot \Big(1 - 0.50\,\max_{s \in \mathcal{D}} \text{CFR}_s\Big)$$

**Illustrative Calibration:**
- *Clean-only specialist* `(98, 92, 35, 10)`, `CFR_max = 0.60` $\implies M_{\text{geom}} = 42.1,\; NS = \mathbf{23.9}$
- *Printed+Photo strong, HW weak* `(95, 88, 80, 30)`, `CFR_max = 0.25` $\implies M_{\text{geom}} = 66.9,\; NS = \mathbf{50.5}$
- *Balanced universal system* `(90, 84, 76, 68)`, `CFR_max = 0.04` $\implies M_{\text{geom}} = 79.1,\; NS = \mathbf{74.8}$

---

## 5 Paired-Capture Controlled Isolation Protocols

Single-image averages conflate visual degradation with musical complexity. `OMR-NS` embeds five paired protocols sharing identical musical ground truth (`paired_anchor_id`):

- **Protocol A — Digital-to-Physical Capture Ladder (`CDR → Print → Scan → Phone`):** 8-step ladder (`A0` clean render through `A7` WhatsApp-recompressed 4× Xerox phone photo) isolating pure camera/optics/codec drop (`Δ_photo`).
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

- [`docs/01-architecture-and-governance.md`](docs/01-architecture-and-governance.md) — 6-tier × 4-domain matrix, `dev-v1`/`test-v1`/`fresh-YYYY` splits, zero-leakage disjointness, blind pHash/8-gram scrubbing, semantic versioning & 92.0 escalation trigger.
- [`docs/02-notational-taxonomy.md`](docs/02-notational-taxonomy.md) — Complete musicological taxonomy (`T1.1`–`T3.7`), phenomenon-level edge-case quotas, and `<= 15%` per-engraver diversity cap.
- [`docs/03-visual-and-manuscript-domains.md`](docs/03-visual-and-manuscript-domains.md) — Quantitative physical grounding, `>= 98%` human-legibility floor, 4 domains × Easy/Medium/Hard/Brutal sub-tiers, and Paired Protocols A–E.
- [`docs/04-metrics-and-human-utility.md`](docs/04-metrics-and-human-utility.md) — CER data model, 2-stage alignment, 4-axis scoring, `wFPP` / `NTSR` / `CFR`, 3-bucket equivalence rules, and `NS` composite formula.
- [`docs/05-adversarial-stress-tests.md`](docs/05-adversarial-stress-tests.md) — All 9 adversarial stress-test families and Private Parametric Trap Generator protocol.
- [`schemas/page_metadata.schema.json`](schemas/page_metadata.schema.json) — JSON Schema (Draft 2020-12) for per-page physical, notational, provenance, and pairing metadata.
- [`schemas/report.schema.json`](schemas/report.schema.json) — JSON Schema (Draft 2020-12) for the canonical 4-layer `report.json` output.

---

## Quickstart (Reference Scorer & Tests)

`omr_northstar` is a zero-external-dependency Python 3.10+ reference package.

```bash
# Run unit tests
python -m unittest discover -s tests -v

# Score a page-level results JSON file and print the 4-layer leaderboard scorecard
python -m omr_northstar score path/to/results.json

# Emit canonical JSON report
python -m omr_northstar score path/to/results.json --json-out report.json

# Validate a page metadata manifest against required fields and enums
python -m omr_northstar validate-manifest path/to/manifest.json
```

---

## Mandatory Anti-Gaming Rules

1. **Single Unified Pipeline (Zero Oracle Routing):** Systems receive only raw page image(s). Ground-truth domain tags, staff counts, instrumentation, or manual per-domain model switching are prohibited (internal learned routing is permitted).
2. **Work, Writer & Device Disjointness:** Zero work-level, edition-level, or manuscript-hand overlap between train/dev and sealed test splits.
3. **Full-Matrix Mandatory Entry & Anti-Abstention:** Missing strata score `0`; refusing to output on hard pages scores `0` and triggers `CFR`.
4. **Rolling Fresh Holdout Audit:** A `> 3.0`-point drop from `test-v1` to `fresh-YYYY` triggers a public contamination flag.
5. **Mandatory Escalation Trigger:** When any verified system exceeds `92.0 NS` on `test-v1`, `v2.0` is triggered, retiring saturated `T0` weight into expanded `T4`–`T5` slices.

---

## License

- **Code, Reference Scorer, CLI, Tests, and JSON Schemas:** [MIT License](LICENSE)
- **Specifications and Documentation (`README.md`, `docs/`):** [CC-BY-4.0](LICENSE)
