# OMR-NorthStar

**The real-world benchmark for Optical Music Recognition — from clean digital scores to phone photos and handwritten manuscripts.**

[Hugging Face Dataset (v0.1)](https://huggingface.co/datasets/gregoreesmaa/omr-northstar-v0.1) · [Public Leaderboard](#the-reality-check-public-leaderboard-v01-125-pages) · [Quickstart](#download--run-in-30-seconds) · [How Scoring Works](#how-scoring-works-in-60-seconds) · [Specs](docs/01-architecture-and-governance.md) · [License](LICENSE)

---

Snap a phone photo of a slightly yellowed Chopin nocturne on a music stand — or open a digitized Mozart manuscript — and feed it into today's "95% accurate" Optical Music Recognition (OMR) tools. What comes back is almost always a broken MusicXML file: dropped voices, scrambled measures, phantom notes, and missing dynamics. Fixing it takes longer than retyping the piece from scratch.

If you've ever tried OMR in the real world, you already know this. Yet for years, published OMR benchmarks have looked nearly solved. **OMR-NorthStar** closes the gap between benchmark claims and musical reality with one honest, model-agnostic yardstick: a permissive real-world evaluation suite, JSON Schemas, and a zero-dependency Python scorer that measures whether an OMR system actually saves a musician time.

---

## The Reality Check: Public Leaderboard (v0.1, 125 Pages)

Every system below was tested **zero-shot with out-of-the-box defaults** on the exact same 125-page public v0.1 pack — raw page images in, MusicXML out, with no domain hints or manual model switching.

```text
Oracle GT (ceiling)     ████████████████████████████████████████  99.53
homr 0.7.1              ██████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  14.67
Audiveris 5.11.0        █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   2.81
Qwen2.5-VL-7B (VLM)     █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   1.57
SMT++ (GrandStaff)      █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   1.26
oemer 0.1.8             █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   0.79
                        0        25        50        75       100
                        Composite NorthStar Score (NS, 0–100)
```

**OMR is still wide open.** Even the strongest open-source baseline scores **14.67 / 100** — not because it can't read spotless sheet music, but because the moment paper ages, lighting shifts, or human handwriting appears, modern systems fall off a cliff.

| System | NorthStar Score (95% CI) | Clean Digital | Archival Scans | Phone Photos | Handwritten | Time Saved (`NTSR`) | Blank/Text Gate |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Oracle GT** *(ceiling)* | **99.53** `[99.32, 99.68]` | 99.70 | 99.66 | 99.66 | 99.34 | +1.000 | PASS (0.0%) |
| **homr** `0.7.1` *(AGPL-3.0)* | **14.67** `[12.02, 17.78]` | 60.76 | 35.65 | 69.21 | 8.56 | +0.324 | PASS (0.0%) |
| **Audiveris** `5.11.0` *(AGPL-3.0)* | **2.81** `[1.49, 3.80]` | 72.07 | 2.15 | 33.60 | 0.55 | +0.161 | PASS (0.0%) |
| **Qwen2.5-VL-7B** *(Apache-2.0)* | **1.57** `[0.98, 2.02]` | 13.22 | 1.73 | 1.56 | 5.01 | -9.509 | PASS (0.0%) |
| **SMT++** `smt-fp-grandstaff` *(MIT)* | **1.26** `[0.58, 1.92]` | 11.73 | 4.02 | 5.32 | 0.43 | -0.351 | PASS (0.0%) |
| **oemer** `0.1.8` *(MIT)* | **0.79** `[0.41, 1.05]` | 10.68 | 1.26 | 1.96 | 0.52 | -0.191 | PASS (0.0%) |

**What the numbers actually show:**
- **Print tools break outside the lab:** `Audiveris` (`72.07`) and `homr` (`60.76`) read clean digital scores reasonably well, but suffer `92.9%`–`100.0%` catastrophic failure rates on real handwritten manuscripts and up to `100.0%` on archival scans.
- **General VLMs hallucinate fluency:** `Qwen2.5-VL-7B-Instruct` outputs parseable notation on `81.6%` of pages, yet gets the actual notes wrong (`S_note = 0.0336`) and loops so badly that fixing its output takes **~10× longer than retyping by hand** (`mean_NTSR = -9.509`).
- **Narrow neural specialists don't generalize:** `SMT++` (`PRAIG/smt-fp-grandstaff`, trained on printed piano GrandStaff) fails `**kern` parsing on `70.4%` of non-GrandStaff, degraded, and handwritten pages.
- **Zero false music on blank/text pages:** All five baselines pass `4/4` non-music negative controls (`0.0%` hallucination rate). *(Upcoming public baselines: `Clarity-OMR`, `OmniOMR/Zeus`, and `Score-Reader` are currently queued/running.)*

<details>
<summary><strong>Full 12-column metrics, failure rates (CFR), and evaluation protocol</strong></summary>

Scored with the canonicalized `omr-northstar` CER reference evaluator (`89c698c`) on the 125-page v0.1 pilot pack (**31 Clean Digital `CDR` + 31 Archival Scans `ALS` + 31 Phone Photos `RWP` + 28 real historical Handwritten `HWM` + 4 Negative Controls**: 2 blank staff pages + 2 text program-notes pages).

| System | Version / License | NS Score (95% CI) | Tier-W NS | CDR Q (CFR) | ALS Q (CFR) | RWP Q (CFR) | HWM Q (CFR) | `S_note` | `S_expr` | `mean_NTSR` | `NMHR` |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Oracle GT** *(ceiling)* | Ground Truth | **99.53** `[99.32, 99.68]` | 99.69 | 99.70 (0.0%) | 99.66 (0.0%) | 99.66 (0.0%) | 99.34 (0.0%) | 1.0000 | 1.0000 | +1.000 | 0.0% |
| **homr** | `0.7.1` / AGPL-3.0 | **14.67** `[12.02, 17.78]` | 25.24 | 60.76 (31.4%) | 35.65 (67.7%) | 69.21 (19.4%) | 8.56 (92.9%) | 0.5037 | 0.2545 | +0.324 | 0.0% |
| **Audiveris** | `5.11.0` / AGPL-3.0 | **2.81** `[1.49, 3.80]` | 11.90 | 72.07 (22.9%) | 2.15 (100.0%) | 33.60 (67.7%) | 0.55 (100.0%) | 0.3379 | 0.1896 | +0.161 | 0.0% |
| **Qwen2.5-VL-7B-Instruct** *(VLM)* | `7B` / Apache-2.0 | **1.57** `[0.98, 2.02]` | 3.49 | 13.22 (85.7%) | 1.73 (100.0%) | 1.56 (100.0%) | 5.01 (100.0%) | 0.0336 | 0.0401 | -9.509 | 0.0% |
| **SMT++** (`smt-fp-grandstaff`) | `10.9M` / MIT | **1.26** `[0.58, 1.92]` | 2.55 | 11.73 (97.1%) | 4.02 (90.3%) | 5.32 (93.6%) | 0.43 (100.0%) | 0.0635 | 0.0537 | -0.351 | 0.0% |
| **oemer** | `0.1.8` / MIT | **0.79** `[0.41, 1.05]` | 3.09 | 10.68 (97.1%) | 1.26 (100.0%) | 1.96 (100.0%) | 0.52 (100.0%) | 0.0467 | 0.0412 | -0.191 | 0.0% |

- **Hardware & Isolation:** `homr 0.7.1`, `Audiveris 5.11.0`, and `oemer 0.1.8` ran in offline Docker/CPU isolation. `Qwen2.5-VL-7B-Instruct` (deterministic BF16 greedy ABC prompting + `music21` conversion) and `SMT++` (`PRAIG/smt-fp-grandstaff` `**kern` + `music21` conversion) ran on a single NVIDIA L4 GPU (24 GB VRAM).
- **v0.1 Pilot Caveat:** In v0.1, `ALS` and `RWP` are pilot degradations derived from the `CDR` anchors pending full physical flatbed scanning and smartphone capture for v1.0; `HWM` contains real historical handwritten manuscripts.

</details>

---

## Download & Run in 30 Seconds

The unpacked **125-page v0.1 pilot pack** (`125` PNG images + `46` canonical MusicXML ground-truth files + `manifest.json`) is hosted on Hugging Face under a 100% permissive license (`CC-BY-4.0` overall: `97` `CC0-1.0`, `20` `CC0-1.0`/`PDM-1.0`, `8` `CC-BY-4.0`; zero `CC-BY-NC`).

```bash
# 1. Download the unpacked 125-page v0.1 pilot pack from Hugging Face
huggingface-cli download gregoreesmaa/omr-northstar-v0.1 \
  --repo-type dataset \
  --local-dir ./omr-northstar-v0.1

# 2. Install the zero-dependency Python 3.10+ reference scorer & verify tests
git clone https://github.com/gregoreesmaa/omr-northstar.git
cd omr-northstar
pip install -e .
python -m unittest discover -s tests -v

# 3. Score predictions, compare two systems with permutation tests, or validate manifests
python -m omr_northstar score path/to/results.json --json-out report_a.json --include-per-page
python -m omr_northstar compare report_a.json report_b.json --n-perm 1000 --seed 42
python -m omr_northstar validate-manifest ./omr-northstar-v0.1/manifest.json
python -m omr_northstar validate-splits path/to/dev_manifest.json path/to/test_manifest.json
python -m omr_northstar validate-report report_a.json
```

> **Dataset:** [huggingface.co/datasets/gregoreesmaa/omr-northstar-v0.1](https://huggingface.co/datasets/gregoreesmaa/omr-northstar-v0.1) — *(v0.1 `ALS` and `RWP` are pilot degradations derived from `CDR` anchors pending full physical capture in v1.0; `HWM` contains real historical manuscripts.)*

---

## Why OMR-NorthStar Exists

Why did a decade of "95%+ accuracy" papers leave musicians with tools that fail on a single phone photo? Because legacy benchmarks had four blind spots:

1. **One clean world at a time.** Most datasets test either spotless synthetic renders *or* a single archival corpus. A model can ace one and be completely blind to the other.
2. **Simple music hides structural collapse.** Benchmarks dominated by monophonic melodies saturate early while hiding total breakdown on polyphony, vocal+piano, chamber, and orchestral scores.
3. **Toy noise isn't real life.** Adding Gaussian blur to a PDF doesn't capture open-book gutter curl, plastic-sleeve glare, reverse-page bleed-through, or real human handwriting.
4. **Token edit rates lie to musicians.** A 3% symbol error rate sounds impressive — until one missed clef or dropped staff ruins every measure after it.

---

## How Scoring Works in 60 Seconds

### 1. Ask the Only Question That Matters: *"Would a Musician Save Time?"*
After aligning measures and matching voices on a normalized music representation (`CER`), every music page earns a quality score `Q(P)` from `0` to `100` across four weighted pillars:

- **40% Note Accuracy (`S_note`)** — Right pitches, rhythms, and voice assignments.
- **25% Structural Fidelity (`S_struct`)** — Right clefs, key/time signatures, barlines, and system structure.
- **20% Expressive Detail (`S_expr`)** — Dynamics, articulations, slurs, ties, lyrics, and markings that make a score playable.
- **15% Net Time Saved (`NTSR`)** — Based on **Weighted Fixes Per Page (`wFPP`)** vs. retyping from scratch (`> 0` saves time; `< 0` means throwing the output away is faster).

Unparseable MusicXML, dropped systems, and copy-paste hallucination loops are penalized directly.

### 2. Clean-Only Specialists Can't Game the Headline Score
Instead of a forgiving arithmetic average, the four visual domain scores are combined with a **geometric mean + worst-domain floor**, then penalized by the **worst-domain catastrophic failure rate (`CFR_max`)**:

| System Profile | Domain Scores `(Clean, Scan, Photo, Handwritten)` | Worst Failure Rate | NorthStar Score (`NS`) |
|---|---|---:|---:|
| Clean-only specialist | `(98, 92, 35, 10)` | `60%` | **23.88** |
| Strong print + photo, weak handwriting | `(95, 88, 80, 30)` | `25%` | **50.48** |
| Balanced universal system | `(90, 84, 76, 68)` | `4%` | **74.77** |

A companion **Tier-Weighted Score (`NS_tier`)** tracks musical-complexity headroom separately, and a mandatory **Non-Music Gate (`NC-Gate`)** requires **`0%` hallucinated notes** on blank staff paper and text pages.

<details>
<summary><strong>Exact scoring formulas (<code>Q(P)</code>, <code>NS</code>, <code>NS_tier</code>, <code>NTSR</code>)</strong></summary>

Every scored music page is evaluated on its **Canonical Evaluation Representation (`CER`)** via two-stage Needleman–Wunsch measure alignment and Hungarian voice-bijection matching:

$$Q_{\text{raw}}(P) = 100 \cdot \Big( 0.40\,S_{\text{note}}(P) + 0.25\,S_{\text{struct}}(P) + 0.20\,S_{\text{expr}}(P) + 0.15\,\max\big(0,\, \text{NTSR}(P)\big) \Big)$$

$$Q(P) = Q_{\text{raw}}(P) \cdot \mathbb{I}[\text{parseable} \land \neg\text{empty}] \cdot (0.85)^{k_{\text{dropped\_sys}}} \cdot \text{LoopPen}(P)$$

where $\text{NTSR}(P) = 1 - \frac{\text{wFPP}(P)}{\text{TranscribeCost}(P)}$ is the **Net Time Saved Ratio** derived from root-cause-factored **Weighted Fixes Per Page (`wFPP`)** capped at rational musician re-entry cost.

**Headline Cross-Domain NorthStar Score (`NS`):**

$$M_{\text{geom}} = \left(\prod_{s \in \mathcal{D}} \max(Q_s,\, 0.1)\right)^{1/4}, \qquad NS = \Big( 0.75\,M_{\text{geom}} + 0.25\,\min_{s \in \mathcal{D}} Q_s \Big) \cdot \Big(1 - 0.50\,\max_{s \in \mathcal{D}} \text{CFR}_s\Big)$$

**Companion Tier-Weighted Score (`NS_tier`)** with weights $w_t = \{T0: 5\%,\; T1: 15\%,\; T2: 25\%,\; T3: 25\%,\; T4: 18\%,\; T5: 12\%\}$:

$$M_{\text{tier,geom}} = \prod_{t \in \mathcal{T}} \max(Q_t,\, 0.1)^{w_t}, \qquad NS_{\text{tier}} = \Big( 0.75\,M_{\text{tier,geom}} + 0.25\,\min_{t \in \mathcal{T}} Q_t \Big) \cdot \Big(1 - 0.50\,\max_{t \in \mathcal{T}} \text{CFR}_t\Big)$$

Negative-control pages (`n_gt = 0`) are excluded from `Q(P)` so `NTSR` never divides by zero, and enforced via `NC-Gate` (`NC_FP = 0` and `100%` two-page spread ordering; violations trigger `[NC-FAIL]`). Full math in [`docs/04-metrics-and-human-utility.md`](docs/04-metrics-and-human-utility.md).

</details>

---

## What's Inside the Benchmark

Every page is classified along **three independent axes** so difficulty never hides behind averages:

- **4 Real-World Visual Domains:**
  - **Clean Digital Renders (`CDR`)** — Born-digital exports across `>= 6` engraving engines and music fonts (Dorico, MuseScore, LilyPond, Finale, Sibelius, Verovio).
  - **Archival & Library Scans (`ALS`)** — Real 17th–20th century engraved plates, lithography, offset, microfilm, foxing, gutter shadows, and bleed-through.
  - **Real-World Phone Photos (`RWP`)** — Handheld smartphone captures with uneven lighting, perspective tilt, page curl, plastic-sleeve glare, folds, and messaging-app compression.
  - **Handwritten Manuscripts (`HWM`)** — Real human handwriting from archival fair copies to hurried composer drafts and heavily pencil-annotated scores.
- **6 Musical Complexity Tiers (`T0`–`T5`):** From simple monophonic melodies (`T0`) and solo piano (`T1`) to vocal/chamber polyphony (`T2`), full orchestral scores (`T3`), virtuosic/contemporary notation (`T4`), and adversarial grandmaster traps (`T5`).
- **4 Visual Difficulty Levels (`V1`–`V4`):** Easy → Brutal optical difficulty measured independently of musical complexity.
- **Built-In Isolation & Stress Tests:** Paired captures of the *exact same score* across clean → scan → phone → handwriting isolate pure visual drop, while 9 adversarial trap families catch copy-loops, language-model hallucinations, dense collisions, and multi-page drift.

**v0.1 Live Now vs. v1.0 Target:** The **125-page v0.1 pilot pack** is live on Hugging Face today. The canonical **v1.0 architecture** scales to **720 Core Pages** (`240 dev-v1` / `480 sealed test-v1`, ~360 works, strict work/writer/device disjointness) + **80 pages/year rolling fresh holdout (`fresh-YYYY`)** + **40 negative controls (`16 dev / 24 test`)**, plus an optional BYOD non-commercial handwritten eval track (`research_nc_eval_only`).

<details>
<summary><strong>Full v1.0 6×4 target matrix, 5 paired protocols, and 9 adversarial stress-test families</strong></summary>

### 6-Tier × 4-Domain Core Matrix (720 Pages: Dev / Test)

| Tier | Name | Page Share | `NS_tier` Weight | `CDR` (Render) | `ALS` (Scan) | `RWP` (Photo) | `HWM` (Handwritten) | Tier Total (`Dev / Test`) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **T0** | **Atomic Fundamentals** | 10% | 5% | 6 / 12 | 6 / 12 | 6 / 12 | 6 / 12 | **24 / 48** |
| **T1** | **Standard Solo & Homophony** | 20% | 15% | 12 / 24 | 12 / 24 | 12 / 24 | 12 / 24 | **48 / 96** |
| **T2** | **Polyphony, Vocal & Chamber** | 25% | 25% | 15 / 30 | 15 / 30 | 15 / 30 | 15 / 30 | **60 / 120** |
| **T3** | **Large Ensemble & Orchestral** | 20% | 25% | 12 / 24 | 14 / 28 | 12 / 24 | 10 / 20 | **48 / 96** |
| **T4** | **Virtuosic & Contemporary** | 15% | 18% | 10 / 20 | 10 / 20 | 8 / 16 | 8 / 16 | **36 / 72** |
| **T5** | **Grandmaster & Adversarial** | 10% | 12% | 2 / 4 | 6 / 12 | 8 / 16 | 8 / 16 | **24 / 48** |
| **Total** | **Core Matrix** | **100%** | **100%** | **57 / 114** | **63 / 126** | **61 / 122** | **59 / 118** | **240 / 480 (720)** |

### 5 Paired-Capture Protocols (Shared Ground Truth via `paired_anchor_id`)
- **Protocol A (`CDR → Print → Scan → Phone`):** 8-step ladder (`A0` clean render to `A7` recompressed photocopy phone photo) isolating pure capture drop (`Δ_photo`).
- **Protocol B (Multi-Engraver & Scale):** Identical scores across 5 engraver/font stacks and 3 staff-space scales (`Δ_engraver`, `s_cliff`).
- **Protocol C (Archival Multi-Digitization):** 600 DPI cradle scan vs. 1-bit microfilm vs. handheld phone photo of the same physical volume.
- **Protocol D (Handwritten & Hybrid):** Clean vs. pencil/highlighter pairs, 3-writer variance pairs (`Δ_handwriting`), and flatbed scan vs. phone photo pairs.
- **Protocol E (Factorized Single-Stressor Sweeps):** 5-step single-axis cliff probes for scale, keystone tilt, blur, lighting gradient, bleed-through, and occlusion.

### 9 Adversarial Stress-Test Families (Anti-Saturation)
1. **1A — Repetition, Copy-Loop & "Spot-the-Diff" Ostinato Traps** (`TSR >= 95%`, Exact Bar-Count `100%`, Copy-Loop `0%`)
2. **1B — Language-Model Prior & "Anti-Prior" Traps** (Counterfactual Classics `PHR <= 2%`, Atonal/12-Tone, Polymeter & Irrational Tuplets)
3. **1C — Long-Context & Dense-Page Exhaustion** (24–32 staff orchestral & 2,000–4,500 symbol solo pages; `Q4/Q1 >= 0.95`, zero dropped systems)
4. **2A — Dense Graphical Collisions & Extreme Voice Packing** (cluster chords, 5–8 ledger lines, 3–4 voices/staff)
5. **2B — Semantic & Structural Edge Cases** (cue notes, ossias, mid-bar clef/key/time changes, non-standard barlines, bleed-through)
6. **2C — Mixed-Domain & Heavily Annotated Score Traps** (pencil fingerings, bowings, highlighter, crossed-out bars, taped patches)
7. **3A — Paired Counterfactuals** (exact-score 5-view domain-drop matrix)
8. **3B — Out-of-Distribution & Negative Controls** (non-music/blank-staff rejection `FP = 0`, 90°/180°/270° rotation invariance, 2-page spread ordering)
9. **3C — Multi-Page Continuity & Long-Horizon Coherence** (cross-page ties/slurs/hairpins/ottavas, French-score staff tracking `>= 98%`)

*(Optional `research_nc_eval_only` BYOD track supports eval-only pointers to `CC-BY-NC` handwritten corpora such as CVC-MUSCIMA / MUSCIMA++ as a zero-leakage check for permissive models; never redistributed commercially.)*

</details>

---

## Fair-Play Rules (Anti-Gaming)

1. **One Unified Pipeline:** Systems receive raw page image(s) only — no domain tags, staff counts, or manual per-domain model switching (learned internal routing is allowed).
2. **Strict Disjointness:** Zero work-, edition-, writer-, or capture-device overlap between dev and sealed test splits (`validate-splits`).
3. **No Cherry-Picking or Abstaining:** Missing strata score `0`; refusing to output on hard pages scores `0` and triggers catastrophic failure (`CFR`).
4. **Zero Hallucinations on Non-Music:** Any hallucinated note on blank staff paper or text pages (`NC_FP > 0`) disqualifies Verified Leaderboard Status.
5. **Fresh-Holdout Contamination & Saturation Guard:** A `> 3.0`-point drop on `fresh-YYYY` triggers a contamination flag; exceeding `92.0 NS` on `test-v1` automatically triggers `v2.0`.

---

## Specifications & Schemas

| File | What's Inside |
|---|---|
| [`docs/01-architecture-and-governance.md`](docs/01-architecture-and-governance.md) | 720-page v1.0 architecture, 3 orthogonal axes, disjointness, pHash/8-gram scrubbing, versioning |
| [`docs/02-notational-taxonomy.md`](docs/02-notational-taxonomy.md) | Musicological repertoire taxonomy (`T0`–`T5`), edge-case quotas, early-music & non-Latin scope |
| [`docs/03-visual-and-manuscript-domains.md`](docs/03-visual-and-manuscript-domains.md) | Physical grounding, `>= 98%` human-legibility floor, visual severity `V1`–`V4`, Paired Protocols A–E |
| [`docs/04-metrics-and-human-utility.md`](docs/04-metrics-and-human-utility.md) | CER alignment, 4-axis scoring, `wFPP` / `NTSR` / `CFR`, `NC-Gate`, `NS` & `NS_tier` |
| [`docs/05-adversarial-stress-tests.md`](docs/05-adversarial-stress-tests.md) | All 9 adversarial stress-test families & Parametric Trap Generator protocol |
| [`docs/06-submission-and-harness.md`](docs/06-submission-and-harness.md) | Sealed `test-v1` Docker container contract, network isolation, timeouts, CLI workflow |
| [`schemas/page_metadata.schema.json`](schemas/page_metadata.schema.json) | Draft 2020-12 JSON Schema for per-page metadata, provenance, and pairing |
| [`schemas/report.schema.json`](schemas/report.schema.json) | Draft 2020-12 JSON Schema for the canonical 4-layer `report.json` output |

---

## License

- **Code, Reference Scorer, CLI, Tests, and JSON Schemas:** [MIT License](LICENSE)
- **Specifications and Documentation (`README.md`, `docs/`):** [CC-BY-4.0](LICENSE)
- **Test Data (`pilot_dev_v0_1/` & [Hugging Face v0.1 Pilot Pack](https://huggingface.co/datasets/gregoreesmaa/omr-northstar-v0.1)):** [CC-BY-4.0](LICENSE) overall (`CC0-1.0` / `PDM-1.0` / `CC-BY-4.0`, 100% permissive Core Track, zero `CC-BY-NC`; per-page licenses and attributions in the dataset `LICENSE` and `manifest.json`)
