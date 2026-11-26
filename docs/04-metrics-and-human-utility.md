# 04 — Metrics, Ground-Truth Equivalence & Human-Utility Specification

## 0. Canonical Evaluation Representation (CER) & Two-Stage Alignment

Raw token Sequence Error Rate (SER) is sensitive to serialization order, XML verbosity, and arbitrary voice/ID numbering; conversely, forgiving MIDI-only F1 hides spelling, notation, structural, and expressive errors. All evaluation operates on a **Canonical Evaluation Representation (CER)** extracted deterministically from standard digital notation output (MusicXML / MEI), never on raw internal tokens.

### 0.1 CER Data Model

Each page/piece is parsed into an ordered sequence of **measures** $m \in \{1, \dots, M\}$, where each measure contains:
- **Context state** per staff: `(clef, key_sig, time_sig, ottava, barline_left, barline_right, repeat_markers, volta_bracket)`.
- **Note events:** $e = (\text{staff}, \text{voice}, t, d, \text{step}, \text{alter}, \text{octave}, \text{tie\_start}, \text{tie\_stop}, \text{grace}, \text{tuplet}, \text{cue})$, where onset $t \in \mathbb{Q}_{\ge 0}$ and written duration $d \in \mathbb{Q}_{>0}$ are exact rational fractions of a whole note.
- **Rest events:** $r = (\text{staff}, \text{voice}, t, d, \text{is\_full\_measure})$.
- **Expressive attachments:** point symbols anchored at `(staff, t)` (or note-attached) and spanners anchored at `(staff, t_start, t_end)`.
- **System/Page graph:** mapping of staves to logical parts/instruments across system and page breaks.

### 0.2 Two-Stage Hierarchical Alignment (Prevents Cascade Collapse)

1. **Measure-Level Sequence Alignment:** Align predicted measures $\hat{\mathcal{M}}$ to ground-truth measures $\mathcal{M}$ via Needleman–Wunsch dynamic programming with measure similarity:
   $$\text{sim}(\hat{m}, m) = 0.7 \cdot \text{Dice}\big(\text{multiset}(p, d)_{\hat{m}},\; \text{multiset}(p, d)_m\big) + 0.3 \cdot \mathbb{I}[\text{barline\_type}(\hat{m}) = \text{barline\_type}(m)]$$
   and gap penalty $\gamma = 0.25$. Unaligned GT measures count as 100% false negatives; unaligned predicted measures count as 100% false positives.
2. **Optimal Voice-Bijection + Bipartite Event Matching:** Within each aligned system/measure and staff, solve the linear sum assignment (Hungarian algorithm) over voice-ID bijections $\pi: \hat{V} \leftrightarrow V$ maximizing matched notes, then perform exact bipartite matching over notes/rests. This eliminates arbitrary voice-numbering penalties (`v1`/`v2` swapped) while strictly penalizing voice merging or splitting.

---

## 1. Four-Axis Scoring Hierarchy

### Axis 1: Core Note Accuracy (`S_note`)

Evaluated across a 3-tier diagnostic ladder:

| Tier | Match Criteria (Exact Rational Time) | Diagnostic Purpose |
|---|---|---|
| **T1a: Sounding** | MIDI pitch $p \in [0, 127]$ + measure-relative onset $t$ | Basic pitch/onset detection |
| **T1b: Rhythmic** | T1a + written duration $d$ (plus grace/tuplet flag) | Rhythm & duration fidelity |
| **T1c: Strict Engraving (Headline)** | T1b + exact diatonic spelling `(step, alter, octave)` + staff assignment + voice class under optimal $\pi$ + chord/cue flags | True engraving-grade note accuracy |

$$S_{\text{note}}(P) = F1_{\text{strict}}(P) = \frac{2\,|M_{\text{strict}}|}{|N_{\text{pred}}| + |N_{\text{gt}}|}$$

Secondary diagnostics logged alongside `S_note`: Conditional Spelling Accuracy ($\text{Acc}_{\text{spell}}$), Conditional Duration Accuracy ($\text{Acc}_{\text{dur}}$), and Rest F1 ($F1_{\text{rest}}$).

### Axis 2: Structural & Rhythmic Integrity (`S_struct`)

1. **Measure-Exact Rate (`MER`):** Fraction of GT measures with 100% match on strict notes, rests, barlines, and beat sum.
2. **Barline Alignment F1 (`F1_bar`):** Exact match on measure boundary alignment and barline subtype (`regular`, `double`, `final`, `repeat-start`, `repeat-end`, `dashed`).
3. **Beat-Sum Validity Rate (`BSVR`):** Fraction of predicted `(measure, staff, voice)` streams whose rational duration sum equals the active time signature (exempting GT-annotated anacruses, cadenzas, and mid-measure repeats).
4. **Active Context State Accuracy (`A_state`):** Evaluated at *every matched note onset* for exact equality of active `(clef, key_sig, time_sig, ottava)`.
5. **Repeat / Volta Roadmap Similarity (`A_road`):** Unroll GT and predicted repeat roadmaps into measure sequences $R_{\text{gt}}, R_{\text{pred}}$; score via $1 - \frac{\text{Lev}(R_{\text{pred}}, R_{\text{gt}})}{\max(|R_{\text{pred}}|, |R_{\text{gt}}|)}$.
6. **Multi-System / Multi-Page Continuity (`S_cont`):** F1 over part-to-staff grouping and cross-system ties/slurs/hairpins.

$$S_{\text{struct}}(P) = 0.25\,\text{MER} + 0.20\,F1_{\text{bar}} + 0.15\,\text{BSVR} + 0.20\,A_{\text{state}} + 0.10\,A_{\text{road}} + 0.10\,S_{\text{cont}}$$

*(Components not present on a page have their weights renormalized across active components.)*

### Axis 3: Expressive & Non-Note Fidelity (`S_expr`)

Category-macro-averaged F1 across 5 non-note layers:
1. **Mid-Piece Context Changes (`F1_ctx`):** Explicit mid-piece clef, key, time, and ottava markers.
2. **Articulations & Ornaments (`F1_art`):** Staccato, accent, tenuto, marcato, fermata, trill, mordent, turn, arpeggio, breath mark.
3. **Dynamics & Hairpins (`F1_dyn`):** Point dynamics within $\pm \frac{1}{8}$ beat; hairpins matched on type + span $\text{IoU} \ge 0.70$.
4. **Slurs & Ties (`F1_slur`):** Ties matched strictly on `(start_note, end_note)`; slurs matched on `(staff, start_onset, end_onset)` with span $\text{IoU} \ge 0.75$.
5. **Textual & Symbolic Layers (`F1_text`):** Lyrics (syllable $1 - \text{CER}$), chord symbols `(root, quality, extensions, bass)`, and tempo/metronome/rehearsal marks.

$$S_{\text{expr}}(P) = \frac{1}{|\mathcal{C}_{\text{active}}(P)|} \sum_{c \in \mathcal{C}_{\text{active}}(P)} F1_c(P)$$

### Axis 4: Playability vs. Engraving Fidelity (`S_play` vs. `S_eng`)

- **Playability Score (`S_play`):** Render both GT and prediction through a deterministic reference performance engine (expanding repeats/voltas, resolving ties, applying key signatures, accidentals, and ottavas) and compute unrolled sounding MIDI-event F1 on `(midi_pitch, unrolled_beat_onset, sounding_duration)`.
- **Engraving–Playability Gap (`Δ = S_play - S_eng`):** Exposes models that sound acceptable in MIDI while producing unusable notation.

---

## 2. Human-Utility & Product Metrics

### 2.1 Weighted Fixes Per Page (`wFPP`) & Root-Cause Bulk-Repair Factoring

We compute a **minimum-cost editor script** transforming prediction → GT in a standard notation editor, measured in **Normalized Edit Units (`NEU`)** where $1.0\text{ NEU} \approx 1$ atomic symbol edit ($\approx 2.5\text{–}3.0\text{ s}$ for a proficient engraver).

**Root-Cause Bulk-Repair Factoring:** Before counting per-note edits, the scorer evaluates bulk structural repairs (fixing a wrong clef + diatonic span shift; fixing a key signature + respelling accidentals; inserting/deleting a barline + re-barring) and takes the minimum of `(bulk structural fix + residual note fixes)` vs. `(individual note fixes)`.

| Cost Tier | `NEU` | Representative Notation-Editor Operations |
|---|---:|---|
| **T0: Micro** | 0.5 | Add/delete/change staccato, accent, tenuto, fermata, dynamic mark, tie toggle |
| **T1: Atomic** | 1.0 | Fix single note pitch, accidental/enharmonic respell, duration tweak (no reflow), slur endpoint adjust, lyric syllable edit, chord symbol edit |
| **T2: Local Structural** | 2.5 | Insert/delete note or rest (cursor + rhythm + pitch), reassign note voice/staff, fix tuplet bracket |
| **T3: Measure/Context Structural** | 5.0 | Insert/delete barline + re-bar, fix clef/key/time change (+ bulk span shift), fix repeat/volta bracket |
| **T4: System Structural** | 12.0 | Fix merged/split grand staff, repair part/instrument bracket across system break |

### 2.2 Rational Musician Re-Entry Cap & Net Time Saved Ratio (`NTSR`)

For every measure $m$, edit cost is capped at full manual re-entry:

$$\text{Cost}_{\text{eff}}(m) = \min\!\Big(\text{EditCost}(\hat{m} \to m),\; c_{\text{clear}} + \text{TranscribeCost}(m)\Big), \qquad c_{\text{clear}} = 1.5\text{ NEU}$$

Summed over page $P$:
- **Weighted Fixes Per Page:** $\text{wFPP}(P) = \sum_{m \in P} \text{Cost}_{\text{eff}}(m)$ (plus density-normalized `wFP100N` = NEU per 100 GT notes).
- **Net Time Saved Ratio:**
  $$\text{NTSR}(P) = 1 - \frac{\text{wFPP}(P)}{\text{TranscribeCost}(P)} \in (-\infty, 1]$$
  - `NTSR = 1.0`: Zero-touch perfection.
  - `NTSR >= 0.70`: High-utility product bar (`>= 70%` manual engraving time saved).
  - `NTSR <= 0`: Worse than starting from a blank page. Report **Negative-Utility Page Rate (`NUPR`)** = fraction of pages with `NTSR <= 0`.

### 2.3 Catastrophic Failure Penalties & Rate (`CFR`)

A page triggers a **Catastrophic Failure** flag (`CF = 1`) if *any* of the following occur:
1. **Unparseable / Schema-Invalid Output:** Fails reference MusicXML/MEI parser $\implies Q(P) = 0$, $\text{wFPP}(P) = 1.2 \cdot \text{TranscribeCost}(P)$.
2. **Skipped / Empty Page:** $N_{\text{pred}} = 0$ when $N_{\text{gt}} > 0 \implies Q(P) = 0$.
3. **Dropped System(s):** Any GT system has note recall `< 15%`. Applies multiplicative penalty $0.85^{k_{\text{dropped}}}$ to $Q(P)$.
4. **Hallucination / Repetition Loop:** $N_{\text{pred}} > 1.5 \cdot N_{\text{gt}}$ on a page, OR a predicted measure $n$-gram ($n \ge 2$) repeats $\ge 3\times$ consecutively when absent in GT. Applies penalty $\text{LoopPen}(P) = \min\!\big(1,\, \frac{1.5\,N_{\text{gt}}}{\max(N_{\text{pred}}, 1)}\big)$.
5. **Structural Collapse:** Page $\text{BSVR} < 0.50$ OR $|M_{\text{pred}} - M_{\text{gt}}| / M_{\text{gt}} > 0.25$.

Report $\text{CFR}_s$ per domain stratum and $\text{CFR}_{\max} = \max_s \text{CFR}_s$.

---

## 3. Three-Bucket Ground-Truth Equivalence Policy

| Bucket | Rule | Examples |
|---|---|---|
| **A: Strictly Equivalent** *(Canonicalized away — 0 penalty)* | Visually/internally equivalent representations with identical musical meaning | Stem direction (unless disambiguating polyphony), beaming groupings *(auxiliary only)*, exact $x/y$ coordinates, slur Bézier points, internal voice/part IDs (matched via optimal bijection $\pi$), note order within a chord, full-measure rest vs. equal-duration rest, redundant cautionary accidentals. |
| **B: Musicologically Distinct** *(Penalized in Engraving; tolerated only in Playability)* | Changes harmonic, rhythmic, or structural reading | Enharmonic spelling (`B#3` vs `C4`, `Fx` vs `G`), within-measure tie/syncopation spelling, voice merging or splitting, wrong staff in grand-staff notation, omitting a key signature and writing inline accidentals instead. |
| **C: Genuine Ambiguity** *(Accepted via Multi-Reference `alt_set`)* | Real human disagreement in handwritten manuscripts or heavily degraded scans | Ambiguous slur endpoint ($\pm 1$ note), illegible ink blot vs. staccato dot, tie vs. slur on repeated notes, smeared accidental. |

All `ALS` and `HWM` test pieces undergo dual-independent human transcription + 3rd-party adjudication, and inter-annotator agreement (`Annotator A` vs `Annotator B` via the exact reference scorer) is published as the empirical human ceiling per stratum.

---

## 4. Composite North Star Score (`NS`)

### Per-Page Quality Score $Q(P) \in [0, 100]$

$$Q_{\text{raw}}(P) = 100 \cdot \Big( 0.40\,S_{\text{note}}(P) + 0.25\,S_{\text{struct}}(P) + 0.20\,S_{\text{expr}}(P) + 0.15\,\max\big(0,\, \text{NTSR}(P)\big) \Big)$$

$$Q(P) = Q_{\text{raw}}(P) \cdot \mathbb{I}[\text{parseable} \land \neg\text{empty}] \cdot (0.85)^{k_{\text{dropped\_sys}}} \cdot \text{LoopPen}(P)$$

Each domain stratum score $Q_s$ ($s \in \{\text{CDR}, \text{ALS}, \text{RWP}, \text{HWM}\}$) is the **piece-level macro-average** of $Q(P)$, reported with 95% piece-level cluster-bootstrap confidence intervals.

### Cross-Domain Composite (`NS`)

$$M_{\text{geom}} = \left(\prod_{s \in \mathcal{D}} \max(Q_s,\, 0.1)\right)^{1/4}$$

$$NS = \Big( 0.75\,M_{\text{geom}} + 0.25\,\min_{s \in \mathcal{D}} Q_s \Big) \cdot \Big(1 - 0.50\,\max_{s \in \mathcal{D}} \text{CFR}_s\Big)$$

### Calibration Examples (Verified in `tests/test_scoring.py`)

| System Profile | `(CDR, ALS, RWP, HWM)` | `CFR_max` | `M_geom` | `NS` |
|---|---|---:|---:|---:|
| Clean-only specialist | `(98, 92, 35, 10)` | `0.60` | `42.15` | **`23.88`** (~23.9) |
| Printed+Photo strong, HW weak | `(95, 88, 80, 30)` | `0.25` | `66.93` | **`50.48`** (~50.5) |
| Balanced universal system | `(90, 84, 76, 68)` | `0.04` | `79.06` | **`74.77`** (~74.8) |
