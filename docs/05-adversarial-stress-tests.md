# 05 — Adversarial Stress-Tests & Anti-Saturation Design

Aggregate F1 hides rare, high-leverage failures: a single hallucinated accidental in a 16-bar ostinato costs 0.1% F1 but ruins the performance. `OMR-NS` separates **average-case reading** from **adversarial trap survival** via **9 stress-test families** across 3 pillars plus a **Private Parametric Trap Generator**.

---

## Pillar 1: Neural, Autoregressive & VLM Failure Traps

### 1A. Repetition, Copy-Loop & "Spot-the-Diff" Ostinato Traps
- **Failure mode targeted:** Autoregressive copy-loops (infinite repetition or early `<EOS>`), measure-count drift across visually identical bars, and copying bar $k-1$ instead of attending to bar $k$.
- **Test cases:**
  1. *Minimalist Ostinato Drift:* 16–32 consecutive identical arpeggio bars with exact 1-symbol mutations injected at unpredictable bars (semitone accidental flip, tie→slur, dropped 16th-note in a `15/16` bar).
  2. *Identical-Measure Count-Accuracy:* 8, 12, and 16 identical repeated bars (written-out and `%`/`//` measure-repeat signs).
  3. *Strophic Verse / Refrain Micro-Variants:* 3-verse vocal score with a 1-syllable rhythmic split in verse 2 and an altered pickup/ending in verse 3.
- **Pass criteria:**
  - **Trap-Site Recall (`TSR`) `>= 95%`** evaluated strictly on mutated symbols/bars.
  - **Exact Bar-Count Match Rate = `100%`**.
  - **Copy-Loop / Runaway-Generation Rate = `0%`**.

### 1B. Language-Model Prior & "Anti-Prior" Traps
- **Failure mode targeted:** Strong music language models and VLMs overriding visual evidence with harmonic/melodic priors or memorized canonical repertoire.
- **Test cases:**
  1. *Counterfactual Classics (Memorization Probe):* Famous canonical works (e.g., Bach BWV 846, Mozart K.545, Beethoven Für Elise, Chopin Op. 28 No. 4) with **3–8% surgical mutations** (unexpected accidentals, swapped clefs, altered time signatures, displaced octaves) engraved cleanly.
  2. *Atonal / 12-Tone / Non-Functional Harmony:* Second Viennese School–style passages, extreme intervallic leaps, dense per-note accidentals, and non-standard synthetic key signatures (e.g., `F# + Bb + C#`).
  3. *Asymmetric Meters, Polymeters & Irrational Tuplets:* Additive meters (`7/16`, `11/8`, `13/16`, `5/4 + 3/8`), simultaneous polymeters (`3/4` vs `7/8`), nested/irrational tuplets (`7:5`, `11:8`, `3:2` inside `5:4`).
- **Pass criteria:**
  - **Prior-Hallucination Rate (`PHR`) `<= 2%`** on Counterfactual Classics, where:
    $$\text{PHR} = \frac{\text{\# mutated sites predicted as the original canonical note}}{\text{\# mutated sites}}$$
  - **Mutation-Site Pitch/Accidental F1 `>= 95%`**.
  - **Rhythmic Duration & Bar-Sum Validity `>= 98%`** on asymmetric/irrational meters.

### 1C. Long-Context & Dense-Page Exhaustion
- **Failure mode targeted:** Vision resolution bottlenecks, attention dilution, positional decay, and decoder context exhaustion causing bottom-of-page degradation or silent system omission.
- **Test cases:**
  1. *Hyper-Dense Full-Orchestra Pages:* 24–32 staves/system, 2,500–4,500+ musical symbols/page, multi-measure rests, transposing instruments, dense dynamics/articulations.
  2. *Virtuosic High-Density Solo Pages:* 20+ systems or 64th/128th-note cadenzas with 2,000+ symbols/page.
- **Pass criteria:**
  - **Positional Decay Ratio (`Q4 / Q1 F1`) `>= 0.95`** (bottom 25% of page stays within 5% of top 25%).
  - **System-Count Completeness = `100%`**.
  - **Cross-Staff Vertical Alignment Error `<= 1%`**.

---

## Pillar 2: Optical & Structural Ambiguity Traps

### 2A. Dense Graphical Collisions & Extreme Voice Packing
- **Test cases:** Tight cluster chords with displaced seconds and staggered double-sharp/double-flat columns; 5–8 ledger-line stacks; 3–4 independent voices on one staff with voice-crossing and shared unison noteheads; graphic overlap storms (slurs, ties, cross-staff beams, lyrics, figured bass, fingerings, articulation stacks).
- **Pass criteria:** Cluster Chord Exact-Pitch-Set Match `>= 93%`; Ledger-Line Pitch Accuracy (`>= 4` ledger lines) `>= 95%`; Voice-Assignment & Cross-Staff Beat Integrity `>= 92%`.

### 2B. Semantic & Structural Edge Cases
- **Test cases:** Small-scale cue notes vs. main notes; temporary *ossia* staves; mid-bar clef/key/time changes; cautionary vs. active accidentals; invisible/dashed/*mensurstrich* barlines and unmeasured cadenzas; double-sided verso bleed-through (15–35% ink contrast).
- **Pass criteria:** Cue/Main Role Separation F1 `>= 92%`; Mid-Bar Clef/Key State-Propagation Accuracy `>= 96%`; Bleed-Through False-Positive Rate `<= 1%`.

### 2C. Mixed-Domain & Heavily Annotated Score Traps
- **Test cases:** Printed scores overlaid with pencil fingerings, bowings, circled accidentals, breathing commas, dynamic underlines, translucent yellow/pink highlighter bands, crossed-out bars (`X` / `VI-DE`), taped-in handwritten patch measures, and marginal sticky-notes.
- **Pass criteria (Dual-Layer Evaluation):**
  - **Printed-Text Invariance (`Clean-vs-Annotated ΔF1`) `<= 3%`**.
  - **Annotation False-Note Rate `<= 0.5%`**.
  - *(Optional Tier-2)* **Explicit Annotation Separation F1 `>= 85%`**.

---

## Pillar 3: Robustness, Invariance & Negative Controls

### 3A. Paired Counterfactuals (Exact-Score Domain-Drop Matrix)
50–100 diverse scores rendered across 5 paired views sharing identical ground truth: `clean_digital`, `archival_scan`, `phone_photo_mild`, `phone_photo_extreme`, and `handwritten_copy`.
- **Target domain-drop gates ($\Delta_{\text{domain}} = F1_{\text{clean}} - F1_{\text{target}}$):** $\Delta_{\text{scan}} \le 2\%$, $\Delta_{\text{phone\_mild}} \le 3\%$, $\Delta_{\text{phone\_extreme}} \le 6\%$, $\Delta_{\text{handwritten}} \le 8\%$.

### 3B. Out-of-Distribution (OOD) & Negative Controls
- **Non-Music Rejection Set:** Title pages, tables of contents, text-only program notes/libretti, copyright pages, blank manuscript staff paper (empty staves, zero notes), non-music graphics/tables.
- **Orientation & Framing Traps:** Scores rotated `90°`, `180°`, `270°`; two-page open-book spreads across a curved center gutter; partial-page crops.
- **Pass criteria:**
  - **Non-Music Hallucination Rate = `0%`** (`FP Note Count = 0`).
  - **Orientation Invariance:** $F1(90^\circ/180^\circ/270^\circ) \ge F1(0^\circ) - 1\%$.
  - **Two-Page Spread Ordering Accuracy = `100%`** (never merge left-page system 1 with right-page system 1).

### 3C. Multi-Page Continuity & Long-Horizon Score Coherence
2–4 consecutive pages testing cross-page ties/slurs/hairpins/`8va`/`Ped.`, implicit instrument-name & transposition carryover from page 1, French-score staff hiding/reappearing across pages, and cross-page measure numbering / `D.S. al Coda` navigation.
- **Pass criteria:**
  - **Cross-Page Spanner Closure F1 `>= 95%`**.
  - **French-Score Staff-Identity Tracking Accuracy `>= 98%`**.
  - **Zero pitch-transposition or octave-register jumps at page turns**.

---

## Pillar 4: Private Parametric Trap Generator & Longevity Rules

1. **Trap-Conditional Scoring (No Dilution):** Every leaderboard entry reports the Worst-Group & Trap-Site Scorecard (`PHR`, `TSR`, `Q4/Q1`, `Non-Music FP`, `Cross-Page Spanner F1`, `French-Score Staff Tracking`).
2. **Private Parametric Trap Generator:** A held-out seeded generator mints fresh, unseen *Counterfactual Classics*, *Ostinato Spot-the-Diff*, and *Atonal/Asymmetric Meter* instances every evaluation cycle, making training-set memorization mathematically useless.
3. **Human-Effort Tail Gate:** Report **P90 and P99 `wFPP`** across stress suites alongside median `wFPP`.
4. **Hidden `Adversarial-Max` Split:** Compound traps (e.g., phone photo of a pencil-annotated, bleed-through, 24-staff French-score page with a mid-bar clef change and counterfactual accidental) serve as the long-horizon ceiling.
