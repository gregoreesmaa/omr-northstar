# 03 — Visual, Imaging, Capture & Manuscript Domains

## 1. Design Principles & Quantitative Physical Grounding

To measure true universal reading ability—and resist saturation for years—**Visual Severity (`V1`–`V4`)** and **Visual Domain (`CDR`, `ALS`, `RWP`, `HWM`)** are modeled strictly orthogonally to **Musical Complexity Tier (`T0`–`T5`)**.

### Core Principles

1. **Quantitative Physical Grounding:** Every Visual Severity level (`V1`–`V4`) is defined by measurable optical, geometric, photometric, and morphological parameters—never vague labels alone:
   - Median staff-space height $s$ (pixels), plus min/max across cue/ossia staves and perspective gradients.
   - Keystone tilt angles $(\theta_x, \theta_y)$ and in-plane rotation $\phi$ (degrees).
   - Effective point-spread-function (PSF) blur $\sigma_{\text{px}}$ (pixels).
   - Local Weber contrast $C_w$.
   - Verso bleed-through opacity $\alpha$ (%).
   - Dynamic-range illumination gradient in exposure values (EV).
   - Stroke-width coefficient of variation $CV_w$ (for handwriting).
2. **Human-Legibility Floor (`>= 98%`):** Even **`V4` Brutal (`B`)** samples must remain readable by a trained musician/engraver. Inclusion requires `>= 2` independent human experts to transcribe **`>= 98%`** of symbols consistently (with explicit `ambiguous` spans allowed only for genuinely ambiguous historical composer drafts). Physically destroyed, completely blown-out, or sub-Nyquist unreadable pages are excluded.
3. **4 Visual Domains × 4 Visual Severity Levels = 16 Primary Visual Cells:**
   - **`V1` — Easy (schema code `E`):** Sanity and upper-bound baseline, still diverse across fonts/media to expose single-pipeline overfitting.
   - **`V2` — Medium (schema code `M`):** Everyday real-world production distribution (typical library scan, casual desk phone snap, clean copyist/student sheet).
   - **`V3` — Hard (schema code `H`):** Demanding real-world conditions humans read fluently (dim stage music stand, aged 18th-c. plate with show-through, composer fair copy with edits).
   - **`V4` — Brutal (schema code `B`):** Frontier stress tests (oblique glare-hit plastic sleeve over a 4× photocopy, chaotic composer draft with cross-outs and iron-gall bleed-through, 1-bit microfilm of movable-type print).
4. **Canonical Per-Image Metadata Schema:** Every page ships with structured metadata validated against [`schemas/page_metadata.schema.json`](../schemas/page_metadata.schema.json) (`visual_tier ∈ {"E", "M", "H", "B"}` corresponding 1-to-1 to `V1`–`V4`).

---

## 2. Domain 1: Clean Digital Renders (`CDR`)

Tests pure visual-symbolic generalization across engraving aesthetics, resolution scales, layout densities, and digital rasterization pipelines—with zero physical-world optics noise.

### Visual Severity Levels (`CDR`)

| Severity Level | Resolution & Staff Space ($s$) | Fonts & Engravers | Layout & Density | Rasterization & Digital State |
|---|---|---|---|---|
| **`V1` Easy (`E`)** | 300 DPI; $s \in [12, 24]\text{ px}$ | Standard classical SMuFL (Bravura, Leland, Emmentaler/Feta, Maestro) | 1–8 staves/page, generous whitespace, standard portrait | Lossless 24-bit/8-bit PNG, crisp anti-aliasing, pure white bg |
| **`V2` Medium (`M`)** | 150–300 DPI; $s \in [8, 12]\text{ px}$ or $[24, 36]\text{ px}$ | Broad mix including legacy Finale/Sibelius + clean jazz fonts (MuseJazz, Petaluma) | 8–16 staves/page (chamber, choral+piano, dense piano), cues/ossias | Standard PDF-to-PNG rasterization, JPEG Q=75–90, mild cream/off-white bg |
| **`V3` Hard (`H`)** | 96–150 DPI; $s \in [5.5, 8]\text{ px}$ | Diverse/unusual fonts (Petrucci, Inkpen2, Broadway, historical C-clef fonts), non-Latin lyrics (Cyrillic, CJK, Greek, Hebrew) | 16–28 staves/page (orchestral/band), landscape/oblong, tight vertical packing & legal collisions | 1-bit bi-level raster (occasional 1px staff-line aliasing), JPEG Q=45–70, dark-mode/inverted or colored tablet themes |
| **`V4` Brutal (`B`)** | Ultra-low-res / pocket-score renders; $s \in [3.5, 5.5]\text{ px}$ (cues down to $2.2\text{ px}$) | Rare/unseen music fonts + mixed fonts on same page + heavy handwritten-style digital fonts | 28–40+ staves/page miniature orchestral scores, extreme vertical density, heavy cross-staff/polyrhythmic overlaps | Aggressive 1-bit dithering/aliasing with partial staff-line dropout, heavy JPEG/WebP (Q=20–40) mosquito ringing, semi-transparent diagonal watermark overlays |

---

## 3. Domain 2: Archival & Library Scans (`ALS`)

Tests resilience to 17th–20th century printing technologies (movable music type, intaglio copper/pewter plates, lithography, offset, mimeograph/spirit-duplicator/diazo), centuries of paper/ink aging, and real library digitization pipelines.

### Visual Severity Levels (`ALS`)

| Severity Level | Print Era & Ink Quality | Paper & Bleed-Through | Scanner & Geometry | Artifacts & Binarization |
|---|---|---|---|---|
| **`V1` Easy (`E`)** | Clean 20th-c. offset/engraved plates; crisp uniform ink ($C_w > 0.70$) | Light cream/off-white paper; bleed-through $\alpha < 5\%$ | Flatbed 300–600 DPI grayscale/color; skew $< 1.5^\circ$; flat page | Clean margins; at most 1 non-overlapping library stamp in margin |
| **`V2` Medium (`M`)** | 19th-c. engraved plates or lithography; mild uneven inking | Moderate yellowing, light foxing, bleed-through $\alpha \in [5\%, 15\%]$ | Overhead book cradle; mild gutter shadow & slight inner curvature; skew $1.5^\circ\text{–}4^\circ$ | Mild library stamps/catalog numbers, minor edge wear, standard PDF JPEG2000/JBIG2 |
| **`V3` Hard (`H`)** | 18th-c. intaglio / movable type (segmented staves) or faded diazo/mimeograph; ink starvation or over-inking | Heavy browning/mottling, water tide-lines, prominent verso bleed-through $\alpha \in [15\%, 30\%]$ | Strong gutter shadow + spine warp, double-page spreads, skew $4^\circ\text{–}8^\circ$, tight/chipped margins | 1-bit library/microfilm scans with moderate speckle & broken thin lines; stamps/stickers partially overlapping notation |
| **`V4` Brutal (`B`)** | Severely worn plates, broken movable type, or faint blue/purple spirit-duplicator copies ($C_w < 0.25$) | Severe iron-gall/ink bleed-through $\alpha \in [30\%, 45\%]$ competing with recto contrast; tears, holes, dark brown foxed rag paper | Harsh 1-bit microfilm/microfiche (heavy speckle, vanished staff segments, filled-in white noteheads) OR severe spine warp + deep gutter shadow + skew $> 8^\circ$ | Heavy dark library stamps directly across dense notation, taped tears, extreme JBIG2/CCITT binarization collapse |

---

## 4. Domain 3: Real-World Phone & Camera Photos (`RWP`)

Tests end-to-end visual robustness under uncontrolled consumer smartphone capture in practice rooms, concert halls, rehearsal studios, and desks. **100% real camera captures** (zero synthetic camera filters).

### Visual Severity Levels (`RWP`)

| Severity Level | Geometry & Framing | Illumination & Photometry | Optics, Blur & Codec | Physical Paper & Clutter |
|---|---|---|---|---|
| **`V1` Easy (`E`)** | Handheld overhead desk; keystone $\theta < 8^\circ$; flat sheet; clean desk border | Even diffuse daylight/indoor light ($< 0.5$ EV gradient); no glare or shadows | Sharp focus ($\sigma < 0.8\text{ px}$); full phone res ($s \ge 12\text{ px}$); clean JPEG/HEIC | Crisp printed sheet; no folds, clips, or occlusions |
| **`V2` Medium (`M`)** | Casual desk or stand shot; $\theta \in [8^\circ, 18^\circ]$; mild page curl; small stand/desk clutter | Mild directional gradient ($0.5\text{–}1.5$ EV), soft peripheral shadow, slight warm/cool WB cast | Minor corner softness ($\sigma \in [0.8, 1.5]\text{ px}$); standard JPEG Q=65–85; $s \in [8, 12]\text{ px}$ | Light fold crease, 1–2× photocopy, binder clips or stand lip in margins |
| **`V3` Hard (`H`)** | Oblique stand shot or open bound book; $\theta \in [18^\circ, 32^\circ]$; pronounced S-curve/gutter warp; tight crop or cluttered stand + fingers at edge | Strong stage/stand-lamp gradient ($1.5\text{–}3$ EV), sharp cast shadow across staves, moderate glare on glossy paper, low-light ISO noise | Noticeable tilt-shift DoF blur or slight handshake ($\sigma \in [1.5, 2.5]\text{ px}$); messaging-app resize ($s \in [6, 8]\text{ px}$); JPEG Q=45–65 | Deep tri-fold creases, 3×–4× photocopy with toner streaks, coffee ring, spiral binding, plastic sleeve reflections |
| **`V4` Brutal (`B`)** | Extreme oblique stand shot ($\theta \in [32^\circ, 45^\circ]$) + severe open-book curvature + crumpled/wavy page; fingers/pegs/clips intruding into outer notation; perspective scale gradient $> 2.2\times$ | Brutal 3–5 EV spotlight falloff, intense specular hotspot on plastic sleeve washing out local contrast, heavy dark-stage high-ISO noise, mixed-temp shadows | Significant motion blur or strong tilt-shift defocus ($\sigma \in [2.5, 4.0]\text{ px}$), lens distortion, heavy messaging-app compression (Q=25–45) with $s_{\min} \in [4, 6]\text{ px}$ | 5× degraded photocopy, heavy creases/tears/tape glare, dense coffee/water damage, cluttered multi-page stand overlap |

---

## 5. Domain 4: Handwritten Manuscripts — Historical & Modern (`HWM`)

Tests calligraphic generalization, non-rigid hand-drawn/rastrum staff geometry, noisy ink/pencil media, editorial corrections (cross-outs, paste-overs/*collettes*), and printed+handwritten hybrid scores (`HWM-Hybrid`).

### Visual Severity Levels (`HWM`)

| Severity Level | Calligraphic Clarity | Staff & Substrate | Ink / Medium & Contrast | Edits, Lyrics & Clutter |
|---|---|---|---|---|
| **`V1` Easy (`E`)** | Neat professional copyist (`HWM-a`, classical or jazz) or very tidy modern manuscript (`HWM-b`); well-separated notes & clear pitch alignment | Crisp pre-printed manuscript paper; straight parallel staves; clean scan | High-contrast dark black/blue ink ($C_w > 0.65$); consistent stroke width | Minimal/no cross-outs; clean printed-style or neat block lyrics; `HWM-Hybrid` with light sparse pencil fingerings |
| **`V2` Medium (`M`)** | Composer fair copy or typical musician/student sheet; moderate personal glyph stylization & slight slant | Slightly wavy rastrum staves or lightly aged manuscript paper | Fountain pen, ballpoint, or medium-dark pencil ($C_w \in [0.40, 0.65]$); mild ink variation or light smudges | Occasional clean cross-outs/insertions, cursive lyrics, rehearsal marks; `HWM-Hybrid` with typical orchestral bowings/fingerings |
| **`V3` Hard (`H`)** | Rapid composer working copy (`HWM-c`), messy student homework, or ornate 18th-c. calligraphic script; uneven horizontal spacing & crowded chords | Highly wavy/irregular rastrum staves, freehand staves, or lined/graph notebook paper (grid distractors) | Faint graphite pencil with erased ghost notes ($C_w \in [0.20, 0.40]$), feathered iron-gall ink with moderate bleed-through, or faded colored ink | Frequent scratch-outs, white-out/paste-overs, dense historical cursive lyrics, shorthand repeat signs; `HWM-Hybrid` with heavy pencil + highlighter + cut brackets |
| **`V4` Brutal (`B`)** | Chaotic composer draft/sketch (`HWM-c`); extreme gestural abbreviation, colliding voices, ambiguous vertical spacing requiring musical context | Broken/faint freehand staves, severe rastrum drift/overlap, dark aged paper with heavy iron-gall corrosion & verso bleed-through ($\alpha > 25\%$) | Very faint smudged pencil ($C_w < 0.20$) or blotchy corrosive iron-gall ink + mixed red/blue/pencil layers | Dense layered revisions (cross-outs over cross-outs, marginalia, arrows to insertions, *collettes*), dense historical cursive; `HWM-Hybrid` where heavy pencil/highlighter/taped ossias physically cross staff lines and noteheads |

---

## 6. Five Paired-Capture Controlled Isolation Protocols (A–E)

A single-image test set conflates *visual failure* with *notational failure*. `OMR-NS` mandates **5 paired protocols** linking identical musical ground truth via `paired_anchor_id`:

### Protocol A — Digital-to-Physical Capture Ladder (`CDR → Print → Scan → Phone`)
30 Dev / 60 Test canonical symbolic scores linked across an 8-step physical capture ladder:
1. `A0_CDR_Clean`: 300 DPI lossless digital render (notational baseline).
2. `A1_Print_Flatbed`: Laser-printed on matte 80gsm paper, 300 DPI flatbed scan.
3. `A2_Phone_Easy`: Overhead desk smartphone photo, diffuse light, flat (`V1`).
4. `A3_Phone_Stand_Med`: Music-stand handheld photo ($\theta \approx 15^\circ$), indoor light (`V2`).
5. `A4_Phone_Book_Hard`: Bound/curved open-book photo with gutter shadow + clip (`V3`).
6. `A5_Phone_Glare_Brutal`: Placed in glossy plastic sleeve under directional stand lamp + oblique $\theta \approx 35^\circ$ (`V4`).
7. `A6_Xerox4x_Phone`: Photocopied 4× (tilted, speckling, dropout) → folded → low-light smartphone photo (`V4`).
8. `A7_Messaging_Recomp`: `A5` or `A6` passed through messaging-app resize + JPEG Q=40 (`V4`).

*Diagnostic yield:* $\Delta(\text{A0} \to \text{A1})$ isolates print/scan drop; $\Delta(\text{A1} \to \text{A2..A7})$ isolates pure camera, glare, photocopy, and codec loss at zero musical confound (`Δ_photo`).

### Protocol B — Multi-Engraver & Scale Invariance Pairs (Within `CDR`)
Render 20 Dev / 40 Test canonical MusicXML/MEI scores across 3–5 distinct engraver/font stacks (Bravura, Leland, Emmentaler/Feta, Maestro, MuseJazz/Broadway), 3 staff-space scales ($s \in \{18, 9, 5\}\text{ px}$), and 2 raster modes (anti-aliased PNG vs. 1-bit bi-level TIFF). Quantifies `Δ_engraver` and minimum legible staff-space cliff $s_{\text{cliff}}$.

### Protocol C — Archival Multi-Digitization Pairs (`ALS` + `RWP`)
15–20 real historical library pages captured in 3 modalities of the exact same physical volume: `C0` (600 DPI color archival cradle scan), `C1` (legacy 1-bit bitonal/microfilm PDF scan), and `C2` (researcher handheld smartphone photo on a reading-room desk).

### Protocol D — Handwritten & Hybrid Isolation Pairs (`HWM` + `RWP`)
- **D1 (`HWM-Hybrid`):** Same printed physical score photographed/scanned (1) pristine, (2) light orchestral pencil markings, and (3) heavy pencil + highlighter + taped ossia/cut markings.
- **D2 (Calligraphic-Variance):** 3 human writers (neat copyist, average musician, rapid/messy writer) hand-copy the *same* 15 Dev / 30 Test musical excerpts (`Δ_handwriting`).
- **D3 (Manuscript Scan-vs-Phone):** Same real handwritten pages via 600 DPI color flatbed scan vs. oblique low-light music-stand phone photo.

### Protocol E — Factorized Single-Stressor Diagnostic Sweeps (Cliff Probes)
On a 10-page anchor set, vary **one physical stressor at a time** across 5 calibrated steps (all else pristine):
1. Staff-space scale $s$: `20 → 12 → 8 → 5.5 → 4 px`
2. Keystone angle $\theta$: `0° → 10° → 20° → 30° → 42°`
3. Defocus/motion blur $\sigma$: `0 → 0.8 → 1.6 → 2.6 → 3.8 px`
4. Illumination gradient: `0 → 1 → 2 → 3.5 → 5 EV`
5. Verso bleed-through opacity $\alpha$: `0% → 10% → 20% → 32% → 45%`
6. Localized occlusion area: `0% → 2% → 5% → 10% → 15%`

---

## 7. Non-Canonical Element Ground-Truth Policy

- **Must be ignored (zero false positives):** Verso bleed-through, library stamps, catalog stickers, ruled notebook grids, desk/stand clutter, fingers/clips, crossed-out/scratched-out notes, erased pencil ghosts.
- **Hybrid-score policy (`HWM-Hybrid`):** Primary score evaluates **canonical printed notation recovery**; secondary optional track evaluates **annotation extraction** (bowings, fingerings, cuts).
