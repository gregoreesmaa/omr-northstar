# 02 — Musicological & Notational Taxonomy

To guarantee both **zero-regression mastery of everyday music** and an **anti-saturation ceiling that remains hard even for frontier models**, `OMR-NS` defines **15 fine-grained musicological repertoire strata (`T1.1`–`T3.7`)** organized into three repertoire bands, mapped explicitly onto the **6 Musical Complexity Tiers (`T0`–`T5`)** of the **720-page core corpus** (~360 distinct works at 1–3 pages/work), plus mandatory phenomenon-level edge-case quotas.

> **Axis Clarity:** Repertoire Stratum IDs (`T1.1`–`T3.7`, recorded in `page_metadata.schema.json` under `repertoire_stratum`) classify *musical genre/instrumentation*. Musical Complexity Tiers (`T0`–`T5`, recorded under `complexity_tier`) classify *notational/engraving difficulty* orthogonally to Visual Domain (`CDR`/`ALS`/`RWP`/`HWM`) and Visual Severity (`V1`–`V4`).

---

## 0. Repertoire Stratum (`T1.1`–`T3.7`) ↔ Complexity Tier (`T0`–`T5`) Mapping

| Repertoire Band | Stratum ID | Category | Repertoire Share | Target Complexity Tiers (`T0`–`T5`) |
|---|---|---|---:|---|
| **Band 1: Basics & Core (25%)** | `T1.1` | Monophonic Melodies & Solo Studies | 7.0% | `T0` (basic), `T1` (intermediate) |
| | `T1.2` | Simple Piano (Beginner / Early-Intermediate) | 7.0% | `T1` |
| | `T1.3` | Hymnal & Simple SATB Chorales | 6.0% | `T1` (homophonic), `T2` (4-voice split) |
| | `T1.4` | Pop / Folk Lead Sheets | 5.0% | `T0` (simple), `T1` (with chords/lyrics) |
| **Band 2: Standard Classical & Modern (35%)** | `T2.1` | Solo Piano (Classical & Early Romantic) | 11.0% | `T2`, `T4` (virtuosic) |
| | `T2.2` | Chamber Music & String Quartets | 9.0% | `T2` |
| | `T2.3` | Lieder, Art Song, Choral & Keyboard/Organ | 8.0% | `T2` |
| | `T2.4` | Concert Band & Small Ensemble (5–12 Staves) | 7.0% | `T2` (5–8 staves), `T3` (9–12 staves) |
| **Band 3: Extreme / Grandmaster (40%)** | `T3.1` | Dense Romantic & Impressionist Piano & Harp | 7.0% | `T4`, `T5` (compound/autograph) |
| | `T3.2` | Full Orchestral Scores (16–30+ Staves) | 7.0% | `T3` (16–24 staves), `T5` (28–32+ miniature/adversarial) |
| | `T3.3` | Baroque & Early Music (Common-Practice) | 5.0% | `T4`, `T5` (historical plates/autographs) |
| | `T3.4` | Jazz, Big Band, Chord Grids & Nashville Charts | 5.0% | `T3` (Big Band), `T4` (complex charts/NNS) |
| | `T3.5` | Guitar TAB, Fretted Instruments & Fretboards | 5.0% | `T4`, `T5` |
| | `T3.6` | Unpitched Percussion & Drumset | 5.0% | `T4`, `T5` |
| | `T3.7` | Contemporary & Extended Techniques (20th–21st c.) | 6.0% | `T4`, `T5` |

---

## 1. Band 1 — Basics & Core Repertoire (`T1.1`–`T1.4`, 25% of Repertoire)

*Role:* Regression floor (`T0`–`T1`). Production OMR systems must achieve near-flawless engraving fidelity here (`>= 98%` Strict Note & Symbol F1); errors in this band represent fundamental reading failures.

| Stratum ID | Category | Share | Mandatory Notational Scope |
|---|---|---:|---|
| **T1.1** | **Monophonic Melodies & Solo Studies** | 7.0% | Single staff; folk tunes, sight-singing exercises, beginner/intermediate woodwind/brass/string etudes. Treble & bass clefs; `0–4` sharps/flats; simple & compound meters (`2/4`, `3/4`, `4/4`, `C`, `2/2`, `3/8`, `6/8`); dotted rhythms, simple triplets, basic ties, slurs, dynamics (`pp`–`ff`, hairpins), basic articulations (staccato, accent, tenuto, fermata). |
| **T1.2** | **Simple Piano (Beginner / Early-Intermediate)** | 7.0% | 2-staff grand staff (treble + bass, occasional both-treble or both-bass); 1 voice per staff; simple 2–3 note block chords, broken chords, fingerings (`1`–`5`), basic phrase slurs, simple repeat signs (`||:` `:||`), `Fine`, basic tempo/character text (*Allegro*, *Andante*). |
| **T1.3** | **Hymnal & Simple SATB Chorales** | 6.0% | Split 50/50 between **closed-score hymnal** (2 staves, 2 voices/staff with opposing stems, shared/split rhythms, 1–2 verses between staves) and **open-score 4-staff SATB** (treble, treble, treble-8vb tenor, bass; 1 verse). Tests multi-voice stem separation, homophonic alignment, and basic syllabic lyrics. |
| **T1.4** | **Pop / Folk Lead Sheets** | 5.0% | Single vocal melody staff + chord symbols above staff (major, minor, dom7, maj7, sus4, slash-bass `G/B`) + 1–2 lyric lines below + simple 1st/2nd endings, double bars, and basic section labels (`Verse`, `Chorus`, `Bridge`). |

---

## 2. Band 2 — Intermediate & Standard Classical/Modern Repertoire (`T2.1`–`T2.4`, 35% of Repertoire)

*Role:* Primary real-world benchmark for working musicians, students, arrangers, and archivists (`T2`–`T3`; production utility target `>= 92%` F1 across all four visual domains).

| Stratum ID | Category | Share | Mandatory Notational Scope |
|---|---|---:|---|
| **T2.1** | **Solo Piano (Classical & Early Romantic)** | 11.0% | Haydn, Mozart, Beethoven, Schubert, Chopin, Schumann, Bach fugues/inventions. 2–4 contrapuntal voices across 2 staves; **cross-staff beaming and cross-staff chords**; full pedaling (`Ped.`/`*` and continuous bracketed pedal lines with half-pedal/retake notches); Alberti/arpeggiated accompaniments; grace notes (acciaccatura & appoggiatura); standard ornaments (trills with accidental above, turns, mordents); extreme key signatures (5–7 sharps/flats); `8va`/`8vb` passages; dense dynamics/hairpins/expressive text. |
| **T2.2** | **Chamber Music & String Quartets** | 9.0% | 3–5 staves (string trios/quartets/quintets, piano trios, wind quintets). **Mandatory alto clef (viola) and tenor clef (cello/bassoon high register)**; double/triple/quadruple stops; string bowings (up/down-bow, `sul ponticello`, `pizz.`/`arco`, natural/artificial harmonics, mute `con sord.`); synchronized multi-staff measure alignment, rehearsal letters/numbers, occasional transposing horn/clarinet part. |
| **T2.3** | **Lieder, Art Song, Choral & Organ/Accordion Keyboard** | 8.0% | Voice + piano (3 staves), SATB + piano/organ reduction (4–6 staves), **3-staff organ pedalboard scores** (Great/Swell manuals + dedicated Pedal staff with heel/toe `^`/`u`/`o` pedaling marks and registration stops), and **accordion** (Stradella/free-bass `M`/`m`/`7`/`d` bass buttons, register switches, bellows shake). **Multi-verse stacked lyrics (2–5 stanzas)**, hyphenated syllabification, long **melisma extender lines**, elisions/elision slurs, verse numbers (`1.`, `2.`), **multi-script lyrics** (Latin with diacritics plus **Cyrillic, CJK, Greek, and right-to-left Hebrew**), vocal breath marks, recitative/free-meter passages, and choral `divisi`. |
| **T2.4** | **Concert Band & Small Ensemble (5–12 Staves)** | 7.0% | Mixed woodwind, brass, and percussion scores (5–12 staves). **Transposing instruments** (B♭ Clarinet/Trumpet, E♭ Alto Sax, F Horn, B♭ Tenor Sax/Euphonium) with divergent key signatures on the same system; **condensed dual-part staves** (`Fl. 1, 2`, `Cl. 1, 2` with `a2`, `1.`, `2.`, opposing stems); cue notes (small-size cue passages with instrument label); rehearsal marks; multi-measure rests in individual parts alongside conductor scores. |

---

## 3. Band 3 — Extreme / Grandmaster Notation (`T3.1`–`T3.7`, 40% of Repertoire)

*Role:* Deliberate anti-saturation ceiling (`T3`–`T5`) representing the hardest engraved and handwritten scores in music history.

| Stratum ID | Category | Share | Mandatory Notational Scope |
|---|---|---:|---|
| **T3.1** | **Dense Romantic & Impressionist Piano & Harp** | 7.0% | Liszt, Rachmaninoff, Brahms, Debussy, Ravel, Scriabin, Alkan, Godowsky, plus solo/orchestral harp. **3-staff and 4-staff piano layouts**; **harp pedaling diagrams** (Salzedo/standard 7-pedal crossbar `D C B \| E F G A` with flat/natural/sharp notches, text pedal changes `F#`/`Eb`, *bisbigliando* & glissando notation); extreme **polyrhythms** (`3:2`, `5:4`, `7:6`, `11:8`, `13:12` across hands/voices); **nested tuplets** (e.g., triplet inside a quintuplet); **grace-note cascades & unmeasured small-note cadenzas** (10–40 beamed small notes spanning bars/staves); 5+ ledger lines; dense chord clusters & interlocking hand brackets (`m.d.`/`m.s.`, L.H./R.H. hooks); simultaneous overlapping voices with shared noteheads. |
| **T3.2** | **Full Orchestral Scores (16–30+ Staves)** | 7.0% | Late Romantic, Impressionist, & 20th-century orchestral scores (Mahler, Strauss, Stravinsky, Ravel, Prokofiev). **16–30+ staves per page** (tiny staff height, tight vertical spacing, heavy bracket/brace hierarchy); mixed transposing instruments + C-score comparisons; complex **string divisi** (`div. a 3`, `div. a 4`, split staves mid-page, `gli altri`, `unis.`); **mid-piece instrument switches** (`muta in Picc.`, `take Eng. Hn.`, `to Bass Cl.`); cue notes; **French/cutaway orchestral layouts** (empty staves hidden on select systems, changing staff count from system to system); brass stopped/muted notation (`+`, `o`, `con sord.`). |
| **T3.3** | **Baroque & Early Music (Common-Practice)** | 5.0% | Bach, Handel, Couperin, Rameau, Monteverdi, Corelli (Urtext editions & historical engravings/manuscripts). **Figured bass / basso continuo** (stacked 1–4 vertical figures, slashed numerals `6/`, `5/`, standalone accidentals, continuation lines `—`, `tasto solo`); **Baroque ornament tables** (French & German symbols: pincé, tremblement, coulé, Schleifer, compound turn-trills, arpeggiation brackets); **early C-clefs** (soprano C1, mezzo-soprano C2, baritone F3/C5, French violin G1); alla-breve & white-note common-practice conventions (`3/1`, `3/2`, `6/4`, `9/8`, `12/8`, breves, longa). *(See §3.1 for explicit `v1.0` boundary on pre-1600 mensural & Gregorian/neume notation.)* |
| **T3.4** | **Jazz, Big Band, Chord Grids & Nashville Charts** | 5.0% | Real Book lead sheets, studio rhythm-section charts, Big Band scores/parts, **chord-grid sheets**, and **Nashville Number System (NNS) charts** (Arabic/Roman scale-degree numerals `1`, `4`, `5m`, `6-`, split-bar underlines, diamond whole-note holds, rhythmic pushes `^`/`<`, modulation arrows). **Slash notation** (beat slashes without stems + **rhythmic slash notation with stems/beams/ties/rests** for comping & ensemble kicks); **complex jazz harmony** (polychords `D/C`, altered extensions `G7(b9,#11,b13)`, `ø7`, `Δ7`, `°7`, parenthesized substitutions, `N.C.`); **complex repeat roadmaps** (`D.S. al Coda`, `D.S.S. al Doppia Coda`, `D.C. al Fine`, 1st/2nd/3rd/4th volta endings, `To Coda ⊕`, `Play 3x`, vamp/repeat-bar `%` and 2-bar `%%` similes); jazz articulations (falls, doits, scoops, plops, shakes, turns, ghost notes in parentheses). |
| **T3.5** | **Guitar TAB, Fretted Instruments & Fretboards** | 5.0% | Classical, fingerstyle, rock/metal, jazz guitar, and lute/ukulele. **Coupled standard notation + 6-string TAB** (plus 4/5-string bass TAB and 7-string/dropped tunings with explicit tuning headers); **fretboard chord diagrams** (barre arcs, muted `x`, open `o`, fret offset `5fr`); TAB-specific techniques: bends (pre-bend, release, `full`, `1/2`, `1/4`), slides (legato/shift), hammer-on/pull-off (`H`/`P`), vibrato waves, palm mute (`P.M.----`), let-ring (`let ring----`), tapping (`T`), natural/pinch harmonics (`<12>`, diamond noteheads), rhythmic stems & beams attached to TAB numerals, capo indications. |
| **T3.6** | **Unpitched Percussion & Drumset** | 5.0% | Rock/jazz/prog drumset transcriptions, marching snare/tenor, and multi-percussion orchestral/contemporary solos. **1-line, 3-line, and 5-line neutral-clef staves**; **extended notehead taxonomy** (X-head, circled-X, triangle, diamond, slash, square, ghost-parenthesized); hi-hat open/closed (`o`, `+`), choke/dampen, rimshot, cross-stick; **sticking** (`R`/`L`), **rudimental ornaments** (flams, drags, ruffs, buzz/press rolls `z` on stem, diddles); mid-staff instrument-change labels & pictorial beater/mallet icons. |
| **T3.7** | **Contemporary & Extended Techniques (20th–21st c.)** | 6.0% | Bartók, Ligeti, Crumb, Berio, Boulez, Penderecki, Ferneyhough, Grisey, Saariaho. **Microtones** (quarter-tone sharps/flats, three-quarter-tone, eighth-tone arrows, Helmholtz-Ellis / Sagittal / Stein-Zimmermann accidentals); **unmetered / *senza misura* / proportional spatial notation** (no time signature, dashed/dotted barlines, durational beams/lines, seconds timestamps `5"`–`8"`); **feathered beaming** (accelerando/ritardando converging/diverging secondary beams); **tone clusters** (solid black/white cluster bars, chromatic/diatonic/black-key cluster brackets); **ossia staves** (small-scale floating alternate staves); **cutaway staves**, **aleatoric box-repeats**, Bartók snap-pizzicato, multiphonic fingerings, Sprechstimme (`x` on stems/heads), complex irrational meters (`5/8`, `7/16`, `4/6`, `3/10`, metric modulation `♩. = ♩`). |

### 3.1 Explicit `v1.0` Scope Boundary: Mensural & Gregorian / Neume Notation

To keep `v1.0` CER alignment and MusicXML 4.0 / MEI evaluation strictly well-defined:
- **In-Scope (`v1.0` Core `NS`):** Standard 5-line Common Western Notation (including Baroque C-clefs, figured bass, breves/longas, and white-note `3/1`–`3/2` meters in `T3.3`), 1/3/5-line unpitched percussion (`T3.6`), 4–7-string fretted TAB (`T3.5`), chord-grid & Nashville Number sheets (`T3.4`), 3-staff organ/accordion (`T2.3`), and harp pedaling diagrams (`T3.1`).
- **Out-of-Scope for `v1.0` Core `NS` (Scoped as Optional `T5` / `v2.0` Early-Music Extension):** Pre-1600 **white/black mensural notation** (mensural ligatures, coloration, perfection/imperfection, *prolatio* mensuration signs without barlines) and **4-line Gregorian / square neume chant notation** (*virga*, *punctum*, *podatus*, *clivis*, *torculus*, *climacus*, *liquescent* neumes). These require a dedicated MEI-native neume/mensural CER evaluator rather than common-practice measure alignment and will be introduced as an optional `v2.0` Early-Music Extension track.

---

## 4. Mandatory Edge-Case Phenomenon Quotas

Every phenomenon below is tagged in ground-truth metadata (`notational_tags`), must appear across **both printed (`CDR`/`ALS`) AND camera/manuscript (`RWP`/`HWM`) domains**, and is scored as an explicit phenomenon-level recall/precision slice:

### 4.1 Mid-Measure & Boundary Structural Changes
- **Mid-measure clef changes** (`>= 35` instances across `>= 20` pieces): exact temporal attachment verified *before* target note (including grace notes preceding a clef change).
- **Mid-measure key signature changes** (`>= 15` instances across `>= 10` pieces): including natural-cancellation naturals and transposing multi-staff scores.
- **Mid-measure & frequent meter changes** (`>= 30` instances across `>= 18` pieces): alternating/additive/composite meters and simultaneous polymeter across staves.
- **Cautionary end-of-system changes** (`>= 30` instances): preview clef/key/time after final barline of a system (must NOT be double-counted).

### 4.2 Bar-Length & Metric Anomalies
- **Pickup (anacrusis) bars** (`>= 40` pieces): opening incomplete bar + matching shortened final/internal bar.
- **Mid-piece split / incomplete bars** (`>= 20` instances): bars split across repeat signs or system breaks; irregular cadenza/recitative bars.
- **Multi-measure rests** (`>= 45` instances across `>= 20` pieces): modern numbered H-bar rests (`4`, `12`, `37`) and archival church-rest ladders, including interruptions by rehearsal marks or key/time changes.

### 4.3 Cross-Boundary Spanners & Ties/Slurs
- **Ties across system breaks and page turns** (`>= 40` instances): including accidental persistence across barlines and courtesy accidentals after system breaks.
- **Slurs across system breaks & cross-staff slurs** (`>= 40` instances): S-shaped cross-staff slurs, nested phrasing/articulation slurs, and *laissez vibrer* (`l.v.`) open ties.
- **Ottava spanners (`8va`, `8vb`, `15ma`, `15mb`, `loco`)** (`>= 35` instances across `>= 20` pieces, `>= 12` crossing system breaks): verified for sounding/written pitch consistency.
- **Pedal spanners & hairpins across system breaks** (`>= 30` instances).

### 4.4 Accidentals & Pitch-Spelling Edge Cases
- **Double sharps (`𝄪`) and double flats (`𝄫`)** (`>= 40` instances across `>= 20` pieces): including 19th-century cancellation forms (`♮♯`, `♮♭`) vs modern bare cancellation.
- **Courtesy / cautionary accidentals** (`>= 50` instances): parenthesized and unparenthesized reminders.
- **Accidental scoping & collision stacking** (`>= 30` dense chords): staggered columns in 4–6 note seconds/clusters, voice-local vs staff-local persistence, and microtonal accidentals.

### 4.5 Rhythmic, Instrumental & Textual Extensions
- **Tremolos** (`>= 30` instances across `>= 18` pieces): single-note measured/unmeasured AND two-note fingered tremolos (visual duration doubled relative to written noteheads).
- **Arpeggios** (`>= 30` instances across `>= 18` pieces): standard, arrowed, cross-staff, and `non-arpeggiato` brackets.
- **Repeat & measure-simile shorthands** (`>= 25` instances): 1-bar (`%`), 2-bar (`%%`), beat-repeat slashes, `colla parte`/`simile`, and nested `D.S.`/Coda jump roadmaps.
- **Specialized Keyboard, Harp & Commercial Notation** (`>= 12` pages each): 3-staff organ pedalboard + accordion Stradella/free-bass notation; harp 7-pedal crossbar diagrams & pedal changes; chord-grid / Nashville Number sheets.
- **Multi-Script Lyrics** (`>= 16` pages across `>= 4` non-Latin scripts): Cyrillic, CJK (horizontal + vertical/furigana), Greek, and right-to-left Hebrew syllabic alignment with Unicode normalization (NFC).

---

## 5. Engraving-House, Font & Manuscript Diversity Cap

No single engraving engine, publisher house, or music font may exceed **`15%`** of the printed/clean portion of the benchmark:

- **Historical & Classical Publishing Houses (`ALS` & `RWP`):** Breitkopf & Härtel, Edition Peters, G. Henle Verlag, Durand / Editions Salabert, B. Schott's Söhne / Universal Edition, Ricordi, Boosey & Hawkes / Hal Leonard / Sher Music, Dover / Kalmus reprints & 18th/19th-century copper/pewter plates.
- **Modern Digital Engravers & SMuFL/Legacy Fonts (`CDR` & `RWP`):** Dorico (*Bravura*, *Petaluma*), LilyPond (*Emmentaler*, *Gonville*), MuseScore (*Leland*, *Edwin*), Finale (*Maestro*, *Jazz*), Sibelius (*Opus*, *Helsinki*), Verovio (*Leipzig*), plus serif/sans-serif/italic/Fraktur and non-Latin text/lyric fonts.
- **Handwritten Manuscript Spectrum (`HWM`):** Balanced across neat archival copyist manuscripts (`HWM-a`, 30%), authentic composer fair copies & modern commissioned sheets (`HWM-b`, 40%), and high-difficulty working autographs, pencil sketches & heavily annotated performer scores (`HWM-c` + `HWM-Hybrid`, 30%).
