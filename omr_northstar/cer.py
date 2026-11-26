"""Deterministic MusicXML -> Canonical Evaluation Representation (CER) & 2-Stage Aligner.

Zero external dependencies (Python 3.10+ standard library only).
Implements the CER data model and two-stage hierarchical alignment specified in
docs/04-metrics-and-human-utility.md:
  1. Measure-level Needleman-Wunsch DP (sim = 0.7 * Dice + 0.3 * barline_match, gap = 0.25).
  2. Optimal voice-ID bijection (exact linear sum assignment via Hungarian algorithm)
     + exact bipartite/multiset matching across T1a (Sounding), T1b (Rhythmic),
     and T1c (Strict Engraving) tiers.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from fractions import Fraction
from typing import Hashable, Sequence

from .scoring import PageMetrics, compute_effective_measure_cost

STEP_TO_SEMITONE: dict[str, int] = {
    "C": 0,
    "D": 2,
    "E": 4,
    "F": 5,
    "G": 7,
    "A": 9,
    "B": 11,
}

BAR_STYLE_MAP: dict[str, str] = {
    "regular": "regular",
    "light-light": "double",
    "light-heavy": "final",
    "heavy-light": "regular",
    "heavy-heavy": "double",
    "dashed": "dashed",
    "dotted": "dashed",
    "none": "regular",
}


def _local_tag(tag: str) -> str:
    """Strip XML namespace URI if present."""
    return tag.rsplit("}", 1)[-1]


def _find_child(elem: ET.Element, name: str) -> ET.Element | None:
    for child in elem:
        if _local_tag(child.tag) == name:
            return child
    return None


def _iter_children(elem: ET.Element, name: str) -> list[ET.Element]:
    return [child for child in elem if _local_tag(child.tag) == name]


def _child_text(elem: ET.Element, name: str, default: str = "") -> str:
    child = _find_child(elem, name)
    if child is None or child.text is None:
        return default
    return child.text.strip()


def _f1_from_counts(matched: int, n_pred: int, n_gt: int) -> float:
    if n_pred == 0 and n_gt == 0:
        return 1.0
    if n_pred == 0 or n_gt == 0:
        return 0.0
    return (2.0 * matched) / (n_pred + n_gt)


def _multiset_intersection_size(a: Counter[Hashable], b: Counter[Hashable]) -> int:
    return sum((a & b).values())


@dataclass(frozen=True)
class CERContext:
    """Active musical context state on a staff at a given onset."""

    clef: str = "G2"
    key_fifths: int = 0
    time_num: int = 4
    time_den: int = 4
    ottava: int = 0


@dataclass(frozen=True)
class CERNote:
    """Canonical note event with exact rational onset and duration (in whole notes)."""

    staff: int
    voice: str
    onset: Fraction
    duration: Fraction
    step: str
    alter: int
    octave: int
    midi: int
    tie_start: bool = False
    tie_stop: bool = False
    grace: bool = False
    tuplet: tuple[int, int] | None = None
    is_chord: bool = False
    cue: bool = False
    articulations: tuple[str, ...] = ()
    lyrics: tuple[str, ...] = ()
    slur_starts: int = 0
    slur_stops: int = 0


@dataclass(frozen=True)
class CERRest:
    """Canonical rest event with exact rational onset and duration (in whole notes)."""

    staff: int
    voice: str
    onset: Fraction
    duration: Fraction
    is_full_measure: bool = False


@dataclass(frozen=True)
class CERDirection:
    """Expressive or textual attachment anchored to a staff and rational onset."""

    kind: str
    value: str
    staff: int
    onset: Fraction
    end_onset: Fraction | None = None


@dataclass(frozen=True)
class CERMeasure:
    """Canonical representation of a single measure within a part."""

    part_id: str
    number: str
    index: int
    barline_left: str = "regular"
    barline_right: str = "regular"
    repeat_start: bool = False
    repeat_end: bool = False
    volta: str | None = None
    is_implicit: bool = False
    contexts: dict[int, CERContext] = field(default_factory=dict)
    context_changes: tuple[tuple[int, Fraction, str, str], ...] = ()
    notes: tuple[CERNote, ...] = ()
    note_contexts: tuple[CERContext, ...] = ()
    rests: tuple[CERRest, ...] = ()
    directions: tuple[CERDirection, ...] = ()


@dataclass(frozen=True)
class CERScore:
    """Canonical Evaluation Representation of a parsed MusicXML score."""

    parts: tuple[str, ...]
    measures: tuple[CERMeasure, ...]
    num_staves: int = 1
    system_breaks: tuple[int, ...] = ()

    @property
    def total_notes(self) -> int:
        return sum(len(m.notes) for m in self.measures)

    @property
    def total_measures(self) -> int:
        return len(self.measures)


@dataclass(frozen=True)
class CERAlignmentResult:
    """Full diagnostic output from 2-stage hierarchical CER alignment."""

    f1_t1a: float
    f1_t1b: float
    f1_t1c: float
    f1_rest: float
    mer: float
    f1_bar: float
    bsvr: float
    a_state: float
    s_note: float
    s_struct: float
    s_expr: float
    s_play: float
    wfpp: float
    transcribe_cost: float
    n_gt: int
    n_pred: int
    m_gt: int
    m_pred: int
    k_dropped_systems: int
    has_repetition_loop: bool
    measure_alignment: tuple[tuple[int | None, int | None], ...]


def _split_multistaff_part(
    part_id: str,
    part_measures: list[CERMeasure],
    part_max_staff: int,
    part_system_breaks: list[int],
    base_measure_offset: int,
) -> tuple[list[str], list[CERMeasure], list[int]]:
    """Canonicalize a multi-staff <part> into separate 1-staff part streams (Bucket A).

    Ensures a 1-part N-staff grand-staff encoding and an N-part 1-staff encoding
    produce identical CER measure sequences, normalized staff=1 coordinates,
    per-staff contexts, and system break indices.
    """
    if part_max_staff <= 1:
        return (
            [part_id],
            part_measures,
            [base_measure_offset + b for b in part_system_breaks],
        )

    split_parts: list[str] = []
    split_measures: list[CERMeasure] = []
    split_breaks: list[int] = []
    n_measures = len(part_measures)

    for s in range(1, part_max_staff + 1):
        s_part_id = f"{part_id}_s{s}"
        split_parts.append(s_part_id)
        s_offset = base_measure_offset + (s - 1) * n_measures
        for b in part_system_breaks:
            split_breaks.append(s_offset + b)

        for m in part_measures:
            s_notes: list[CERNote] = []
            s_note_ctxs: list[CERContext] = []
            for n, ctx in zip(m.notes, m.note_contexts):
                if n.staff == s:
                    s_notes.append(replace(n, staff=1))
                    s_note_ctxs.append(ctx)

            s_rests = tuple(replace(r, staff=1) for r in m.rests if r.staff == s)
            s_dirs = tuple(replace(d, staff=1) for d in m.directions if d.staff == s)
            s_ctx_changes = tuple(
                (1, c_onset, c_kind, c_val)
                for c_staff, c_onset, c_kind, c_val in m.context_changes
                if c_kind in ("key", "time") or c_staff == s
            )
            s_ctx = m.contexts.get(s) or next(iter(m.contexts.values()), CERContext())
            split_measures.append(
                CERMeasure(
                    part_id=s_part_id,
                    number=m.number,
                    index=m.index,
                    barline_left=m.barline_left,
                    barline_right=m.barline_right,
                    repeat_start=m.repeat_start,
                    repeat_end=m.repeat_end,
                    volta=m.volta,
                    is_implicit=m.is_implicit,
                    contexts={1: s_ctx},
                    context_changes=s_ctx_changes,
                    notes=tuple(s_notes),
                    note_contexts=tuple(s_note_ctxs),
                    rests=s_rests,
                    directions=s_dirs,
                )
            )

    return split_parts, split_measures, split_breaks


def parse_musicxml(xml_str: str) -> CERScore:
    """Parse a standard MusicXML (<score-partwise>) document into a CERScore.

    Multi-staff <part> elements (<staves> > 1) are deterministically split into
    separate canonical 1-staff part streams so grand-staff and multi-part
    encodings are Bucket-A equivalent.

    Raises ValueError or xml.etree.ElementTree.ParseError on invalid input.
    """
    if not xml_str or not xml_str.strip():
        raise ValueError("Empty MusicXML input.")

    root = ET.fromstring(xml_str)
    if _local_tag(root.tag) != "score-partwise":
        raise ValueError(
            f"Unsupported MusicXML root element <{_local_tag(root.tag)}>; "
            "expected <score-partwise>."
        )

    part_elems = _iter_children(root, "part")
    parts: list[str] = []
    all_measures: list[CERMeasure] = []
    system_breaks: list[int] = []

    for p_idx, part_el in enumerate(part_elems):
        part_id = part_el.attrib.get("id", f"P{p_idx + 1}")
        part_measures: list[CERMeasure] = []
        part_system_breaks: list[int] = []

        divisions = 1
        num_staves = 1
        part_max_staff = 1
        staff_clefs: dict[int, str] = {1: "G2"}
        staff_ottava: dict[int, int] = {1: 0}
        key_fifths = 0
        time_num = 4
        time_den = 4
        initialized_context = False

        measure_elems = _iter_children(part_el, "measure")
        for m_idx, m_el in enumerate(measure_elems):
            m_num = m_el.attrib.get("number", str(m_idx + 1))
            is_implicit = m_el.attrib.get("implicit", "no").lower() == "yes"

            cursor = Fraction(0)
            last_note_onset = Fraction(0)
            barline_left = "regular"
            barline_right = "regular"
            repeat_start = False
            repeat_end = False
            volta: str | None = None

            notes: list[CERNote] = []
            note_contexts: list[CERContext] = []
            rests: list[CERRest] = []
            directions: list[CERDirection] = []
            context_changes: list[tuple[int, Fraction, str, str]] = []
            open_wedges: dict[int, tuple[str, Fraction]] = {}

            for child in m_el:
                tag = _local_tag(child.tag)

                if tag == "print":
                    if child.attrib.get("new-system", "no").lower() == "yes":
                        part_system_breaks.append(m_idx)

                elif tag == "attributes":
                    div_txt = _child_text(child, "divisions")
                    if div_txt:
                        parsed_div = int(div_txt)
                        if parsed_div <= 0:
                            raise ValueError(f"Invalid divisions={parsed_div}.")
                        divisions = parsed_div

                    staves_txt = _child_text(child, "staves")
                    if staves_txt:
                        num_staves = max(1, int(staves_txt))
                        part_max_staff = max(part_max_staff, num_staves)
                        for s_i in range(1, num_staves + 1):
                            staff_clefs.setdefault(s_i, "F4" if s_i == 2 else "G2")
                            staff_ottava.setdefault(s_i, 0)

                    for key_el in _iter_children(child, "key"):
                        fifths_txt = _child_text(key_el, "fifths")
                        if fifths_txt:
                            new_fifths = int(fifths_txt)
                            if initialized_context and new_fifths != key_fifths:
                                context_changes.append((1, cursor, "key", str(new_fifths)))
                            key_fifths = new_fifths

                    for time_el in _iter_children(child, "time"):
                        beats_txt = _child_text(time_el, "beats")
                        btype_txt = _child_text(time_el, "beat-type")
                        if beats_txt and btype_txt:
                            new_num = int(beats_txt.split("+")[0])
                            new_den = int(btype_txt)
                            if new_num <= 0 or new_den <= 0:
                                raise ValueError(f"Invalid time signature {new_num}/{new_den}.")
                            if initialized_context and (new_num != time_num or new_den != time_den):
                                context_changes.append((1, cursor, "time", f"{new_num}/{new_den}"))
                            time_num = new_num
                            time_den = new_den

                    for clef_el in _iter_children(child, "clef"):
                        s_num = max(1, int(clef_el.attrib.get("number", "1")))
                        sign = _child_text(clef_el, "sign", "G")
                        line = _child_text(clef_el, "line", "2")
                        clef_str = f"{sign}{line}"
                        prev_clef = staff_clefs.get(s_num)
                        if initialized_context and prev_clef is not None and clef_str != prev_clef:
                            context_changes.append((s_num, cursor, "clef", clef_str))
                        staff_clefs[s_num] = clef_str
                        staff_ottava.setdefault(s_num, 0)
                        part_max_staff = max(part_max_staff, s_num)

                    if m_idx == 0 and cursor == 0:
                        initialized_context = True

                elif tag == "backup":
                    dur_txt = _child_text(child, "duration")
                    if dur_txt:
                        delta = Fraction(int(dur_txt), 4 * divisions)
                        cursor = max(Fraction(0), cursor - delta)

                elif tag == "forward":
                    dur_txt = _child_text(child, "duration")
                    if dur_txt:
                        cursor += Fraction(int(dur_txt), 4 * divisions)

                elif tag == "barline":
                    loc = child.attrib.get("location", "right").lower()
                    style_txt = _child_text(child, "bar-style")
                    mapped_style = BAR_STYLE_MAP.get(style_txt, "regular") if style_txt else "regular"
                    rep_el = _find_child(child, "repeat")
                    end_el = _find_child(child, "ending")
                    if end_el is not None and end_el.attrib.get("number"):
                        volta = end_el.attrib["number"]
                    if rep_el is not None:
                        direction = rep_el.attrib.get("direction", "").lower()
                        if direction == "forward":
                            repeat_start = True
                            barline_left = "repeat-start"
                        elif direction == "backward":
                            repeat_end = True
                            barline_right = "repeat-end"
                    else:
                        if loc == "left":
                            barline_left = mapped_style
                        else:
                            barline_right = mapped_style

                elif tag == "direction":
                    dir_staff = max(1, int(_child_text(child, "staff", "1")))
                    part_max_staff = max(part_max_staff, dir_staff)
                    offset_txt = _child_text(child, "offset")
                    dir_onset = cursor
                    if offset_txt:
                        dir_onset = max(
                            Fraction(0),
                            cursor + Fraction(int(offset_txt), 4 * divisions),
                        )
                    for dt in _iter_children(child, "direction-type"):
                        for dt_child in dt:
                            dt_tag = _local_tag(dt_child.tag)
                            if dt_tag == "dynamics":
                                for dyn_el in dt_child:
                                    dyn_name = _local_tag(dyn_el.tag)
                                    directions.append(
                                        CERDirection("dynamic", dyn_name, dir_staff, dir_onset)
                                    )
                            elif dt_tag == "wedge":
                                w_type = dt_child.attrib.get("type", "").lower()
                                if w_type in ("crescendo", "diminuendo"):
                                    open_wedges[dir_staff] = (w_type, dir_onset)
                                elif w_type == "stop" and dir_staff in open_wedges:
                                    w_kind, w_start = open_wedges.pop(dir_staff)
                                    directions.append(
                                        CERDirection("wedge", w_kind, dir_staff, w_start, dir_onset)
                                    )
                            elif dt_tag == "octave-shift":
                                o_type = dt_child.attrib.get("type", "").lower()
                                o_size = int(dt_child.attrib.get("size", "8"))
                                oct_val = 1 if o_size <= 8 else 2
                                if o_type == "down":
                                    new_ott = oct_val  # 8va alta shifts written notes down
                                elif o_type == "up":
                                    new_ott = -oct_val  # 8vb bassa shifts written notes up
                                else:
                                    new_ott = 0
                                prev_ott = staff_ottava.get(dir_staff, 0)
                                if new_ott != prev_ott:
                                    context_changes.append((dir_staff, dir_onset, "ottava", str(new_ott)))
                                staff_ottava[dir_staff] = new_ott
                            elif dt_tag == "metronome":
                                pm = _child_text(dt_child, "per-minute")
                                if pm:
                                    directions.append(
                                        CERDirection("tempo", pm, dir_staff, dir_onset)
                                    )
                            elif dt_tag == "rehearsal":
                                r_txt = (dt_child.text or "").strip()
                                if r_txt:
                                    directions.append(
                                        CERDirection("rehearsal", r_txt, dir_staff, dir_onset)
                                    )

                elif tag == "harmony":
                    h_staff = max(1, int(_child_text(child, "staff", "1")))
                    part_max_staff = max(part_max_staff, h_staff)
                    root_el = _find_child(child, "root")
                    root_step = _child_text(root_el, "root-step") if root_el is not None else ""
                    root_alt = _child_text(root_el, "root-alter") if root_el is not None else ""
                    kind_txt = _child_text(child, "kind", "major")
                    if root_step:
                        h_val = f"{root_step}{root_alt}:{kind_txt}"
                        directions.append(CERDirection("harmony", h_val, h_staff, cursor))

                elif tag == "note":
                    initialized_context = True
                    is_grace = _find_child(child, "grace") is not None
                    is_chord = _find_child(child, "chord") is not None
                    is_cue = _find_child(child, "cue") is not None
                    rest_el = _find_child(child, "rest")
                    pitch_el = _find_child(child, "pitch")

                    voice = _child_text(child, "voice", "1")
                    staff = max(1, int(_child_text(child, "staff", "1")))
                    part_max_staff = max(part_max_staff, staff)
                    staff_clefs.setdefault(staff, "F4" if staff == 2 else "G2")
                    staff_ottava.setdefault(staff, 0)

                    dur_txt = _child_text(child, "duration")
                    if is_grace:
                        dur = Fraction(0)
                    elif dur_txt:
                        dur = Fraction(int(dur_txt), 4 * divisions)
                    else:
                        dur = Fraction(0)

                    onset = last_note_onset if is_chord else cursor

                    if rest_el is not None:
                        is_full = rest_el.attrib.get("measure", "no").lower() == "yes"
                        rests.append(
                            CERRest(
                                staff=staff,
                                voice=voice,
                                onset=onset,
                                duration=dur,
                                is_full_measure=is_full,
                            )
                        )
                    elif pitch_el is not None:
                        step = _child_text(pitch_el, "step", "C").upper()
                        if step not in STEP_TO_SEMITONE:
                            raise ValueError(f"Invalid pitch step {step!r}.")
                        alter = int(_child_text(pitch_el, "alter", "0"))
                        octave = int(_child_text(pitch_el, "octave", "4"))
                        midi = (octave + 1) * 12 + STEP_TO_SEMITONE[step] + alter

                        tie_start = False
                        tie_stop = False
                        for tie_el in _iter_children(child, "tie"):
                            t_type = tie_el.attrib.get("type", "").lower()
                            if t_type == "start":
                                tie_start = True
                            elif t_type == "stop":
                                tie_stop = True

                        tuplet: tuple[int, int] | None = None
                        tm_el = _find_child(child, "time-modification")
                        if tm_el is not None:
                            act_txt = _child_text(tm_el, "actual-notes")
                            norm_txt = _child_text(tm_el, "normal-notes")
                            if act_txt and norm_txt:
                                tuplet = (int(act_txt), int(norm_txt))

                        artics: list[str] = []
                        slur_starts = 0
                        slur_stops = 0
                        for not_el in _iter_children(child, "notations"):
                            for sub_not in not_el:
                                sn_tag = _local_tag(sub_not.tag)
                                if sn_tag in ("articulations", "ornaments", "technical"):
                                    for item in sub_not:
                                        artics.append(_local_tag(item.tag))
                                elif sn_tag in ("fermata", "arpeggiate"):
                                    artics.append(sn_tag)
                                elif sn_tag == "slur":
                                    s_type = sub_not.attrib.get("type", "").lower()
                                    if s_type == "start":
                                        slur_starts += 1
                                    elif s_type == "stop":
                                        slur_stops += 1
                                elif sn_tag == "tied":
                                    t_type = sub_not.attrib.get("type", "").lower()
                                    if t_type == "start":
                                        tie_start = True
                                    elif t_type == "stop":
                                        tie_stop = True

                        lyrics_list: list[str] = []
                        for lyr_el in _iter_children(child, "lyric"):
                            syl = _child_text(lyr_el, "text")
                            if syl:
                                lyrics_list.append(syl)

                        ctx = CERContext(
                            clef=staff_clefs[staff],
                            key_fifths=key_fifths,
                            time_num=time_num,
                            time_den=time_den,
                            ottava=staff_ottava[staff],
                        )
                        notes.append(
                            CERNote(
                                staff=staff,
                                voice=voice,
                                onset=onset,
                                duration=dur,
                                step=step,
                                alter=alter,
                                octave=octave,
                                midi=midi,
                                tie_start=tie_start,
                                tie_stop=tie_stop,
                                grace=is_grace,
                                tuplet=tuplet,
                                is_chord=is_chord,
                                cue=is_cue,
                                articulations=tuple(sorted(artics)),
                                lyrics=tuple(lyrics_list),
                                slur_starts=slur_starts,
                                slur_stops=slur_stops,
                            )
                        )
                        note_contexts.append(ctx)

                    if not is_chord and not is_grace:
                        last_note_onset = cursor
                        cursor += dur

            # Flush any wedges left open at end of measure
            for w_staff, (w_kind, w_start) in open_wedges.items():
                directions.append(
                    CERDirection("wedge", w_kind, w_staff, w_start, max(cursor, w_start))
                )

            measure_contexts = {
                s_i: CERContext(
                    clef=staff_clefs.get(s_i, "F4" if s_i == 2 else "G2"),
                    key_fifths=key_fifths,
                    time_num=time_num,
                    time_den=time_den,
                    ottava=staff_ottava.get(s_i, 0),
                )
                for s_i in range(1, max(num_staves, part_max_staff) + 1)
            }
            part_measures.append(
                CERMeasure(
                    part_id=part_id,
                    number=m_num,
                    index=m_idx,
                    barline_left=barline_left,
                    barline_right=barline_right,
                    repeat_start=repeat_start,
                    repeat_end=repeat_end,
                    volta=volta,
                    is_implicit=is_implicit,
                    contexts=measure_contexts,
                    context_changes=tuple(context_changes),
                    notes=tuple(notes),
                    note_contexts=tuple(note_contexts),
                    rests=tuple(rests),
                    directions=tuple(directions),
                )
            )

        s_parts, s_measures, s_breaks = _split_multistaff_part(
            part_id=part_id,
            part_measures=part_measures,
            part_max_staff=part_max_staff,
            part_system_breaks=part_system_breaks,
            base_measure_offset=len(all_measures),
        )
        parts.extend(s_parts)
        all_measures.extend(s_measures)
        system_breaks.extend(s_breaks)

    return CERScore(
        parts=tuple(parts),
        measures=tuple(all_measures),
        num_staves=max(1, len(parts)),
        system_breaks=tuple(sorted(set(system_breaks))),
    )


def _measure_similarity(m_pred: CERMeasure, m_gt: CERMeasure) -> float:
    """Measure similarity per docs/04: 0.7 * Dice(multiset(p, d)) + 0.3 * barline_match."""
    c_pred: Counter[Hashable] = Counter((n.midi, n.duration) for n in m_pred.notes)
    c_gt: Counter[Hashable] = Counter((n.midi, n.duration) for n in m_gt.notes)
    total = sum(c_pred.values()) + sum(c_gt.values())
    if total == 0:
        dice = 1.0
    else:
        dice = (2.0 * _multiset_intersection_size(c_pred, c_gt)) / total
    bar_match = (
        1.0
        if (
            m_pred.barline_left == m_gt.barline_left
            and m_pred.barline_right == m_gt.barline_right
        )
        else 0.0
    )
    return 0.7 * dice + 0.3 * bar_match


def align_measures(
    pred_measures: Sequence[CERMeasure],
    gt_measures: Sequence[CERMeasure],
    gap_penalty: float = 0.25,
) -> list[tuple[int | None, int | None]]:
    """Stage 1: Needleman-Wunsch DP alignment over measures."""
    n = len(pred_measures)
    m = len(gt_measures)
    if n == 0 and m == 0:
        return []
    if n == 0:
        return [(None, j) for j in range(m)]
    if m == 0:
        return [(i, None) for i in range(n)]

    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    bt = [[0] * (m + 1) for _ in range(n + 1)]  # 0=diag, 1=up (pred gap), 2=left (gt gap)

    for i in range(1, n + 1):
        dp[i][0] = -gap_penalty * i
        bt[i][0] = 1
    for j in range(1, m + 1):
        dp[0][j] = -gap_penalty * j
        bt[0][j] = 2

    for i in range(1, n + 1):
        mp = pred_measures[i - 1]
        for j in range(1, m + 1):
            sim = _measure_similarity(mp, gt_measures[j - 1])
            diag = dp[i - 1][j - 1] + sim
            up = dp[i - 1][j] - gap_penalty
            left = dp[i][j - 1] - gap_penalty
            if diag >= up and diag >= left:
                dp[i][j] = diag
                bt[i][j] = 0
            elif up >= left:
                dp[i][j] = up
                bt[i][j] = 1
            else:
                dp[i][j] = left
                bt[i][j] = 2

    aligned: list[tuple[int | None, int | None]] = []
    i, j = n, m
    while i > 0 or j > 0:
        move = bt[i][j]
        if i > 0 and j > 0 and move == 0:
            aligned.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif i > 0 and (j == 0 or move == 1):
            aligned.append((i - 1, None))
            i -= 1
        else:
            aligned.append((None, j - 1))
            j -= 1
    aligned.reverse()
    return aligned


def _hungarian_max_weight(weights: list[list[int]]) -> list[int]:
    """Exact Kuhn-Munkres (Hungarian) algorithm for maximum-weight assignment.

    Given an R x C non-negative integer matrix, returns assignment[r] = c
    (or -1 if row r is unmatched when R > C).
    """
    n_rows = len(weights)
    if n_rows == 0:
        return []
    n_cols = len(weights[0])
    if n_cols == 0:
        return [-1] * n_rows

    size = max(n_rows, n_cols)
    max_w = max((w for row in weights for w in row), default=0)

    # Convert to square minimization matrix of size x size
    cost = [[max_w] * size for _ in range(size)]
    for r in range(n_rows):
        for c in range(n_cols):
            cost[r][c] = max_w - weights[r][c]

    u = [0] * (size + 1)
    v = [0] * (size + 1)
    p = [0] * (size + 1)
    way = [0] * (size + 1)

    for i in range(1, size + 1):
        p[0] = i
        j0 = 0
        minv = [10**18] * (size + 1)
        used = [False] * (size + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = 10**18
            j1 = 0
            for j in range(1, size + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(size + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    row_to_col = [-1] * n_rows
    for j in range(1, size + 1):
        if p[j] != 0 and p[j] <= n_rows and j <= n_cols:
            row_to_col[p[j] - 1] = j - 1
    return row_to_col


def _optimal_voice_bijection(
    pred_notes: Sequence[CERNote],
    gt_notes: Sequence[CERNote],
    pred_rests: Sequence[CERRest],
    gt_rests: Sequence[CERRest],
) -> dict[str, str]:
    """Solve optimal voice-ID bijection pi: V_pred <-> V_gt maximizing matched events."""
    v_pred = sorted({n.voice for n in pred_notes} | {r.voice for r in pred_rests})
    v_gt = sorted({n.voice for n in gt_notes} | {r.voice for r in gt_rests})
    if not v_pred or not v_gt:
        return {}

    def _note_sig(n: CERNote) -> Hashable:
        return (
            n.onset,
            n.duration,
            n.step,
            n.alter,
            n.octave,
            n.grace,
            n.tuplet,
            n.is_chord,
            n.cue,
        )

    def _rest_sig(r: CERRest) -> Hashable:
        return ("R", r.onset, r.duration)

    pred_by_v: dict[str, Counter[Hashable]] = {v: Counter() for v in v_pred}
    for n in pred_notes:
        pred_by_v[n.voice][_note_sig(n)] += 1
    for r in pred_rests:
        pred_by_v[r.voice][_rest_sig(r)] += 1

    gt_by_v: dict[str, Counter[Hashable]] = {v: Counter() for v in v_gt}
    for n in gt_notes:
        gt_by_v[n.voice][_note_sig(n)] += 1
    for r in gt_rests:
        gt_by_v[r.voice][_rest_sig(r)] += 1

    weights = [
        [_multiset_intersection_size(pred_by_v[vp], gt_by_v[vg]) for vg in v_gt]
        for vp in v_pred
    ]
    assignment = _hungarian_max_weight(weights)
    mapping: dict[str, str] = {}
    for idx, col in enumerate(assignment):
        if col >= 0:
            mapping[v_pred[idx]] = v_gt[col]
        else:
            mapping[v_pred[idx]] = f"__unmatched_pred_{v_pred[idx]}"
    return mapping


def _compute_bsvr(measures: Sequence[CERMeasure]) -> float:
    """Beat-Sum Validity Rate across predicted (measure, staff, voice) streams."""
    valid = 0
    total = 0
    for m in measures:
        if m.is_implicit:
            continue
        by_sv: dict[tuple[int, str], Fraction] = defaultdict(Fraction)
        for n in m.notes:
            if not n.is_chord and not n.grace:
                by_sv[(n.staff, n.voice)] += n.duration
        for r in m.rests:
            by_sv[(r.staff, r.voice)] += r.duration
        for (staff, _voice), dur_sum in by_sv.items():
            if dur_sum == 0:
                continue
            ctx = m.contexts.get(staff) or next(iter(m.contexts.values()), CERContext())
            expected = Fraction(ctx.time_num, ctx.time_den)
            total += 1
            if dur_sum == expected:
                valid += 1
    if total == 0:
        return 1.0
    return valid / total


def _extract_sounding_events(score: CERScore) -> Counter[Hashable]:
    """Resolve ties and unroll measure onsets into sounding MIDI events."""
    events: Counter[Hashable] = Counter()
    # Track active tied notes per (part_id, staff, voice, midi) -> (start_onset, acc_dur)
    active_ties: dict[tuple[str, int, str, int], tuple[Fraction, Fraction]] = {}
    part_offset: dict[str, Fraction] = defaultdict(Fraction)

    for m in score.measures:
        m_start = part_offset[m.part_id]
        ctx0 = next(iter(m.contexts.values()), CERContext())
        m_len = Fraction(ctx0.time_num, ctx0.time_den)

        for n in m.notes:
            if n.grace:
                continue
            key = (m.part_id, n.staff, n.voice, n.midi)
            abs_onset = m_start + n.onset
            if n.tie_stop and key in active_ties:
                start_on, acc_dur = active_ties.pop(key)
                total_dur = acc_dur + n.duration
                if n.tie_start:
                    active_ties[key] = (start_on, total_dur)
                else:
                    events[(n.midi, start_on, total_dur)] += 1
            elif n.tie_start:
                active_ties[key] = (abs_onset, n.duration)
            else:
                events[(n.midi, abs_onset, n.duration)] += 1

        part_offset[m.part_id] = m_start + m_len

    for (_, _, _, midi), (start_on, acc_dur) in active_ties.items():
        events[(midi, start_on, acc_dur)] += 1
    return events


def _has_repetition_loop(pred_measures: Sequence[CERMeasure], gt_measures: Sequence[CERMeasure]) -> bool:
    """Detect measure n-gram (n >= 2) repeating >= 3x consecutively when absent in GT."""
    def _sig(m: CERMeasure) -> tuple[tuple[int, Fraction], ...]:
        return tuple(sorted((n.midi, n.duration) for n in m.notes))

    pred_sigs = [_sig(m) for m in pred_measures]
    gt_sigs = [_sig(m) for m in gt_measures]

    if len(pred_sigs) < 6:
        return False

    for n_gram in (2, 3, 4):
        if len(pred_sigs) < 3 * n_gram:
            continue
        gt_triples = {
            tuple(gt_sigs[i : i + 3 * n_gram])
            for i in range(max(0, len(gt_sigs) - 3 * n_gram + 1))
        }
        for i in range(len(pred_sigs) - 3 * n_gram + 1):
            chunk = tuple(pred_sigs[i : i + n_gram])
            if not any(chunk):  # ignore all-empty measures
                continue
            if (
                tuple(pred_sigs[i + n_gram : i + 2 * n_gram]) == chunk
                and tuple(pred_sigs[i + 2 * n_gram : i + 3 * n_gram]) == chunk
            ):
                triple = tuple(pred_sigs[i : i + 3 * n_gram])
                if triple not in gt_triples:
                    return True
    return False


def _measure_transcribe_cost(m: CERMeasure) -> float:
    """Baseline manual transcription cost in NEU for a single GT measure."""
    n_notes = len(m.notes)
    n_rests = len(m.rests)
    n_expr = len(m.directions) + sum(len(n.articulations) + len(n.lyrics) for n in m.notes)
    if n_notes == 0 and n_rests == 0 and n_expr == 0:
        return 0.0
    return 2.5 * n_notes + 1.0 * n_rests + 0.5 * n_expr


def align_and_score_cer(pred: CERScore, gt: CERScore) -> CERAlignmentResult:
    """Run full 2-stage hierarchical alignment and compute all 4 OMR-NS axes."""
    alignment = align_measures(pred.measures, gt.measures, gap_penalty=0.25)

    matched_t1a = 0
    matched_t1b = 0
    matched_t1c = 0
    matched_rests = 0
    matched_barlines = 0
    exact_measures = 0
    ctx_matched = 0
    ctx_total = 0

    # Expressive layer counters: (matched, n_pred, n_gt)
    expr_counts: dict[str, list[int]] = {
        "ctx": [0, 0, 0],
        "art": [0, 0, 0],
        "dyn": [0, 0, 0],
        "tie_slur": [0, 0, 0],
        "text": [0, 0, 0],
    }

    total_wfpp = 0.0
    total_transcribe = 0.0
    gt_measure_matched_notes: dict[int, int] = {}

    for p_idx, g_idx in alignment:
        mp = pred.measures[p_idx] if p_idx is not None else None
        mg = gt.measures[g_idx] if g_idx is not None else None

        if mg is None and mp is not None:
            # Extra hallucinated predicted measure
            extra_edit = 5.0 + 2.5 * len(mp.notes) + 1.0 * len(mp.rests)
            total_wfpp += extra_edit
            expr_counts["ctx"][1] += len(mp.context_changes)
            expr_counts["art"][1] += sum(len(n.articulations) for n in mp.notes)
            expr_counts["dyn"][1] += sum(1 for d in mp.directions if d.kind in ("dynamic", "wedge"))
            expr_counts["tie_slur"][1] += sum(
                (1 if n.tie_start else 0) + n.slur_starts for n in mp.notes
            )
            expr_counts["text"][1] += sum(len(n.lyrics) for n in mp.notes) + sum(
                1 for d in mp.directions if d.kind in ("harmony", "tempo", "rehearsal")
            )
            continue

        if mg is not None and mp is None:
            # Missing GT measure
            m_tc = _measure_transcribe_cost(mg)
            total_transcribe += m_tc
            total_wfpp += compute_effective_measure_cost(5.0 + m_tc, m_tc)
            gt_measure_matched_notes[g_idx] = 0  # type: ignore[index]
            expr_counts["ctx"][2] += len(mg.context_changes)
            expr_counts["art"][2] += sum(len(n.articulations) for n in mg.notes)
            expr_counts["dyn"][2] += sum(1 for d in mg.directions if d.kind in ("dynamic", "wedge"))
            expr_counts["tie_slur"][2] += sum(
                (1 if n.tie_start else 0) + n.slur_starts for n in mg.notes
            )
            expr_counts["text"][2] += sum(len(n.lyrics) for n in mg.notes) + sum(
                1 for d in mg.directions if d.kind in ("harmony", "tempo", "rehearsal")
            )
            continue

        assert mp is not None and mg is not None and g_idx is not None
        m_tc = _measure_transcribe_cost(mg)
        total_transcribe += m_tc

        # Barline match
        bar_ok = (
            mp.barline_left == mg.barline_left and mp.barline_right == mg.barline_right
        )
        if bar_ok:
            matched_barlines += 1

        # Stage 2: per-staff optimal voice bijection
        staves = sorted(
            {n.staff for n in mp.notes}
            | {n.staff for n in mg.notes}
            | {r.staff for r in mp.rests}
            | {r.staff for r in mg.rests}
        )
        voice_map_by_staff: dict[int, dict[str, str]] = {}
        for s in staves:
            p_sn = [n for n in mp.notes if n.staff == s]
            g_sn = [n for n in mg.notes if n.staff == s]
            p_sr = [r for r in mp.rests if r.staff == s]
            g_sr = [r for r in mg.rests if r.staff == s]
            voice_map_by_staff[s] = _optimal_voice_bijection(p_sn, g_sn, p_sr, g_sr)

        # Tier 1a: Sounding (midi, onset)
        c_p_t1a: Counter[Hashable] = Counter((n.midi, n.onset) for n in mp.notes)
        c_g_t1a: Counter[Hashable] = Counter((n.midi, n.onset) for n in mg.notes)
        m_t1a = _multiset_intersection_size(c_p_t1a, c_g_t1a)
        matched_t1a += m_t1a

        # Tier 1b: Rhythmic (midi, onset, duration, grace, tuplet)
        c_p_t1b: Counter[Hashable] = Counter(
            (n.midi, n.onset, n.duration, n.grace, n.tuplet) for n in mp.notes
        )
        c_g_t1b: Counter[Hashable] = Counter(
            (n.midi, n.onset, n.duration, n.grace, n.tuplet) for n in mg.notes
        )
        m_t1b = _multiset_intersection_size(c_p_t1b, c_g_t1b)
        matched_t1b += m_t1b

        # Tier 1c: Strict Engraving (T1b + spelling + staff + mapped voice + chord/cue)
        def _strict_key(n: CERNote, mapped_v: str) -> Hashable:
            return (
                n.midi,
                n.onset,
                n.duration,
                n.grace,
                n.tuplet,
                n.step,
                n.alter,
                n.octave,
                n.staff,
                mapped_v,
                n.is_chord,
                n.cue,
            )

        p_strict_with_ctx: dict[Hashable, list[CERContext]] = defaultdict(list)
        g_strict_with_ctx: dict[Hashable, list[CERContext]] = defaultdict(list)
        c_p_t1c: Counter[Hashable] = Counter()
        c_g_t1c: Counter[Hashable] = Counter()

        for n, ctx in zip(mp.notes, mp.note_contexts):
            mv = voice_map_by_staff.get(n.staff, {}).get(n.voice, n.voice)
            k = _strict_key(n, mv)
            c_p_t1c[k] += 1
            p_strict_with_ctx[k].append(ctx)

        for n, ctx in zip(mg.notes, mg.note_contexts):
            k = _strict_key(n, n.voice)
            c_g_t1c[k] += 1
            g_strict_with_ctx[k].append(ctx)

        m_t1c = _multiset_intersection_size(c_p_t1c, c_g_t1c)
        matched_t1c += m_t1c
        gt_measure_matched_notes[g_idx] = m_t1c

        # Active Context State Accuracy (A_state) at every matched strict note onset
        for k, cnt in (c_p_t1c & c_g_t1c).items():
            p_ctxs = p_strict_with_ctx[k]
            g_ctxs = g_strict_with_ctx[k]
            for idx_c in range(cnt):
                ctx_total += 1
                if p_ctxs[idx_c] == g_ctxs[idx_c]:
                    ctx_matched += 1

        # Rest matching (Bucket A: full-measure rest canonicalized to equal-duration rest)
        c_p_rest: Counter[Hashable] = Counter(
            (r.staff, voice_map_by_staff.get(r.staff, {}).get(r.voice, r.voice), r.onset, r.duration)
            for r in mp.rests
        )
        c_g_rest: Counter[Hashable] = Counter(
            (r.staff, r.voice, r.onset, r.duration) for r in mg.rests
        )
        m_rest = _multiset_intersection_size(c_p_rest, c_g_rest)
        matched_rests += m_rest

        # Measure-Exact Rate (MER)
        m_bsvr = _compute_bsvr([mp])
        if (
            bar_ok
            and m_t1c == len(mp.notes) == len(mg.notes)
            and m_rest == len(mp.rests) == len(mg.rests)
            and m_bsvr == 1.0
        ):
            exact_measures += 1

        # Expressive layers
        # 1. Mid-piece context changes
        c_p_ctx: Counter[Hashable] = Counter(mp.context_changes)
        c_g_ctx: Counter[Hashable] = Counter(mg.context_changes)
        expr_counts["ctx"][0] += _multiset_intersection_size(c_p_ctx, c_g_ctx)
        expr_counts["ctx"][1] += sum(c_p_ctx.values())
        expr_counts["ctx"][2] += sum(c_g_ctx.values())

        # 2. Articulations & ornaments
        c_p_art: Counter[Hashable] = Counter(
            (n.staff, n.onset, n.midi, a) for n in mp.notes for a in n.articulations
        )
        c_g_art: Counter[Hashable] = Counter(
            (n.staff, n.onset, n.midi, a) for n in mg.notes for a in n.articulations
        )
        expr_counts["art"][0] += _multiset_intersection_size(c_p_art, c_g_art)
        expr_counts["art"][1] += sum(c_p_art.values())
        expr_counts["art"][2] += sum(c_g_art.values())

        # 3. Dynamics & hairpins
        c_p_dyn: Counter[Hashable] = Counter(
            (d.kind, d.value, d.staff, d.onset)
            for d in mp.directions
            if d.kind in ("dynamic", "wedge")
        )
        c_g_dyn: Counter[Hashable] = Counter(
            (d.kind, d.value, d.staff, d.onset)
            for d in mg.directions
            if d.kind in ("dynamic", "wedge")
        )
        expr_counts["dyn"][0] += _multiset_intersection_size(c_p_dyn, c_g_dyn)
        expr_counts["dyn"][1] += sum(c_p_dyn.values())
        expr_counts["dyn"][2] += sum(c_g_dyn.values())

        # 4. Ties & slurs
        c_p_ts: Counter[Hashable] = Counter()
        for n in mp.notes:
            if n.tie_start:
                c_p_ts[("tie", n.staff, n.onset, n.midi)] += 1
            if n.slur_starts > 0:
                c_p_ts[("slur", n.staff, n.onset)] += n.slur_starts
        c_g_ts: Counter[Hashable] = Counter()
        for n in mg.notes:
            if n.tie_start:
                c_g_ts[("tie", n.staff, n.onset, n.midi)] += 1
            if n.slur_starts > 0:
                c_g_ts[("slur", n.staff, n.onset)] += n.slur_starts
        expr_counts["tie_slur"][0] += _multiset_intersection_size(c_p_ts, c_g_ts)
        expr_counts["tie_slur"][1] += sum(c_p_ts.values())
        expr_counts["tie_slur"][2] += sum(c_g_ts.values())

        # 5. Textual & symbolic layers (lyrics, harmony, tempo, rehearsal)
        c_p_txt: Counter[Hashable] = Counter(
            ("lyr", n.staff, n.onset, syl) for n in mp.notes for syl in n.lyrics
        )
        for d in mp.directions:
            if d.kind in ("harmony", "tempo", "rehearsal"):
                c_p_txt[(d.kind, d.staff, d.onset, d.value)] += 1
        c_g_txt: Counter[Hashable] = Counter(
            ("lyr", n.staff, n.onset, syl) for n in mg.notes for syl in n.lyrics
        )
        for d in mg.directions:
            if d.kind in ("harmony", "tempo", "rehearsal"):
                c_g_txt[(d.kind, d.staff, d.onset, d.value)] += 1
        expr_counts["text"][0] += _multiset_intersection_size(c_p_txt, c_g_txt)
        expr_counts["text"][1] += sum(c_p_txt.values())
        expr_counts["text"][2] += sum(c_g_txt.values())

        # Per-measure edit cost (NEU) capped by Rational Musician Re-Entry Cap
        # Notes matching T1b but failing T1c -> 1.0 NEU (respell / voice reassign);
        # Notes matching T1a but failing T1b -> 1.0 NEU (duration tweak);
        # Completely unmatched pred/gt notes -> 2.5 NEU (insert/delete).
        respell_edits = max(0, m_t1b - m_t1c)
        dur_edits = max(0, m_t1a - m_t1b)
        ins_del_notes = (len(mp.notes) - m_t1a) + (len(mg.notes) - m_t1a)
        ins_del_rests = (len(mp.rests) - m_rest) + (len(mg.rests) - m_rest)
        bar_edits = 0.0 if bar_ok else 5.0
        expr_err = (
            (sum(c_p_art.values()) + sum(c_g_art.values()) - 2 * _multiset_intersection_size(c_p_art, c_g_art))
            + (sum(c_p_dyn.values()) + sum(c_g_dyn.values()) - 2 * _multiset_intersection_size(c_p_dyn, c_g_dyn))
            + (sum(c_p_ts.values()) + sum(c_g_ts.values()) - 2 * _multiset_intersection_size(c_p_ts, c_g_ts))
        )
        raw_edit_cost = (
            1.0 * respell_edits
            + 1.0 * dur_edits
            + 2.5 * max(0, ins_del_notes)
            + 2.5 * max(0, ins_del_rests)
            + bar_edits
            + 0.5 * max(0, expr_err)
        )
        total_wfpp += compute_effective_measure_cost(raw_edit_cost, m_tc)

    n_pred = pred.total_notes
    n_gt = gt.total_notes
    m_pred = pred.total_measures
    m_gt = gt.total_measures

    f1_t1a = _f1_from_counts(matched_t1a, n_pred, n_gt)
    f1_t1b = _f1_from_counts(matched_t1b, n_pred, n_gt)
    f1_t1c = _f1_from_counts(matched_t1c, n_pred, n_gt)
    total_pred_rests = sum(len(m.rests) for m in pred.measures)
    total_gt_rests = sum(len(m.rests) for m in gt.measures)
    f1_rest = _f1_from_counts(matched_rests, total_pred_rests, total_gt_rests)

    mer = (exact_measures / m_gt) if m_gt > 0 else (1.0 if m_pred == 0 else 0.0)
    f1_bar = _f1_from_counts(matched_barlines, m_pred, m_gt)
    bsvr = _compute_bsvr(pred.measures)
    a_state = (ctx_matched / ctx_total) if ctx_total > 0 else (1.0 if n_gt == 0 and n_pred == 0 else 0.0)

    # S_struct: renormalize weights over active components (MER=0.25, F1_bar=0.20, BSVR=0.15, A_state=0.20)
    struct_w = 0.25 + 0.20 + 0.15 + 0.20
    s_struct = (0.25 * mer + 0.20 * f1_bar + 0.15 * bsvr + 0.20 * a_state) / struct_w

    # S_expr: macro-average F1 over active expressive layers
    active_expr_f1s: list[float] = []
    for matched_e, np_e, ng_e in expr_counts.values():
        if np_e > 0 or ng_e > 0:
            active_expr_f1s.append(_f1_from_counts(matched_e, np_e, ng_e))
    s_expr = (
        sum(active_expr_f1s) / len(active_expr_f1s)
        if active_expr_f1s
        else 1.0
    )

    # S_play: unrolled sounding MIDI-event F1 (resolving ties)
    play_pred = _extract_sounding_events(pred)
    play_gt = _extract_sounding_events(gt)
    play_matched = _multiset_intersection_size(play_pred, play_gt)
    s_play = _f1_from_counts(play_matched, sum(play_pred.values()), sum(play_gt.values()))

    # Dropped system count (when system breaks are annotated in GT)
    k_dropped = 0
    if gt.system_breaks:
        starts = [0] + [b for b in gt.system_breaks if 0 < b < m_gt]
        starts = sorted(set(starts))
        for idx_s, s_start in enumerate(starts):
            s_end = starts[idx_s + 1] if idx_s + 1 < len(starts) else m_gt
            sys_gt_notes = sum(len(gt.measures[k].notes) for k in range(s_start, s_end))
            if sys_gt_notes == 0:
                continue
            sys_matched = sum(gt_measure_matched_notes.get(k, 0) for k in range(s_start, s_end))
            if (sys_matched / sys_gt_notes) < 0.15:
                k_dropped += 1

    rep_loop = _has_repetition_loop(pred.measures, gt.measures)

    return CERAlignmentResult(
        f1_t1a=f1_t1a,
        f1_t1b=f1_t1b,
        f1_t1c=f1_t1c,
        f1_rest=f1_rest,
        mer=mer,
        f1_bar=f1_bar,
        bsvr=bsvr,
        a_state=a_state,
        s_note=f1_t1c,
        s_struct=s_struct,
        s_expr=s_expr,
        s_play=s_play,
        wfpp=total_wfpp,
        transcribe_cost=total_transcribe,
        n_gt=n_gt,
        n_pred=n_pred,
        m_gt=m_gt,
        m_pred=m_pred,
        k_dropped_systems=k_dropped,
        has_repetition_loop=rep_loop,
        measure_alignment=tuple(alignment),
    )


def score_musicxml_pair(
    pred_xml: str,
    gt_xml: str,
    *,
    page_id: str,
    work_id: str,
    domain: str,
    tier: str = "T1",
    paired_anchor_id: str | None = None,
    engraver: str | None = None,
    latency_s: float | None = None,
) -> PageMetrics:
    """End-to-end MusicXML pair scorer producing a PageMetrics for build_report()."""
    gt_score = parse_musicxml(gt_xml)
    gt_tc = sum(_measure_transcribe_cost(m) for m in gt_score.measures)

    try:
        pred_score = parse_musicxml(pred_xml)
    except (ET.ParseError, ValueError):
        return PageMetrics(
            page_id=page_id,
            work_id=work_id,
            domain=domain,
            tier=tier,
            s_note=0.0,
            s_struct=0.0,
            s_expr=0.0,
            s_play=0.0,
            wfpp=max(1.2 * gt_tc, 1.5 if gt_tc == 0.0 else 0.0),
            transcribe_cost=gt_tc,
            parseable=False,
            n_gt=gt_score.total_notes,
            n_pred=0,
            k_dropped_systems=0,
            has_repetition_loop=False,
            bsvr=0.0,
            m_gt=gt_score.total_measures,
            m_pred=0,
            paired_anchor_id=paired_anchor_id,
            engraver=engraver,
            latency_s=latency_s,
        )

    res = align_and_score_cer(pred_score, gt_score)
    return PageMetrics(
        page_id=page_id,
        work_id=work_id,
        domain=domain,
        tier=tier,
        s_note=res.s_note,
        s_struct=res.s_struct,
        s_expr=res.s_expr,
        s_play=res.s_play,
        wfpp=res.wfpp,
        transcribe_cost=res.transcribe_cost,
        parseable=True,
        n_gt=res.n_gt,
        n_pred=res.n_pred,
        k_dropped_systems=res.k_dropped_systems,
        has_repetition_loop=res.has_repetition_loop,
        bsvr=res.bsvr,
        m_gt=res.m_gt,
        m_pred=res.m_pred,
        paired_anchor_id=paired_anchor_id,
        engraver=engraver,
        latency_s=latency_s,
    )
