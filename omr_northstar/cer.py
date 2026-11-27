"""Deterministic MusicXML -> Canonical Evaluation Representation (CER) & 2-Stage Aligner.

Zero external dependencies (Python 3.10+ standard library only).
Implements the CER data model and two-stage hierarchical alignment specified in
docs/04-metrics-and-human-utility.md:
  1. Part-level Hungarian bijection + Measure-level Needleman-Wunsch DP
     (sim = 0.7 * Dice + 0.3 * barline_match when Dice > 0, gap = 0.25).
  2. Optimal voice-ID bijection (exact linear sum assignment via Hungarian algorithm)
     + exact bipartite/multiset matching across L1a (Sounding), L1b (Rhythmic),
     and L1c (Strict Engraving) tiers.
"""

from __future__ import annotations

import math
import unicodedata
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from fractions import Fraction
from pathlib import Path
from typing import Hashable, Sequence

from .scoring import ERROR_BUCKET_KEYS, PageMetrics, compute_effective_measure_cost

STEP_TO_SEMITONE: dict[str, int] = {
    "C": 0,
    "D": 2,
    "E": 4,
    "F": 5,
    "G": 7,
    "A": 9,
    "B": 11,
}

_TYPE_TO_WHOLE: dict[str, Fraction] = {
    "maxima": Fraction(8, 1),
    "long": Fraction(4, 1),
    "breve": Fraction(2, 1),
    "whole": Fraction(1, 1),
    "half": Fraction(1, 2),
    "quarter": Fraction(1, 4),
    "eighth": Fraction(1, 8),
    "16th": Fraction(1, 16),
    "32nd": Fraction(1, 32),
    "64th": Fraction(1, 64),
    "128th": Fraction(1, 128),
    "256th": Fraction(1, 256),
    "512th": Fraction(1, 512),
    "1024th": Fraction(1, 1024),
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

# Standard musical rational denominators (whole-note units) up to 192:
# binary subdivisions (1..128), triplets/sextuplets/nontuplets (3..192),
# quintuplets (5..160), septuplets (7..112), and compound tuplets (15..120).
_MUSICAL_DENOMINATORS: tuple[int, ...] = (
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 16, 18, 20, 24,
    28, 30, 32, 36, 40, 48, 56, 60, 64, 72, 80, 96, 112, 120, 128, 144, 160, 192,
)
_SNAP_MAX_DENOM: int = 192
_SNAP_TIGHT_TOL: Fraction = Fraction(1, 10000)
_SNAP_TOL: Fraction = Fraction(1, 1000)


def _snap_rational(val: Fraction, tol: Fraction = _SNAP_TOL) -> Fraction:
    """Snap MusicXML <divisions> rounding residuals to exact musical rationals.

    Values already on the denominator <= 192 grid (e.g. 1/3, 1/6, 1/12, 1/24,
    1/48, 1/5, 1/7, 1/32, 1/64, 1/128) are returned unchanged. Otherwise, values
    within ``tol`` (default 1e-3 whole notes) of 0 or of a simple musical
    rational with denominator <= 192 are snapped cleanly, using a tight pass
    (1e-4) first so tiny cursor residuals (e.g. 1/40320) preserve exact fine
    subdivisions.
    """
    if val.denominator <= _SNAP_MAX_DENOM:
        return val
    if abs(val) <= tol:
        return Fraction(0)
    tight = min(_SNAP_TIGHT_TOL, tol)
    for pass_tol in (tight, tol):
        for d in _MUSICAL_DENOMINATORS:
            cand = Fraction(round(val * d), d)
            if abs(val - cand) <= pass_tol:
                return cand
    cand = val.limit_denominator(_SNAP_MAX_DENOM)
    if abs(val - cand) <= tol:
        return cand
    return val


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
    stripped = child.text.strip()
    return stripped if stripped else default


def _parse_fraction(txt: str) -> Fraction:
    """Parse an integer or decimal numeric string into an exact Fraction."""
    return Fraction(txt.strip())


def _parse_int(txt: str, default: int = 0) -> int:
    """Parse an integer or decimal-formatted integer string (e.g. '1.0')."""
    s = txt.strip()
    if not s:
        return default
    return int(round(float(_parse_fraction(s))))


def _duration_from_type(note_el: ET.Element) -> Fraction:
    """Fallback duration calculation from <type>, <dot/>, and <time-modification>."""
    type_txt = _child_text(note_el, "type").lower()
    base = _TYPE_TO_WHOLE.get(type_txt, Fraction(0))
    if base <= 0:
        return Fraction(0)
    n_dots = len(_iter_children(note_el, "dot"))
    if n_dots > 0:
        base = base * Fraction((1 << (n_dots + 1)) - 1, 1 << n_dots)
    tm_el = _find_child(note_el, "time-modification")
    if tm_el is not None:
        act_txt = _child_text(tm_el, "actual-notes")
        norm_txt = _child_text(tm_el, "normal-notes")
        if act_txt and norm_txt:
            act_v = _parse_int(act_txt, 0)
            norm_v = _parse_int(norm_txt, 0)
            if act_v > 0 and norm_v > 0:
                base = base * Fraction(norm_v, act_v)
    return base


def _base_part_id(part_id: str) -> str:
    """Strip canonical multi-staff split suffix ('_s<N>') to recover base part ID."""
    idx = part_id.rfind("_s")
    if idx > 0 and part_id[idx + 2 :].isdigit():
        return part_id[:idx]
    return part_id


def _nfc(text: str) -> str:
    """Normalize textual annotations to Unicode NFC."""
    return unicodedata.normalize("NFC", text)


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
    alter: int | float
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
    articulations: tuple[str, ...] = ()


@dataclass(frozen=True)
class CERForward:
    """Canonical MusicXML <forward> cursor advance (in whole notes)."""

    staff: int
    voice: str | None
    onset: Fraction
    duration: Fraction


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
    forwards: tuple[CERForward, ...] = ()
    directions: tuple[CERDirection, ...] = ()
    is_senza_misura: bool = False


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
    error_buckets: dict[str, float] = field(default_factory=dict)


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
    per-staff contexts, and system break indices. Recomputes per-staff chord flags
    and credits cross-staff voice durations via synthetic CERForward entries.
    """
    if part_max_staff <= 1:
        norm_measures = [
            replace(
                m,
                context_changes=tuple(
                    (1 if c_staff == 0 else c_staff, c_onset, c_kind, c_val)
                    for c_staff, c_onset, c_kind, c_val in m.context_changes
                ),
            )
            for m in part_measures
        ]
        return (
            [part_id],
            norm_measures,
            [base_measure_offset + b for b in part_system_breaks],
        )

    split_parts: list[str] = []
    split_measures: list[CERMeasure] = []
    split_breaks: list[int] = []
    n_measures = len(part_measures)

    # Precompute per-measure, per-(voice, staff) occupied onset->duration slots
    # for cross-staff voice BSVR credit.
    measure_voice_staff_slots: list[dict[tuple[str, int], dict[Fraction, Fraction]]] = []
    for m in part_measures:
        slots: dict[tuple[str, int], dict[Fraction, Fraction]] = defaultdict(dict)
        for n in m.notes:
            if not n.grace and n.duration > 0:
                d_map = slots[(n.voice, n.staff)]
                if n.onset not in d_map or n.duration > d_map[n.onset]:
                    d_map[n.onset] = n.duration
        for r in m.rests:
            if r.duration > 0:
                d_map = slots[(r.voice, r.staff)]
                if r.onset not in d_map or r.duration > d_map[r.onset]:
                    d_map[r.onset] = r.duration
        for f in m.forwards:
            if f.voice is not None and f.duration > 0:
                d_map = slots[(f.voice, f.staff)]
                if f.onset not in d_map or f.duration > d_map[f.onset]:
                    d_map[f.onset] = f.duration
        measure_voice_staff_slots.append(slots)

    for s in range(1, part_max_staff + 1):
        s_part_id = f"{part_id}_s{s}"
        split_parts.append(s_part_id)
        s_offset = base_measure_offset + (s - 1) * n_measures
        for b in part_system_breaks:
            split_breaks.append(s_offset + b)

        for m_i, m in enumerate(part_measures):
            s_notes: list[CERNote] = []
            s_note_ctxs: list[CERContext] = []
            seen_ov: set[tuple[str, Fraction, bool]] = set()
            for n, ctx in zip(m.notes, m.note_contexts):
                if n.staff == s:
                    ov_key = (n.voice, n.onset, n.grace)
                    new_is_chord = ov_key in seen_ov
                    seen_ov.add(ov_key)
                    s_notes.append(replace(n, staff=1, is_chord=new_is_chord))
                    s_note_ctxs.append(ctx)

            s_rests = tuple(replace(r, staff=1) for r in m.rests if r.staff == s)
            s_forwards_list: list[CERForward] = [
                replace(f, staff=1) for f in m.forwards if f.staff == s
            ]

            # Credit cross-staff voice events from sibling staves via synthetic CERForward
            active_voices = {n.voice for n in s_notes} | {r.voice for r in s_rests}
            slots = measure_voice_staff_slots[m_i]
            for v in sorted(active_voices):
                own_onsets = (
                    {n.onset for n in s_notes if n.voice == v and not n.grace}
                    | {r.onset for r in s_rests if r.voice == v}
                    | {f.onset for f in s_forwards_list if f.voice == v}
                )
                sib_onsets: dict[Fraction, Fraction] = {}
                for other_s in range(1, part_max_staff + 1):
                    if other_s == s:
                        continue
                    for sib_on, sib_dur in slots.get((v, other_s), {}).items():
                        if sib_on not in own_onsets:
                            if sib_on not in sib_onsets or sib_dur > sib_onsets[sib_on]:
                                sib_onsets[sib_on] = sib_dur
                for sib_on in sorted(sib_onsets):
                    s_forwards_list.append(
                        CERForward(
                            staff=1,
                            voice=v,
                            onset=sib_on,
                            duration=sib_onsets[sib_on],
                        )
                    )

            s_dirs = tuple(replace(d, staff=1) for d in m.directions if d.staff == s)
            s_ctx_changes = tuple(
                (1, c_onset, c_kind, c_val)
                for c_staff, c_onset, c_kind, c_val in m.context_changes
                if c_staff == 0 or c_staff == s
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
                    forwards=tuple(s_forwards_list),
                    directions=s_dirs,
                    is_senza_misura=m.is_senza_misura,
                )
            )

    return split_parts, split_measures, split_breaks


def _timewise_to_partwise(root: ET.Element) -> ET.Element:
    """Convert a <score-timewise> XML tree to an equivalent <score-partwise> tree."""
    pw_root = ET.Element("score-partwise", dict(root.attrib))
    part_nodes: dict[str, ET.Element] = {}
    part_order: list[str] = []

    for child in root:
        tag = _local_tag(child.tag)
        if tag != "measure":
            pw_root.append(child)
            continue
        m_attrib = dict(child.attrib)
        for p_idx, part_el in enumerate(_iter_children(child, "part")):
            pid = part_el.attrib.get("id", f"P{p_idx + 1}")
            if pid not in part_nodes:
                part_nodes[pid] = ET.Element("part", {"id": pid})
                part_order.append(pid)
            m_copy = ET.Element("measure", m_attrib)
            for m_child in part_el:
                m_copy.append(m_child)
            part_nodes[pid].append(m_copy)

    for pid in part_order:
        pw_root.append(part_nodes[pid])
    return pw_root


def parse_musicxml(xml_str: str) -> CERScore:
    """Parse a MusicXML (<score-partwise> or <score-timewise>) document into a CERScore.

    Multi-staff <part> elements (<staves> > 1) are deterministically split into
    separate canonical 1-staff part streams so grand-staff and multi-part
    encodings are Bucket-A equivalent.

    Raises ValueError or xml.etree.ElementTree.ParseError on invalid input.
    """
    if not isinstance(xml_str, str) or not xml_str.strip():
        raise ValueError("Empty MusicXML input.")

    root = ET.fromstring(xml_str)
    root_tag = _local_tag(root.tag)
    if root_tag == "score-timewise":
        root = _timewise_to_partwise(root)
    elif root_tag != "score-partwise":
        raise ValueError(
            f"Unsupported MusicXML root element <{root_tag}>; "
            "expected <score-partwise> or <score-timewise>."
        )

    part_elems = _iter_children(root, "part")
    parts: list[str] = []
    all_measures: list[CERMeasure] = []
    system_breaks: list[int] = []

    for p_idx, part_el in enumerate(part_elems):
        part_id = part_el.attrib.get("id", f"P{p_idx + 1}")
        part_measures: list[CERMeasure] = []
        part_measure_max_cursors: list[Fraction] = []
        part_system_breaks: list[int] = []

        divisions: Fraction = Fraction(1)
        num_staves = 1
        part_max_staff = 1
        staff_clefs: dict[int, str] = {1: "G2"}
        staff_clef_initialized: set[int] = set()
        staff_ottava: dict[int, int] = {1: 0}
        staff_time: dict[int, tuple[int, int]] = {1: (4, 4)}
        staff_senza_misura: dict[int, bool] = {1: False}
        part_time: tuple[int, int] = (4, 4)
        part_senza_misura: bool = False
        part_transpose: int = 0
        staff_transpose: dict[int, int] = {}
        key_fifths = 0
        initialized_context = False

        # Carry open wedges across measures keyed by (staff, wedge_number)
        open_wedges: dict[tuple[int, str], tuple[str, int, Fraction]] = {}

        measure_elems = _iter_children(part_el, "measure")
        for m_idx, m_el in enumerate(measure_elems):
            if m_idx > 0:
                initialized_context = True
            m_num = m_el.attrib.get("number", str(m_idx + 1))
            m_num_s = m_num.strip()
            is_implicit = (
                m_el.attrib.get("implicit", "no").lower() == "yes"
                or m_num_s == "0"
                or m_num_s.upper().startswith("X")
            )

            cursor = Fraction(0)
            measure_max_cursor = Fraction(0)
            last_note_onset = Fraction(0)
            barline_left = "regular"
            barline_right = "regular"
            repeat_start = False
            repeat_end = False
            volta: str | None = None

            notes: list[CERNote] = []
            note_contexts: list[CERContext] = []
            rests: list[CERRest] = []
            raw_forwards: list[tuple[int, Fraction, Fraction, str | None, int | None]] = []
            seg_idx = 0
            seg_voices: dict[int, set[str]] = defaultdict(set)
            seg_voice_staves: dict[tuple[int, str], set[int]] = defaultdict(set)
            measure_voice_staves: dict[str, set[int]] = defaultdict(set)
            directions: list[CERDirection] = []
            context_changes: list[tuple[int, Fraction, str, str]] = []

            for child in m_el:
                tag = _local_tag(child.tag)

                if tag == "print":
                    if (
                        child.attrib.get("new-system", "no").lower() == "yes"
                        or child.attrib.get("new-page", "no").lower() == "yes"
                    ):
                        part_system_breaks.append(m_idx)

                elif tag == "attributes":
                    is_mid_context = initialized_context or _snap_rational(cursor) > 0

                    div_txt = _child_text(child, "divisions")
                    if div_txt:
                        parsed_div = _parse_fraction(div_txt)
                        divisions = parsed_div if parsed_div > 0 else Fraction(0)

                    staves_txt = _child_text(child, "staves")
                    if staves_txt:
                        num_staves = max(1, _parse_int(staves_txt, 1))
                        part_max_staff = max(part_max_staff, num_staves)
                        for s_i in range(1, num_staves + 1):
                            staff_clefs.setdefault(s_i, "F4" if s_i == 2 else "G2")
                            staff_ottava.setdefault(s_i, 0)
                            staff_time.setdefault(s_i, part_time)
                            staff_senza_misura.setdefault(s_i, part_senza_misura)

                    for tr_el in _iter_children(child, "transpose"):
                        chrom = _parse_int(_child_text(tr_el, "chromatic", "0"), 0)
                        oct_c = _parse_int(_child_text(tr_el, "octave-change", "0"), 0)
                        shift = chrom + 12 * oct_c
                        tr_num_txt = tr_el.attrib.get("number")
                        if tr_num_txt:
                            tr_s = max(1, _parse_int(tr_num_txt, 1))
                            staff_transpose[tr_s] = shift
                        else:
                            part_transpose = shift
                            staff_transpose.clear()

                    for key_el in _iter_children(child, "key"):
                        fifths_txt = _child_text(key_el, "fifths")
                        if fifths_txt:
                            new_fifths = _parse_int(fifths_txt, 0)
                            if is_mid_context and new_fifths != key_fifths:
                                k_num_txt = key_el.attrib.get("number")
                                k_staff = max(1, _parse_int(k_num_txt, 1)) if k_num_txt else 0
                                context_changes.append(
                                    (k_staff, _snap_rational(cursor), "key", str(new_fifths))
                                )
                            key_fifths = new_fifths

                    for time_el in _iter_children(child, "time"):
                        t_num_txt = time_el.attrib.get("number")
                        t_staff = max(1, _parse_int(t_num_txt, 1)) if t_num_txt else 0
                        if _find_child(time_el, "senza-misura") is not None:
                            if t_staff > 0:
                                staff_senza_misura[t_staff] = True
                                part_max_staff = max(part_max_staff, t_staff)
                            else:
                                part_senza_misura = True
                                for s_i in range(1, max(num_staves, part_max_staff) + 1):
                                    staff_senza_misura[s_i] = True
                            continue

                        beats_els = _iter_children(time_el, "beats")
                        btype_els = _iter_children(time_el, "beat-type")
                        if beats_els and btype_els:
                            pairs: list[tuple[int, int]] = []
                            for b_el, bt_el in zip(beats_els, btype_els):
                                b_txt = (b_el.text or "").strip()
                                bt_txt = (bt_el.text or "").strip()
                                if not b_txt or not bt_txt:
                                    continue
                                b_parts = [_parse_int(tok, 0) for tok in b_txt.split("+") if tok.strip()]
                                b_sum = sum(b_parts) if b_parts else 0
                                bt_val = _parse_int(bt_txt, 0)
                                if b_sum <= 0 or bt_val <= 0:
                                    raise ValueError(f"Invalid time signature {b_sum}/{bt_val}.")
                                pairs.append((b_sum, bt_val))
                            if not pairs:
                                continue
                            if len(pairs) == 1:
                                new_num, new_den = pairs[0]
                            else:
                                common_den = math.lcm(*(d for _, d in pairs))
                                new_num = sum(n * (common_den // d) for n, d in pairs)
                                new_den = common_den

                            if t_staff > 0:
                                prev_t = staff_time.get(t_staff, part_time)
                                if is_mid_context and (new_num, new_den) != prev_t:
                                    context_changes.append(
                                        (t_staff, _snap_rational(cursor), "time", f"{new_num}/{new_den}")
                                    )
                                staff_time[t_staff] = (new_num, new_den)
                                staff_senza_misura[t_staff] = False
                                part_max_staff = max(part_max_staff, t_staff)
                            else:
                                if is_mid_context and (new_num, new_den) != part_time:
                                    context_changes.append(
                                        (0, _snap_rational(cursor), "time", f"{new_num}/{new_den}")
                                    )
                                part_time = (new_num, new_den)
                                part_senza_misura = False
                                for s_i in range(1, max(num_staves, part_max_staff) + 1):
                                    staff_time[s_i] = part_time
                                    staff_senza_misura[s_i] = False

                    for clef_el in _iter_children(child, "clef"):
                        s_num = max(1, _parse_int(clef_el.attrib.get("number", "1"), 1))
                        sign = _child_text(clef_el, "sign", "G")
                        line = _child_text(clef_el, "line", "2")
                        oct_chg_txt = _child_text(clef_el, "clef-octave-change")
                        oct_chg = _parse_int(oct_chg_txt, 0) if oct_chg_txt else 0
                        clef_str = f"{sign}{line}({oct_chg:+d}o)" if oct_chg != 0 else f"{sign}{line}"
                        prev_clef = staff_clefs.get(s_num)
                        if (
                            is_mid_context
                            and s_num in staff_clef_initialized
                            and prev_clef is not None
                            and clef_str != prev_clef
                        ):
                            context_changes.append((s_num, _snap_rational(cursor), "clef", clef_str))
                        staff_clefs[s_num] = clef_str
                        staff_clef_initialized.add(s_num)
                        staff_ottava.setdefault(s_num, 0)
                        staff_time.setdefault(s_num, part_time)
                        staff_senza_misura.setdefault(s_num, part_senza_misura)
                        part_max_staff = max(part_max_staff, s_num)

                elif tag == "backup":
                    dur_txt = _child_text(child, "duration")
                    if dur_txt and divisions > 0:
                        raw_ticks = max(Fraction(0), _parse_fraction(dur_txt))
                        delta = raw_ticks / (4 * divisions)
                        cursor = _snap_rational(max(Fraction(0), cursor - delta))
                    seg_idx += 1

                elif tag == "forward":
                    dur_txt = _child_text(child, "duration")
                    if dur_txt and divisions > 0:
                        fwd_ticks = max(Fraction(0), _parse_fraction(dur_txt))
                        fwd_onset = _snap_rational(cursor)
                        fwd_raw = fwd_ticks / (4 * divisions)
                        fwd_dur = _snap_rational(fwd_raw)
                        if fwd_ticks > 0:
                            fwd_voice_txt = _child_text(child, "voice")
                            fwd_voice = fwd_voice_txt if fwd_voice_txt else None
                            fwd_staff_txt = _child_text(child, "staff")
                            fwd_staff = max(1, _parse_int(fwd_staff_txt, 1)) if fwd_staff_txt else None
                            if fwd_staff is not None:
                                part_max_staff = max(part_max_staff, fwd_staff)
                            raw_forwards.append((seg_idx, fwd_onset, fwd_dur, fwd_voice, fwd_staff))
                        cursor = _snap_rational(cursor + fwd_raw)
                        if cursor > measure_max_cursor:
                            measure_max_cursor = cursor

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
                    dir_staff = max(1, _parse_int(_child_text(child, "staff", "1"), 1))
                    part_max_staff = max(part_max_staff, dir_staff)
                    offset_txt = _child_text(child, "offset")
                    dir_onset = _snap_rational(cursor)
                    if offset_txt and divisions > 0:
                        dir_onset = _snap_rational(
                            max(
                                Fraction(0),
                                cursor + _parse_fraction(offset_txt) / (4 * divisions),
                            )
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
                                w_num = dt_child.attrib.get("number", "1")
                                w_key = (dir_staff, w_num)
                                if w_type in ("crescendo", "diminuendo"):
                                    open_wedges[w_key] = (w_type, m_idx, dir_onset)
                                elif w_type == "stop" and w_key in open_wedges:
                                    w_kind, start_m_idx, w_start = open_wedges.pop(w_key)
                                    if start_m_idx == m_idx:
                                        directions.append(
                                            CERDirection(
                                                "wedge",
                                                w_kind,
                                                dir_staff,
                                                w_start,
                                                max(dir_onset, w_start),
                                            )
                                        )
                                    else:
                                        prev_m = part_measures[start_m_idx]
                                        part_measures[start_m_idx] = replace(
                                            prev_m,
                                            directions=prev_m.directions
                                            + (
                                                CERDirection(
                                                    "wedge", w_kind, dir_staff, w_start, None
                                                ),
                                            ),
                                        )
                            elif dt_tag == "octave-shift":
                                o_type = dt_child.attrib.get("type", "").lower()
                                if o_type == "continue":
                                    continue
                                o_size = _parse_int(dt_child.attrib.get("size", "8"), 8)
                                oct_val = 1 if o_size <= 8 else 2
                                if o_type == "down":
                                    new_ott = oct_val  # 8va alta shifts written notes down
                                elif o_type == "up":
                                    new_ott = -oct_val  # 8vb bassa shifts written notes up
                                else:
                                    new_ott = 0
                                prev_ott = staff_ottava.get(dir_staff, 0)
                                if new_ott != prev_ott:
                                    context_changes.append(
                                        (dir_staff, dir_onset, "ottava", str(new_ott))
                                    )
                                staff_ottava[dir_staff] = new_ott
                            elif dt_tag == "metronome":
                                pm = _nfc(_child_text(dt_child, "per-minute"))
                                beat_unit = _nfc(_child_text(dt_child, "beat-unit"))
                                bu_dots = len(_iter_children(dt_child, "beat-unit-dot"))
                                if pm:
                                    t_val = (
                                        f"{beat_unit}{'.' * bu_dots}={pm}" if beat_unit else pm
                                    )
                                    directions.append(
                                        CERDirection("tempo", t_val, dir_staff, dir_onset)
                                    )
                            elif dt_tag == "rehearsal":
                                r_txt = _nfc((dt_child.text or "").strip())
                                if r_txt:
                                    directions.append(
                                        CERDirection("rehearsal", r_txt, dir_staff, dir_onset)
                                    )
                            elif dt_tag == "words":
                                w_txt = _nfc((dt_child.text or "").strip())
                                if w_txt:
                                    directions.append(
                                        CERDirection("words", w_txt, dir_staff, dir_onset)
                                    )

                elif tag == "harmony":
                    h_staff = max(1, _parse_int(_child_text(child, "staff", "1"), 1))
                    part_max_staff = max(part_max_staff, h_staff)
                    root_el = _find_child(child, "root")
                    root_step = _nfc(_child_text(root_el, "root-step")) if root_el is not None else ""
                    root_alt_raw = _child_text(root_el, "root-alter") if root_el is not None else ""
                    root_alt = ""
                    if root_alt_raw:
                        alt_frac = _parse_fraction(root_alt_raw)
                        if alt_frac != 0:
                            root_alt = str(int(alt_frac)) if alt_frac.denominator == 1 else str(float(alt_frac))
                    kind_txt = _nfc(_child_text(child, "kind", "major"))
                    if root_step:
                        h_val = f"{root_step}{root_alt}:{kind_txt}"
                        directions.append(
                            CERDirection("harmony", h_val, h_staff, _snap_rational(cursor))
                        )

                elif tag == "note":
                    initialized_context = True
                    is_grace = _find_child(child, "grace") is not None
                    is_chord = _find_child(child, "chord") is not None
                    is_cue = _find_child(child, "cue") is not None
                    rest_el = _find_child(child, "rest")
                    pitch_el = _find_child(child, "pitch")
                    unpitched_el = _find_child(child, "unpitched")

                    voice = _child_text(child, "voice", "1")
                    staff = max(1, _parse_int(_child_text(child, "staff", "1"), 1))
                    part_max_staff = max(part_max_staff, staff)
                    staff_clefs.setdefault(staff, "F4" if staff == 2 else "G2")
                    staff_ottava.setdefault(staff, 0)
                    staff_time.setdefault(staff, part_time)
                    staff_senza_misura.setdefault(staff, part_senza_misura)

                    cur_time_num, cur_time_den = staff_time[staff]

                    dur_txt = _child_text(child, "duration")
                    if is_grace:
                        raw_dur = Fraction(0)
                    elif dur_txt and divisions > 0:
                        raw_dur = max(Fraction(0), _parse_fraction(dur_txt) / (4 * divisions))
                    else:
                        raw_dur = _duration_from_type(child)
                    dur = _snap_rational(raw_dur)

                    onset = (
                        _snap_rational(cursor)
                        if is_grace
                        else (last_note_onset if is_chord else _snap_rational(cursor))
                    )
                    seg_voices[seg_idx].add(voice)
                    seg_voice_staves[(seg_idx, voice)].add(staff)
                    measure_voice_staves[voice].add(staff)

                    if rest_el is not None:
                        is_full = rest_el.attrib.get("measure", "no").lower() == "yes"
                        expected_m_dur = Fraction(cur_time_num, cur_time_den)
                        if is_full and (dur == 0 or dur == Fraction(1, 1)):
                            dur = expected_m_dur
                            raw_dur = expected_m_dur
                        rest_artics: list[str] = []
                        for not_el in _iter_children(child, "notations"):
                            for sub_not in not_el:
                                if _local_tag(sub_not.tag) == "fermata":
                                    rest_artics.append("fermata")
                        rests.append(
                            CERRest(
                                staff=staff,
                                voice=voice,
                                onset=onset,
                                duration=dur,
                                is_full_measure=is_full,
                                articulations=tuple(sorted(rest_artics)),
                            )
                        )
                    elif pitch_el is not None or unpitched_el is not None:
                        if pitch_el is not None:
                            step = _child_text(pitch_el, "step", "C").upper()
                            if step not in STEP_TO_SEMITONE:
                                raise ValueError(f"Invalid pitch step {step!r}.")
                            alt_frac = _parse_fraction(_child_text(pitch_el, "alter", "0"))
                            alter: int | float = (
                                int(alt_frac) if alt_frac.denominator == 1 else float(alt_frac)
                            )
                            octave = _parse_int(_child_text(pitch_el, "octave", "4"), 4)
                            tr_shift = staff_transpose.get(staff, part_transpose)
                            midi = (
                                int(round((octave + 1) * 12 + STEP_TO_SEMITONE[step] + float(alt_frac)))
                                + tr_shift
                            )
                        else:
                            assert unpitched_el is not None
                            step = _child_text(unpitched_el, "display-step", "C").upper()
                            if step not in STEP_TO_SEMITONE:
                                step = "C"
                            alter = 0
                            octave = _parse_int(_child_text(unpitched_el, "display-octave", "4"), 4)
                            midi = (octave + 1) * 12 + STEP_TO_SEMITONE[step]

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
                                tuplet = (_parse_int(act_txt, 1), _parse_int(norm_txt, 1))

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
                                    elif t_type == "continue":
                                        tie_start = True
                                        tie_stop = True

                        lyrics_list: list[str] = []
                        for lyr_el in _iter_children(child, "lyric"):
                            syl = _nfc(_child_text(lyr_el, "text"))
                            if syl:
                                lyrics_list.append(syl)

                        ctx = CERContext(
                            clef=staff_clefs[staff],
                            key_fifths=key_fifths,
                            time_num=cur_time_num,
                            time_den=cur_time_den,
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
                        last_note_onset = onset
                        cursor += raw_dur
                        snapped_cur = _snap_rational(cursor)
                        if snapped_cur > measure_max_cursor:
                            measure_max_cursor = snapped_cur

            part_measure_max_cursors.append(measure_max_cursor)

            # Resolve <forward> voice and staff attribution within <backup>-delimited segments
            resolved_forwards: list[CERForward] = []
            for f_seg, f_onset, f_dur, f_voice, f_staff in raw_forwards:
                voice_res = f_voice
                if voice_res is None:
                    s_voices = seg_voices.get(f_seg, set())
                    if len(s_voices) == 1:
                        voice_res = next(iter(s_voices))
                staff_res = f_staff
                if staff_res is None:
                    if voice_res is not None:
                        sv_staves = seg_voice_staves.get((f_seg, voice_res)) or measure_voice_staves.get(voice_res)
                        if sv_staves and len(sv_staves) == 1:
                            staff_res = next(iter(sv_staves))
                        else:
                            staff_res = 1
                    else:
                        staff_res = 1
                resolved_forwards.append(
                    CERForward(
                        staff=staff_res,
                        voice=voice_res,
                        onset=f_onset,
                        duration=f_dur,
                    )
                )

            measure_contexts = {
                s_i: CERContext(
                    clef=staff_clefs.get(s_i, "F4" if s_i == 2 else "G2"),
                    key_fifths=key_fifths,
                    time_num=staff_time.get(s_i, part_time)[0],
                    time_den=staff_time.get(s_i, part_time)[1],
                    ottava=staff_ottava.get(s_i, 0),
                )
                for s_i in range(1, max(num_staves, part_max_staff) + 1)
            }
            any_senza = part_senza_misura or any(
                staff_senza_misura.get(s_i, False)
                for s_i in range(1, max(num_staves, part_max_staff) + 1)
            )
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
                    forwards=tuple(resolved_forwards),
                    directions=tuple(directions),
                    is_senza_misura=any_senza,
                )
            )

        # Flush any wedges still open at the end of the part using measure_max_cursor
        for (w_staff, _w_num), (w_kind, start_m_idx, w_start) in open_wedges.items():
            end_cur = max(part_measure_max_cursors[start_m_idx], w_start)
            pm = part_measures[start_m_idx]
            part_measures[start_m_idx] = replace(
                pm,
                directions=pm.directions
                + (CERDirection("wedge", w_kind, w_staff, w_start, end_cur),),
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


def parse_musicxml_file(path: str | Path) -> CERScore:
    """Parse a MusicXML (.xml, .musicxml) or compressed (.mxl) file into a CERScore."""
    p = Path(path)
    if zipfile.is_zipfile(p):
        with zipfile.ZipFile(p, "r") as zf:
            target: str | None = None
            names = zf.namelist()
            if "META-INF/container.xml" in names:
                try:
                    c_root = ET.fromstring(zf.read("META-INF/container.xml").decode("utf-8-sig"))
                    for elem in c_root.iter():
                        if _local_tag(elem.tag) == "rootfile":
                            fp = elem.attrib.get("full-path", "").strip()
                            if fp and fp in names:
                                target = fp
                                break
                except (ET.ParseError, UnicodeDecodeError):
                    target = None
            if target is None:
                for name in names:
                    low = name.lower()
                    if not low.startswith("meta-inf/") and (
                        low.endswith(".musicxml") or low.endswith(".xml")
                    ):
                        target = name
                        break
            if target is None:
                raise ValueError(f"No MusicXML rootfile found inside archive {p}.")
            xml_str = zf.read(target).decode("utf-8-sig")
            return parse_musicxml(xml_str)
    return parse_musicxml(p.read_text(encoding="utf-8-sig"))


def _measure_similarity(m_pred: CERMeasure, m_gt: CERMeasure) -> float:
    """Measure similarity: 0.7 * Dice(multiset(p, d)) + 0.3 * barline_match when Dice > 0."""
    c_pred: Counter[Hashable] = Counter((n.midi, n.duration) for n in m_pred.notes)
    c_gt: Counter[Hashable] = Counter((n.midi, n.duration) for n in m_gt.notes)
    total = sum(c_pred.values()) + sum(c_gt.values())
    if total == 0:
        r_pred: Counter[Hashable] = Counter(r.duration for r in m_pred.rests)
        r_gt: Counter[Hashable] = Counter(r.duration for r in m_gt.rests)
        r_total = sum(r_pred.values()) + sum(r_gt.values())
        if r_total == 0:
            dice = 1.0
        else:
            dice = (2.0 * _multiset_intersection_size(r_pred, r_gt)) / r_total
    else:
        dice = (2.0 * _multiset_intersection_size(c_pred, c_gt)) / total
    if dice <= 0.0:
        return 0.0
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


def _measure_stream_sums(m: CERMeasure) -> dict[tuple[int, str], Fraction]:
    """Compute beat sums per (staff, voice) stream in a measure."""
    by_sv: dict[tuple[int, str], Fraction] = defaultdict(Fraction)
    for n in m.notes:
        if not n.is_chord and not n.grace:
            by_sv[(n.staff, n.voice)] += n.duration
    for r in m.rests:
        by_sv[(r.staff, r.voice)] += r.duration
    for f in m.forwards:
        if f.voice is not None and (f.staff, f.voice) in by_sv:
            by_sv[(f.staff, f.voice)] += f.duration
    return by_sv


def _is_gt_bsvr_exempt(mg: CERMeasure, gt_part_len: int = 1) -> bool:
    """Return True if a GT measure is legitimately exempt from full-measure beat-sum checks."""
    if mg.is_implicit or mg.is_senza_misura:
        return True
    if gt_part_len >= 2 and mg.index in (0, gt_part_len - 1):
        gt_by_sv = _measure_stream_sums(mg)
        non_zero = [(staff, dur_sum) for (staff, _v), dur_sum in gt_by_sv.items() if dur_sum > 0]
        if non_zero:
            all_partial = True
            for staff, dur_sum in non_zero:
                ctx = mg.contexts.get(staff) or next(iter(mg.contexts.values()), CERContext())
                expected = Fraction(ctx.time_num, ctx.time_den)
                if dur_sum >= expected:
                    all_partial = False
                    break
            if all_partial:
                return True
    return False


def _measure_bsvr_counts(
    mp: CERMeasure,
    mg: CERMeasure | None = None,
    *,
    gt_part_len: int = 1,
    standalone: bool = False,
) -> tuple[int, int]:
    """Return (valid_streams, total_streams) for one predicted measure.

    When aligned to GT (standalone=False), exemption is determined strictly by
    the aligned GT measure (preventing gaming via spurious implicit="yes").
    Non-exempt completely empty predicted measures count as invalid (0/1)
    unless the aligned GT measure is also empty.
    """
    if standalone:
        if mp.is_implicit or mp.is_senza_misura:
            return 0, 0
    else:
        if mg is not None and _is_gt_bsvr_exempt(mg, gt_part_len=gt_part_len):
            return 0, 0

    if len(mp.notes) == 0 and len(mp.rests) == 0:
        if not standalone and mg is not None and len(mg.notes) == 0 and len(mg.rests) == 0:
            return 0, 0
        if standalone:
            return 0, 1
        return 0, 1

    by_sv = _measure_stream_sums(mp)
    valid = 0
    total = 0
    for (staff, _voice), dur_sum in by_sv.items():
        if dur_sum == 0:
            continue
        ctx = mp.contexts.get(staff) or next(iter(mp.contexts.values()), CERContext())
        expected = Fraction(ctx.time_num, ctx.time_den)
        total += 1
        if dur_sum == expected:
            valid += 1
    return valid, total


def _compute_bsvr_counts(measures: Sequence[CERMeasure]) -> tuple[int, int]:
    """Return (valid_streams, total_streams) for standalone Beat-Sum Validity Rate."""
    valid = 0
    total = 0
    for m in measures:
        v_i, t_i = _measure_bsvr_counts(m, standalone=True)
        valid += v_i
        total += t_i
    return valid, total


def _compute_bsvr(measures: Sequence[CERMeasure]) -> float:
    """Beat-Sum Validity Rate across predicted (measure, staff, voice) streams."""
    valid, total = _compute_bsvr_counts(measures)
    if total == 0:
        return 1.0
    return valid / total


def _extract_sounding_events(score: CERScore) -> Counter[Hashable]:
    """Resolve ties and unroll measure onsets into sounding MIDI events.

    - Uses actual duration for implicit/senza-misura/partial measures.
    - Keys cross-staff ties by canonical base part_id (stripping '_sN') so
      cross-staff ties in grand-staff parts chain cleanly.
    - Requires contiguous onset (start_on + acc_dur == abs_onset) to chain tie_stop.
    - Flushes overwritten active ties instead of dropping the earlier note.
    """
    events: Counter[Hashable] = Counter()

    # Compute synchronized measure durations per (base_part_id, measure_index)
    base_m_nominal: dict[tuple[str, int], Fraction] = {}
    base_m_actual: dict[tuple[str, int], Fraction] = defaultdict(Fraction)
    base_m_partial_flag: dict[tuple[str, int], bool] = defaultdict(bool)
    base_m_max_idx: dict[str, int] = defaultdict(int)

    for m in score.measures:
        bpid = _base_part_id(m.part_id)
        key_bm = (bpid, m.index)
        if m.index > base_m_max_idx[bpid]:
            base_m_max_idx[bpid] = m.index
        ctx0 = next(iter(m.contexts.values()), CERContext())
        nom = Fraction(ctx0.time_num, ctx0.time_den)
        base_m_nominal.setdefault(key_bm, nom)
        if m.is_implicit or m.is_senza_misura:
            base_m_partial_flag[key_bm] = True
        max_end = Fraction(0)
        for n in m.notes:
            if not n.grace:
                end = n.onset + n.duration
                if end > max_end:
                    max_end = end
        for r in m.rests:
            end = r.onset + r.duration
            if end > max_end:
                max_end = end
        for f in m.forwards:
            end = f.onset + f.duration
            if end > max_end:
                max_end = end
        if max_end > base_m_actual[key_bm]:
            base_m_actual[key_bm] = max_end

    # Compute cumulative measure start offsets per part_id
    part_offset: dict[str, Fraction] = defaultdict(Fraction)
    notes_by_base: dict[str, list[tuple[Fraction, int, CERNote]]] = defaultdict(list)

    for m in score.measures:
        bpid = _base_part_id(m.part_id)
        key_bm = (bpid, m.index)
        m_start = part_offset[m.part_id]
        nom_len = base_m_nominal[key_bm]
        act_len = base_m_actual[key_bm]
        is_partial_bar = base_m_partial_flag[key_bm] or (
            base_m_max_idx[bpid] >= 1
            and m.index in (0, base_m_max_idx[bpid])
            and Fraction(0) < act_len < nom_len
        )
        m_len = act_len if (is_partial_bar and act_len > 0) else nom_len

        for n in m.notes:
            if n.grace:
                continue
            notes_by_base[bpid].append((m_start + n.onset, m.index, n))

        part_offset[m.part_id] = m_start + m_len

    # Process notes per base part in chronological order so cross-staff ties chain in any direction
    for bpid, note_entries in notes_by_base.items():
        note_entries.sort(key=lambda item: (item[0], item[1], 0 if item[2].tie_stop else 1))
        active_ties: dict[tuple[str, str, int], tuple[Fraction, Fraction]] = {}

        for abs_onset, _m_idx, n in note_entries:
            key = (bpid, n.voice, n.midi)
            if n.tie_stop and key in active_ties:
                start_on, acc_dur = active_ties[key]
                if start_on + acc_dur == abs_onset:
                    active_ties.pop(key)
                    total_dur = acc_dur + n.duration
                    if n.tie_start:
                        active_ties[key] = (start_on, total_dur)
                    else:
                        events[(n.midi, start_on, total_dur)] += 1
                    continue
                # Non-contiguous tie_stop: flush stale active tie if it ended earlier
                if start_on + acc_dur < abs_onset:
                    active_ties.pop(key)
                    events[(n.midi, start_on, acc_dur)] += 1

            if n.tie_start:
                if key in active_ties:
                    old_start, old_dur = active_ties.pop(key)
                    events[(n.midi, old_start, old_dur)] += 1
                active_ties[key] = (abs_onset, n.duration)
            else:
                events[(n.midi, abs_onset, n.duration)] += 1

        for (_, _, midi), (start_on, acc_dur) in active_ties.items():
            events[(midi, start_on, acc_dur)] += 1

    return events


def _has_repetition_loop(
    pred_measures: Sequence[CERMeasure],
    gt_measures: Sequence[CERMeasure],
    alignment: Sequence[tuple[int | None, int | None]] | None = None,
    part_match_map: dict[str, str] | None = None,
) -> bool:
    """Detect runaway decoder repetition loops while sparing genuine musical ostinatos.

    Flags a predicted part stream when a non-empty measure n-gram
    (2 <= n <= min(8, len // 3)) repeats >= 3x consecutively and either:
      - extends beyond the corresponding GT repetition with runaway length inflation, or
      - is absent in GT and is NOT explained by a genuine GT ostinato (either high
        aligned measure similarity >= 0.70 or a matching >= 3x consecutive n-gram
        ostinato structure in GT without runaway measure inflation).
    """
    def _sig(m: CERMeasure) -> tuple[tuple[int, Fraction], ...]:
        return tuple(sorted((n.midi, n.duration) for n in m.notes))

    def _group_by_part(
        measures: Sequence[CERMeasure],
    ) -> list[tuple[str, list[tuple[int, CERMeasure]]]]:
        grouped: dict[str, list[tuple[int, CERMeasure]]] = {}
        order: list[str] = []
        for idx, m in enumerate(measures):
            if m.part_id not in grouped:
                grouped[m.part_id] = []
                order.append(m.part_id)
            grouped[m.part_id].append((idx, m))
        return [(pid, grouped[pid]) for pid in order]

    def _has_any_triple(sigs: Sequence[tuple[tuple[int, Fraction], ...]], n_gram: int) -> bool:
        if len(sigs) < 3 * n_gram:
            return False
        for i in range(len(sigs) - 3 * n_gram + 1):
            chunk = tuple(sigs[i : i + n_gram])
            if not any(chunk):
                continue
            if (
                tuple(sigs[i + n_gram : i + 2 * n_gram]) == chunk
                and tuple(sigs[i + 2 * n_gram : i + 3 * n_gram]) == chunk
            ):
                return True
        return False

    def _max_chunk_run(
        sigs: Sequence[tuple[tuple[int, Fraction], ...]],
        chunk: tuple[tuple[tuple[int, Fraction], ...], ...],
        n_gram: int,
    ) -> int:
        best = 0
        for i in range(len(sigs) - n_gram + 1):
            if tuple(sigs[i : i + n_gram]) == chunk:
                reps = 1
                while (
                    i + (reps + 1) * n_gram <= len(sigs)
                    and tuple(sigs[i + reps * n_gram : i + (reps + 1) * n_gram]) == chunk
                ):
                    reps += 1
                if reps > best:
                    best = reps
        return best

    pred_parts = _group_by_part(pred_measures)
    gt_parts = _group_by_part(gt_measures)
    gt_parts_by_id: dict[str, list[tuple[int, CERMeasure]]] = dict(gt_parts)

    # Map global pred measure index -> aligned global gt measure index (or None)
    pred_to_gt: dict[int, int | None] = {}
    if alignment is not None:
        for p_idx, g_idx in alignment:
            if p_idx is not None:
                pred_to_gt[p_idx] = g_idx

    all_gt_sigs = [_sig(m) for m in gt_measures]

    for p_part_idx, (pid, p_items) in enumerate(pred_parts):
        if len(p_items) < 6:
            continue
        p_sigs = [_sig(m) for _, m in p_items]
        if part_match_map is not None and pid in part_match_map:
            matched_gid = part_match_map[pid]
            g_items = gt_parts_by_id.get(matched_gid, [])
            g_sigs = [_sig(m) for _, m in g_items]
        elif part_match_map is not None:
            g_items = []
            g_sigs = []
        elif p_part_idx < len(gt_parts):
            g_items = gt_parts[p_part_idx][1]
            g_sigs = [_sig(m) for _, m in g_items]
        else:
            g_items = []
            g_sigs = all_gt_sigs
        gt_part_len = len(g_sigs)

        max_n = min(8, len(p_sigs) // 3)
        for n_gram in range(2, max_n + 1):
            if len(p_sigs) < 3 * n_gram:
                continue
            gt_triples = {
                tuple(g_sigs[k : k + 3 * n_gram])
                for k in range(max(0, len(g_sigs) - 3 * n_gram + 1))
            }
            gt_has_ostinato = _has_any_triple(g_sigs, n_gram)
            runaway_len = (
                gt_part_len > 0
                and len(p_sigs) > max(gt_part_len + 1, int(1.15 * gt_part_len))
            )

            for i in range(len(p_sigs) - 3 * n_gram + 1):
                chunk = tuple(p_sigs[i : i + n_gram])
                if not any(chunk):  # ignore all-empty measures
                    continue
                if not (
                    tuple(p_sigs[i + n_gram : i + 2 * n_gram]) == chunk
                    and tuple(p_sigs[i + 2 * n_gram : i + 3 * n_gram]) == chunk
                ):
                    continue

                triple = tuple(p_sigs[i : i + 3 * n_gram])
                if triple in gt_triples:
                    # Exact GT ostinato; only flag if pred runs away far beyond GT's run
                    if runaway_len:
                        pred_reps = 3
                        while (
                            i + (pred_reps + 1) * n_gram <= len(p_sigs)
                            and tuple(
                                p_sigs[i + pred_reps * n_gram : i + (pred_reps + 1) * n_gram]
                            )
                            == chunk
                        ):
                            pred_reps += 1
                        gt_reps = _max_chunk_run(g_sigs, chunk, n_gram)
                        if pred_reps > gt_reps:
                            return True
                    continue

                # Candidate loop not verbatim in GT: distinguish genuine ostinato from loop
                span_len = 3 * n_gram
                no_measure_inflation = gt_part_len > 0 and len(p_sigs) <= max(
                    gt_part_len, int(1.15 * gt_part_len)
                )

                aligned_count = 0
                sim_sum = 0.0
                for offset in range(span_len):
                    p_global_idx, p_meas = p_items[i + offset]
                    if alignment is not None:
                        g_global_idx = pred_to_gt.get(p_global_idx)
                        g_meas = gt_measures[g_global_idx] if g_global_idx is not None else None
                    elif i + offset < len(g_items):
                        g_meas = g_items[i + offset][1]
                    else:
                        g_meas = None
                    if g_meas is not None:
                        aligned_count += 1
                        sim_sum += _measure_similarity(p_meas, g_meas)

                aligned_frac = aligned_count / span_len
                mean_sim = sim_sum / span_len

                # Genuine ostinato with minor read error or subtle GT variation
                if aligned_frac >= (5.0 / 6.0) and (
                    (mean_sim >= 0.70 and no_measure_inflation)
                    or (gt_has_ostinato and no_measure_inflation)
                ):
                    continue

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


def _prune_empty_gt_parts(gt: CERScore) -> CERScore:
    """Prune completely empty GT parts (0 notes across all measures) when GT has notes.

    Prevents hidden/tacet empty staves from inflating m_gt, MER, wFPP, or
    triggering false structural_collapse while preserving empty/negative control pages.
    """
    if gt.total_notes == 0:
        return gt

    part_has_notes: dict[str, bool] = {}
    for m in gt.measures:
        if m.part_id not in part_has_notes:
            part_has_notes[m.part_id] = False
        if len(m.notes) > 0:
            part_has_notes[m.part_id] = True

    if all(part_has_notes.values()):
        return gt

    kept_parts = tuple(pid for pid in gt.parts if part_has_notes.get(pid, False))
    kept_set = set(kept_parts)

    # Preserve local visual system break measure indices even if they were annotated on a pruned part
    break_local_indices = {
        gt.measures[b].index
        for b in gt.system_breaks
        if 0 <= b < len(gt.measures) and gt.measures[b].index > 0
    }

    new_measures: list[CERMeasure] = []
    new_breaks: list[int] = []
    for m in gt.measures:
        if m.part_id in kept_set:
            if m.index in break_local_indices:
                new_breaks.append(len(new_measures))
            new_measures.append(m)

    return CERScore(
        parts=kept_parts,
        measures=tuple(new_measures),
        num_staves=max(1, len(kept_parts)),
        system_breaks=tuple(sorted(set(new_breaks))),
    )


def _align_parts_and_measures(
    pred_measures: Sequence[CERMeasure],
    gt_measures: Sequence[CERMeasure],
    gap_penalty: float = 0.25,
) -> tuple[list[tuple[int | None, int | None]], dict[str, str]]:
    """Stage 1: Hungarian part-level bijection + per-part Needleman-Wunsch measure DP."""
    def _group(measures: Sequence[CERMeasure]) -> list[tuple[str, list[tuple[int, CERMeasure]]]]:
        grouped: dict[str, list[tuple[int, CERMeasure]]] = {}
        order: list[str] = []
        for idx, m in enumerate(measures):
            if m.part_id not in grouped:
                grouped[m.part_id] = []
                order.append(m.part_id)
            grouped[m.part_id].append((idx, m))
        return [(pid, grouped[pid]) for pid in order]

    pred_parts = _group(pred_measures)
    gt_parts = _group(gt_measures)

    if not pred_parts and not gt_parts:
        return [], {}
    if not pred_parts:
        return [(None, g_idx) for g_idx in range(len(gt_measures))], {}
    if not gt_parts:
        return [(p_idx, None) for p_idx in range(len(pred_measures))], {}

    def _part_note_counter(items: list[tuple[int, CERMeasure]]) -> Counter[Hashable]:
        c: Counter[Hashable] = Counter()
        for _, m in items:
            for n in m.notes:
                c[(n.midi, n.duration)] += 1
        return c

    def _part_rest_counter(items: list[tuple[int, CERMeasure]]) -> Counter[Hashable]:
        c: Counter[Hashable] = Counter()
        for _, m in items:
            for r in m.rests:
                c[r.duration] += 1
        return c

    def _part_initial_clef(items: list[tuple[int, CERMeasure]]) -> str:
        if not items:
            return ""
        m0 = items[0][1]
        ctx0 = m0.contexts.get(1) or next(iter(m0.contexts.values()), CERContext())
        return ctx0.clef

    p_notes = [_part_note_counter(items) for _, items in pred_parts]
    g_notes = [_part_note_counter(items) for _, items in gt_parts]
    p_rests = [_part_rest_counter(items) for _, items in pred_parts]
    g_rests = [_part_rest_counter(items) for _, items in gt_parts]
    p_clefs = [_part_initial_clef(items) for _, items in pred_parts]
    g_clefs = [_part_initial_clef(items) for _, items in gt_parts]

    weights: list[list[int]] = []
    for pi in range(len(pred_parts)):
        row: list[int] = []
        for gi in range(len(gt_parts)):
            tot_n = sum(p_notes[pi].values()) + sum(g_notes[gi].values())
            if tot_n > 0:
                sim = (2.0 * _multiset_intersection_size(p_notes[pi], g_notes[gi])) / tot_n
            else:
                tot_r = sum(p_rests[pi].values()) + sum(g_rests[gi].values())
                if tot_r > 0:
                    sim = (2.0 * _multiset_intersection_size(p_rests[pi], g_rests[gi])) / tot_r
                else:
                    sim = 1.0
            clef_bonus = 0.05 if p_clefs[pi] == g_clefs[gi] else 0.0
            pos_bonus = 0.001 if pi == gi else 0.0
            row.append(int(round((0.949 * sim + clef_bonus + pos_bonus) * 10000)))
        weights.append(row)

    assignment = _hungarian_max_weight(weights)
    matched_gt_indices: set[int] = set()
    part_match_map: dict[str, str] = {}
    alignment: list[tuple[int | None, int | None]] = []

    for pi, gi in enumerate(assignment):
        p_pid, p_items = pred_parts[pi]
        if gi >= 0:
            g_pid, g_items = gt_parts[gi]
            matched_gt_indices.add(gi)
            part_match_map[p_pid] = g_pid
            local_aln = align_measures(
                [m for _, m in p_items],
                [m for _, m in g_items],
                gap_penalty=gap_penalty,
            )
            for li, lj in local_aln:
                p_global = p_items[li][0] if li is not None else None
                g_global = g_items[lj][0] if lj is not None else None
                alignment.append((p_global, g_global))
        else:
            for p_global, _ in p_items:
                alignment.append((p_global, None))

    for gi, (_g_pid, g_items) in enumerate(gt_parts):
        if gi not in matched_gt_indices:
            for g_global, _ in g_items:
                alignment.append((None, g_global))

    return alignment, part_match_map


def align_and_score_cer(pred: CERScore, gt: CERScore) -> CERAlignmentResult:
    """Run full 2-stage hierarchical alignment and compute all 4 OMR-NS axes."""
    gt_eff = _prune_empty_gt_parts(gt)
    alignment, part_match_map = _align_parts_and_measures(
        pred.measures, gt_eff.measures, gap_penalty=0.25
    )

    gt_part_lengths: dict[str, int] = Counter(m.part_id for m in gt_eff.measures)

    matched_t1a = 0
    matched_t1b = 0
    matched_t1c = 0
    matched_rests = 0
    matched_barlines = 0
    exact_measures = 0
    ctx_matched = 0
    ctx_total = 0
    measure_ctx_mismatches = 0
    bsvr_valid = 0
    bsvr_total = 0

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

    # Layer 3 raw error-bucket accumulators
    err_layout = 0.0
    err_pitch = 0.0
    err_rhythm = 0.0
    err_poly = 0.0
    err_halluc = 0.0

    for p_idx, g_idx in alignment:
        mp = pred.measures[p_idx] if p_idx is not None else None
        mg = gt_eff.measures[g_idx] if g_idx is not None else None

        if mg is None and mp is not None:
            # Extra hallucinated predicted measure
            v_m, t_m = _measure_bsvr_counts(mp, None)
            bsvr_valid += v_m
            bsvr_total += t_m
            extra_edit = 5.0 + 2.5 * len(mp.notes) + 1.0 * len(mp.rests)
            total_wfpp += extra_edit
            err_halluc += 1.0 + len(mp.notes) + len(mp.rests)
            expr_counts["ctx"][1] += len(mp.context_changes)
            expr_counts["art"][1] += sum(len(n.articulations) for n in mp.notes) + sum(
                len(r.articulations) for r in mp.rests
            )
            expr_counts["dyn"][1] += sum(1 for d in mp.directions if d.kind in ("dynamic", "wedge"))
            expr_counts["tie_slur"][1] += sum(
                (1 if n.tie_start else 0) + n.slur_starts for n in mp.notes
            )
            expr_counts["text"][1] += sum(len(n.lyrics) for n in mp.notes) + sum(
                1 for d in mp.directions if d.kind in ("harmony", "tempo", "rehearsal", "words")
            )
            continue

        if mg is not None and mp is None:
            # Missing GT measure
            m_tc = _measure_transcribe_cost(mg)
            total_transcribe += m_tc
            total_wfpp += compute_effective_measure_cost(5.0 + m_tc, m_tc)
            err_layout += 1.0 + len(mg.notes) + len(mg.rests)
            gt_measure_matched_notes[g_idx] = 0  # type: ignore[index]
            expr_counts["ctx"][2] += len(mg.context_changes)
            expr_counts["art"][2] += sum(len(n.articulations) for n in mg.notes) + sum(
                len(r.articulations) for r in mg.rests
            )
            expr_counts["dyn"][2] += sum(1 for d in mg.directions if d.kind in ("dynamic", "wedge"))
            expr_counts["tie_slur"][2] += sum(
                (1 if n.tie_start else 0) + n.slur_starts for n in mg.notes
            )
            expr_counts["text"][2] += sum(len(n.lyrics) for n in mg.notes) + sum(
                1 for d in mg.directions if d.kind in ("harmony", "tempo", "rehearsal", "words")
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
        else:
            err_layout += 1.0

        # Measure-level active context check (independent of pitch match)
        p_mctx = mp.contexts.get(1) or next(iter(mp.contexts.values()), CERContext())
        g_mctx = mg.contexts.get(1) or next(iter(mg.contexts.values()), CERContext())
        if p_mctx != g_mctx:
            measure_ctx_mismatches += 1

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

        # Tier L1a: Sounding (midi, onset)
        c_p_t1a: Counter[Hashable] = Counter((n.midi, n.onset) for n in mp.notes)
        c_g_t1a: Counter[Hashable] = Counter((n.midi, n.onset) for n in mg.notes)
        m_t1a = _multiset_intersection_size(c_p_t1a, c_g_t1a)
        matched_t1a += m_t1a

        # Tier L1b: Rhythmic (midi, onset, duration, grace, tuplet)
        c_p_t1b: Counter[Hashable] = Counter(
            (n.midi, n.onset, n.duration, n.grace, n.tuplet) for n in mp.notes
        )
        c_g_t1b: Counter[Hashable] = Counter(
            (n.midi, n.onset, n.duration, n.grace, n.tuplet) for n in mg.notes
        )
        m_t1b = _multiset_intersection_size(c_p_t1b, c_g_t1b)
        matched_t1b += m_t1b

        # Tier L1c: Strict Engraving (L1b + spelling + staff + mapped voice + cue; chord-order invariant)
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

        # Separate enharmonic spelling mismatches from polyphony/voice routing mismatches
        c_p_spell: Counter[Hashable] = Counter(
            (n.midi, n.onset, n.duration, n.grace, n.tuplet, n.step, n.alter, n.octave)
            for n in mp.notes
        )
        c_g_spell: Counter[Hashable] = Counter(
            (n.midi, n.onset, n.duration, n.grace, n.tuplet, n.step, n.alter, n.octave)
            for n in mg.notes
        )
        m_spell = _multiset_intersection_size(c_p_spell, c_g_spell)
        spell_err = max(0, m_t1b - m_spell)
        poly_err = max(0, m_spell - m_t1c)
        if gt_eff.total_notes == 0 and len(mp.notes) > 0:
            err_halluc += float(len(mp.notes))
        else:
            err_pitch += max(0, (len(mp.notes) - m_t1a) + (len(mg.notes) - m_t1a)) + spell_err
        err_rhythm += max(0, m_t1a - m_t1b)
        err_poly += poly_err

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
        err_rhythm += max(0, (len(mp.rests) - m_rest) + (len(mg.rests) - m_rest))

        # Measure-Exact Rate (MER) & Aligned-GT-aware BSVR
        v_m, t_m = _measure_bsvr_counts(
            mp, mg, gt_part_len=gt_part_lengths.get(mg.part_id, 1)
        )
        bsvr_valid += v_m
        bsvr_total += t_m
        m_bsvr = (v_m / t_m) if t_m > 0 else 1.0
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

        # 2. Articulations & ornaments (Bucket-A chord-member invariant: keyed on staff, mapped voice, onset, artic)
        c_p_art: Counter[Hashable] = Counter(
            (n.staff, voice_map_by_staff.get(n.staff, {}).get(n.voice, n.voice), n.onset, a)
            for n in mp.notes
            for a in n.articulations
        )
        for r in mp.rests:
            mv = voice_map_by_staff.get(r.staff, {}).get(r.voice, r.voice)
            for a in r.articulations:
                c_p_art[(r.staff, mv, r.onset, a)] += 1
        c_g_art: Counter[Hashable] = Counter(
            (n.staff, n.voice, n.onset, a) for n in mg.notes for a in n.articulations
        )
        for r in mg.rests:
            for a in r.articulations:
                c_g_art[(r.staff, r.voice, r.onset, a)] += 1
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

        # 5. Textual & symbolic layers (lyrics, harmony, tempo, rehearsal, words)
        c_p_txt: Counter[Hashable] = Counter(
            ("lyr", n.staff, n.onset, syl) for n in mp.notes for syl in n.lyrics
        )
        for d in mp.directions:
            if d.kind in ("harmony", "tempo", "rehearsal", "words"):
                c_p_txt[(d.kind, d.staff, d.onset, d.value)] += 1
        c_g_txt: Counter[Hashable] = Counter(
            ("lyr", n.staff, n.onset, syl) for n in mg.notes for syl in n.lyrics
        )
        for d in mg.directions:
            if d.kind in ("harmony", "tempo", "rehearsal", "words"):
                c_g_txt[(d.kind, d.staff, d.onset, d.value)] += 1
        expr_counts["text"][0] += _multiset_intersection_size(c_p_txt, c_g_txt)
        expr_counts["text"][1] += sum(c_p_txt.values())
        expr_counts["text"][2] += sum(c_g_txt.values())

        # Per-measure edit cost (NEU) capped by Rational Musician Re-Entry Cap
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
    n_gt = gt_eff.total_notes
    m_pred = pred.total_measures
    m_gt = gt_eff.total_measures

    f1_t1a = _f1_from_counts(matched_t1a, n_pred, n_gt)
    f1_t1b = _f1_from_counts(matched_t1b, n_pred, n_gt)
    f1_t1c = _f1_from_counts(matched_t1c, n_pred, n_gt)
    total_pred_rests = sum(len(m.rests) for m in pred.measures)
    total_gt_rests = sum(len(m.rests) for m in gt_eff.measures)
    f1_rest = _f1_from_counts(matched_rests, total_pred_rests, total_gt_rests)

    mer = (exact_measures / m_gt) if m_gt > 0 else (1.0 if m_pred == 0 else 0.0)
    f1_bar = _f1_from_counts(matched_barlines, m_pred, m_gt)
    bsvr = (bsvr_valid / bsvr_total) if bsvr_total > 0 else 1.0
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
    play_gt = _extract_sounding_events(gt_eff)
    play_matched = _multiset_intersection_size(play_pred, play_gt)
    s_play = _f1_from_counts(play_matched, sum(play_pred.values()), sum(play_gt.values()))

    # Dropped visual system count (aggregated across all parts/staves by within-part measure index)
    k_dropped = 0
    max_local_idx = max((m.index for m in gt_eff.measures), default=-1)
    if max_local_idx >= 0:
        local_breaks = sorted(
            {
                gt_eff.measures[b].index
                for b in gt_eff.system_breaks
                if 0 <= b < len(gt_eff.measures) and gt_eff.measures[b].index > 0
            }
        )
        starts = [0] + local_breaks
        for idx_s, s_start in enumerate(starts):
            s_end = starts[idx_s + 1] if idx_s + 1 < len(starts) else (max_local_idx + 1)
            sys_gt_notes = 0
            sys_matched = 0
            for g_i, mg in enumerate(gt_eff.measures):
                if s_start <= mg.index < s_end:
                    sys_gt_notes += len(mg.notes)
                    sys_matched += gt_measure_matched_notes.get(g_i, 0)
            if sys_gt_notes == 0:
                continue
            if (sys_matched / sys_gt_notes) < 0.15:
                k_dropped += 1

    rep_loop = _has_repetition_loop(
        pred.measures,
        gt_eff.measures,
        alignment=alignment,
        part_match_map=part_match_map,
    )

    # Layer 3 8-bucket error taxonomy percentages
    err_layout += 4.0 * k_dropped
    err_ctx = (
        float(ctx_total - ctx_matched)
        + float(measure_ctx_mismatches)
        + float(expr_counts["ctx"][1] + expr_counts["ctx"][2] - 2 * expr_counts["ctx"][0])
    )
    err_rhythm += float(bsvr_total - bsvr_valid)
    err_expr = float(
        sum(
            expr_counts[k][1] + expr_counts[k][2] - 2 * expr_counts[k][0]
            for k in ("art", "dyn", "tie_slur", "text")
        )
    )
    if rep_loop:
        err_halluc += 4.0

    raw_buckets = {
        "layout_segmentation_pct": max(0.0, err_layout),
        "global_context_drift_pct": max(0.0, err_ctx),
        "pitch_accidental_pct": max(0.0, err_pitch),
        "rhythm_meter_pct": max(0.0, err_rhythm),
        "polyphony_routing_pct": max(0.0, err_poly),
        "engraving_expressive_pct": max(0.0, err_expr),
        "hallucination_truncation_pct": max(0.0, err_halluc),
        "annotation_contamination_pct": 0.0,
    }
    raw_total = sum(raw_buckets.values())
    if raw_total > 0.0:
        error_buckets = {
            k: round((100.0 * raw_buckets[k]) / raw_total, 4) for k in ERROR_BUCKET_KEYS
        }
    else:
        error_buckets = {k: 0.0 for k in ERROR_BUCKET_KEYS}

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
        error_buckets=error_buckets,
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
    gt_score = _prune_empty_gt_parts(parse_musicxml(gt_xml))
    gt_tc = sum(_measure_transcribe_cost(m) for m in gt_score.measures)

    try:
        pred_score = parse_musicxml(pred_xml)
    except (ET.ParseError, ValueError, TypeError):
        unparseable_buckets = {
            k: (100.0 if k == "hallucination_truncation_pct" else 0.0)
            for k in ERROR_BUCKET_KEYS
        }
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
            error_buckets=unparseable_buckets,
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
        error_buckets=res.error_buckets,
        latency_s=latency_s,
    )
