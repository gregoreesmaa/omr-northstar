"""Adversarial stress-test scorers and parametric MusicXML trap generators.

Zero external dependencies (Python 3.10+ standard library only).
Implements the executable gate evaluators and Private Parametric Trap Generator
specified in docs/05-adversarial-stress-tests.md.
"""

from __future__ import annotations

import random
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

# ---------------------------------------------------------------------------
# Gate thresholds (docs/05-adversarial-stress-tests.md)
# ---------------------------------------------------------------------------

OSTINATO_TSR_MIN = 0.95
COUNTERFACTUAL_PHR_MAX = 0.02
COUNTERFACTUAL_F1_MIN = 0.95
POSITIONAL_DECAY_RATIO_MIN = 0.95
ORIENTATION_MAX_DROP = 0.01
SPANNER_F1_MIN = 0.95
STAFF_TRACKING_ACC_MIN = 0.98
ANNOTATION_DELTA_F1_MAX = 0.03
ANNOTATION_FALSE_NOTE_RATE_MAX = 0.005

_EPS = 1e-9

_ALTER_TO_SYMBOL: dict[int, str] = {-2: "bb", -1: "b", 0: "", 1: "#", 2: "##"}
_ALTER_TO_ACCIDENTAL: dict[int, str] = {
    -2: "flat-flat",
    -1: "flat",
    0: "natural",
    1: "sharp",
    2: "double-sharp",
}

_OSTINATO_PATTERNS: tuple[tuple[tuple[str, int, int], ...], ...] = (
    # (step, alter, octave) x 8 eighth notes = 4/4 bar
    (
        ("C", 0, 4),
        ("E", 0, 4),
        ("G", 0, 4),
        ("C", 0, 5),
        ("B", 0, 4),
        ("G", 0, 4),
        ("E", 0, 4),
        ("D", 0, 4),
    ),
    (
        ("A", 0, 3),
        ("C", 0, 4),
        ("E", 0, 4),
        ("A", 0, 4),
        ("G", 0, 4),
        ("E", 0, 4),
        ("C", 0, 4),
        ("B", 0, 3),
    ),
    (
        ("D", 0, 4),
        ("F", 1, 4),
        ("A", 0, 4),
        ("D", 0, 5),
        ("C", 1, 5),
        ("A", 0, 4),
        ("F", 1, 4),
        ("E", 0, 4),
    ),
)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OstinatoTrapResult:
    """Result of Pillar 1A Ostinato Spot-the-Diff evaluation."""

    tsr: float
    exact_bar_count: bool
    copy_loop_rate: float
    passed: bool
    gate_failures: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class CounterfactualTrapResult:
    """Result of Pillar 1B Counterfactual Classics evaluation."""

    phr: float
    mutation_site_f1: float
    reverted_count: int
    matched_count: int
    total_sites: int
    passed: bool
    gate_failures: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class PositionalDecayResult:
    """Result of Pillar 1C Long-Context Positional Decay evaluation."""

    q4_q1_ratio: float
    system_completeness: float
    passed: bool
    gate_failures: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class NegativeControlResult:
    """Result of Pillar 3B Non-Music Negative Control evaluation."""

    pred_note_count: int
    page_type: str
    passed: bool
    gate_failures: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class OrientationInvarianceResult:
    """Result of Pillar 3B Orientation Invariance evaluation."""

    max_drop: float
    drops_by_angle: dict[int, float]
    passed: bool
    gate_failures: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class MultipageContinuityResult:
    """Result of Pillar 3C Cross-Page Spanner & French-Score Staff Tracking."""

    spanner_f1: float
    staff_tracking_acc: float
    passed: bool
    gate_failures: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class AnnotationInvarianceResult:
    """Result of Pillar 2C Annotated-Score Invariance evaluation."""

    delta_f1: float
    false_note_rate: float
    passed: bool
    gate_failures: tuple[str, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Pillar 1-3 Stress-Test Scorers
# ---------------------------------------------------------------------------


def score_ostinato_trap(
    pred_mutated_sites: Mapping[Any, Any] | Sequence[Any] | set[Any] | frozenset[Any],
    gt_mutated_sites: Mapping[Any, Any] | Sequence[Any] | set[Any] | frozenset[Any],
    pred_bar_count: int,
    gt_bar_count: int,
    has_copy_loop: bool | float,
) -> OstinatoTrapResult:
    """Evaluate Pillar 1A Ostinato Spot-the-Diff trap survival.

    Gates:
      - tsr >= 0.95
      - exact_bar_count == True
      - copy_loop_rate == 0.0
    """
    if pred_bar_count < 0 or gt_bar_count < 0:
        raise ValueError("Bar counts must be non-negative.")

    if isinstance(gt_mutated_sites, Mapping):
        if not isinstance(pred_mutated_sites, Mapping):
            raise TypeError(
                "pred_mutated_sites must be a Mapping when gt_mutated_sites is a Mapping."
            )
        if not gt_mutated_sites:
            tsr = 1.0
        else:
            hits = sum(
                1 for k, v in gt_mutated_sites.items() if pred_mutated_sites.get(k) == v
            )
            tsr = hits / len(gt_mutated_sites)
    elif isinstance(gt_mutated_sites, (set, frozenset)) or isinstance(
        pred_mutated_sites, (set, frozenset)
    ):
        gt_set = set(gt_mutated_sites)
        pred_set = set(pred_mutated_sites)
        tsr = 1.0 if not gt_set else len(pred_set & gt_set) / len(gt_set)
    else:
        gt_seq = list(gt_mutated_sites)
        pred_seq = list(pred_mutated_sites)
        if not gt_seq:
            tsr = 1.0
        else:
            hits = sum(1 for p, g in zip(pred_seq, gt_seq) if p == g)
            tsr = hits / len(gt_seq)

    exact_bar_count = pred_bar_count == gt_bar_count
    if isinstance(has_copy_loop, bool):
        copy_loop_rate = 1.0 if has_copy_loop else 0.0
    else:
        copy_loop_rate = float(has_copy_loop)
        if not (0.0 <= copy_loop_rate <= 1.0):
            raise ValueError("copy_loop_rate must be in [0.0, 1.0].")

    failures: list[str] = []
    if tsr < OSTINATO_TSR_MIN - _EPS:
        failures.append(f"tsr={tsr:.4f} < {OSTINATO_TSR_MIN:.2f}")
    if not exact_bar_count:
        failures.append(f"bar_count_mismatch(pred={pred_bar_count}, gt={gt_bar_count})")
    if copy_loop_rate > 0.0 + _EPS:
        failures.append(f"copy_loop_rate={copy_loop_rate:.4f} > 0.0")

    return OstinatoTrapResult(
        tsr=tsr,
        exact_bar_count=exact_bar_count,
        copy_loop_rate=copy_loop_rate,
        passed=len(failures) == 0,
        gate_failures=tuple(failures),
    )


def score_counterfactual_trap(
    pred_at_sites: Sequence[Any],
    original_at_sites: Sequence[Any],
    mutated_at_sites: Sequence[Any],
) -> CounterfactualTrapResult:
    """Evaluate Pillar 1B Counterfactual Classics memorization probe.

    Gates:
      - phr <= 0.02
      - mutation_site_f1 >= 0.95
    """
    n = len(mutated_at_sites)
    if len(pred_at_sites) != n or len(original_at_sites) != n:
        raise ValueError(
            "pred_at_sites, original_at_sites, and mutated_at_sites must have equal length."
        )
    if n == 0:
        return CounterfactualTrapResult(
            phr=0.0,
            mutation_site_f1=1.0,
            reverted_count=0,
            matched_count=0,
            total_sites=0,
            passed=True,
            gate_failures=(),
        )

    reverted = sum(
        1
        for p, orig, mut in zip(pred_at_sites, original_at_sites, mutated_at_sites)
        if p == orig and p != mut
    )
    matched = sum(1 for p, mut in zip(pred_at_sites, mutated_at_sites) if p == mut)
    n_pred_present = sum(1 for p in pred_at_sites if p is not None)

    phr = reverted / n
    mutation_site_f1 = (
        (2.0 * matched) / (n_pred_present + n) if (n_pred_present + n) > 0 else 0.0
    )

    failures: list[str] = []
    if phr > COUNTERFACTUAL_PHR_MAX + _EPS:
        failures.append(f"phr={phr:.4f} > {COUNTERFACTUAL_PHR_MAX:.2f}")
    if mutation_site_f1 < COUNTERFACTUAL_F1_MIN - _EPS:
        failures.append(
            f"mutation_site_f1={mutation_site_f1:.4f} < {COUNTERFACTUAL_F1_MIN:.2f}"
        )

    return CounterfactualTrapResult(
        phr=phr,
        mutation_site_f1=mutation_site_f1,
        reverted_count=reverted,
        matched_count=matched,
        total_sites=n,
        passed=len(failures) == 0,
        gate_failures=tuple(failures),
    )


def score_positional_decay(
    q1_f1: float,
    q4_f1: float,
    pred_systems: int,
    gt_systems: int,
) -> PositionalDecayResult:
    """Evaluate Pillar 1C Long-Context Positional Decay & System Completeness.

    Gates:
      - q4_q1_ratio >= 0.95
      - system_completeness == 1.0
    """
    if q1_f1 < 0.0 or q4_f1 < 0.0:
        raise ValueError("F1 scores must be non-negative.")
    if pred_systems < 0 or gt_systems < 0:
        raise ValueError("System counts must be non-negative.")

    q4_q1_ratio = 0.0 if q1_f1 <= 0.0 else (q4_f1 / q1_f1)
    if gt_systems == 0:
        system_completeness = 1.0 if pred_systems == 0 else 0.0
    else:
        system_completeness = min(pred_systems, gt_systems) / max(
            pred_systems, gt_systems
        )

    failures: list[str] = []
    if q4_q1_ratio < POSITIONAL_DECAY_RATIO_MIN - _EPS:
        failures.append(
            f"q4_q1_ratio={q4_q1_ratio:.4f} < {POSITIONAL_DECAY_RATIO_MIN:.2f}"
        )
    if abs(system_completeness - 1.0) > _EPS:
        failures.append(
            f"system_completeness={system_completeness:.4f} != 1.0 "
            f"(pred={pred_systems}, gt={gt_systems})"
        )

    return PositionalDecayResult(
        q4_q1_ratio=q4_q1_ratio,
        system_completeness=system_completeness,
        passed=len(failures) == 0,
        gate_failures=tuple(failures),
    )


def score_negative_control(
    pred_note_count: int,
    page_type: str,
) -> NegativeControlResult:
    """Evaluate Pillar 3B Non-Music Negative Control rejection.

    Gate:
      - pred_note_count == 0
    """
    if pred_note_count < 0:
        raise ValueError("pred_note_count must be non-negative.")

    passed = pred_note_count == 0
    failures = (
        ()
        if passed
        else (f"hallucinated_notes={pred_note_count} on {page_type}",)
    )
    return NegativeControlResult(
        pred_note_count=pred_note_count,
        page_type=page_type,
        passed=passed,
        gate_failures=failures,
    )


def score_orientation_invariance(
    f1_0: float,
    f1_90: float,
    f1_180: float,
    f1_270: float,
) -> OrientationInvarianceResult:
    """Evaluate Pillar 3B Orientation Invariance across 90/180/270-degree rotations.

    Gate:
      - max_drop <= 0.01
    """
    for val in (f1_0, f1_90, f1_180, f1_270):
        if not (0.0 <= val <= 1.0):
            raise ValueError("F1 scores must lie in [0.0, 1.0].")

    drops = {
        90: max(0.0, f1_0 - f1_90),
        180: max(0.0, f1_0 - f1_180),
        270: max(0.0, f1_0 - f1_270),
    }
    max_drop = max(drops.values())
    failures = (
        ()
        if max_drop <= ORIENTATION_MAX_DROP + _EPS
        else (f"max_drop={max_drop:.4f} > {ORIENTATION_MAX_DROP:.2f}",)
    )
    return OrientationInvarianceResult(
        max_drop=max_drop,
        drops_by_angle=drops,
        passed=len(failures) == 0,
        gate_failures=failures,
    )


def score_multipage_continuity(
    matched_spanners: int,
    pred_spanners: int,
    gt_spanners: int,
    correct_staff_ids: int,
    total_staff_ids: int,
) -> MultipageContinuityResult:
    """Evaluate Pillar 3C Cross-Page Spanner Closure & French-Score Staff Tracking.

    Gates:
      - spanner_f1 >= 0.95
      - staff_tracking_acc >= 0.98
    """
    if min(
        matched_spanners,
        pred_spanners,
        gt_spanners,
        correct_staff_ids,
        total_staff_ids,
    ) < 0:
        raise ValueError("Counts must be non-negative.")
    if matched_spanners > pred_spanners or matched_spanners > gt_spanners:
        raise ValueError("matched_spanners cannot exceed pred_spanners or gt_spanners.")
    if correct_staff_ids > total_staff_ids:
        raise ValueError("correct_staff_ids cannot exceed total_staff_ids.")

    denom = pred_spanners + gt_spanners
    spanner_f1 = 1.0 if denom == 0 else (2.0 * matched_spanners) / denom
    staff_tracking_acc = (
        1.0 if total_staff_ids == 0 else correct_staff_ids / total_staff_ids
    )

    failures: list[str] = []
    if spanner_f1 < SPANNER_F1_MIN - _EPS:
        failures.append(f"spanner_f1={spanner_f1:.4f} < {SPANNER_F1_MIN:.2f}")
    if staff_tracking_acc < STAFF_TRACKING_ACC_MIN - _EPS:
        failures.append(
            f"staff_tracking_acc={staff_tracking_acc:.4f} < {STAFF_TRACKING_ACC_MIN:.2f}"
        )

    return MultipageContinuityResult(
        spanner_f1=spanner_f1,
        staff_tracking_acc=staff_tracking_acc,
        passed=len(failures) == 0,
        gate_failures=tuple(failures),
    )


def score_annotation_invariance(
    clean_f1: float,
    annotated_f1: float,
    false_note_rate: float,
) -> AnnotationInvarianceResult:
    """Evaluate Pillar 2C Annotated-Score Invariance.

    Gates:
      - delta_f1 <= 0.03
      - false_note_rate <= 0.005
    """
    if not (0.0 <= clean_f1 <= 1.0 and 0.0 <= annotated_f1 <= 1.0):
        raise ValueError("F1 scores must lie in [0.0, 1.0].")
    if not (0.0 <= false_note_rate <= 1.0):
        raise ValueError("false_note_rate must lie in [0.0, 1.0].")

    delta_f1 = max(0.0, clean_f1 - annotated_f1)
    failures: list[str] = []
    if delta_f1 > ANNOTATION_DELTA_F1_MAX + _EPS:
        failures.append(f"delta_f1={delta_f1:.4f} > {ANNOTATION_DELTA_F1_MAX:.3f}")
    if false_note_rate > ANNOTATION_FALSE_NOTE_RATE_MAX + _EPS:
        failures.append(
            f"false_note_rate={false_note_rate:.4f} > {ANNOTATION_FALSE_NOTE_RATE_MAX:.4f}"
        )

    return AnnotationInvarianceResult(
        delta_f1=delta_f1,
        false_note_rate=false_note_rate,
        passed=len(failures) == 0,
        gate_failures=tuple(failures),
    )


# ---------------------------------------------------------------------------
# Pillar 4: Parametric MusicXML Trap Generator
# ---------------------------------------------------------------------------


def _format_pitch(step: str, alter: int, octave: int) -> str:
    return f"{step}{_ALTER_TO_SYMBOL.get(alter, str(alter))}{octave}"


def _build_note_element(
    step: str,
    alter: int,
    octave: int,
    duration: int,
    note_type: str,
    *,
    show_accidental: bool = False,
    voice: str = "1",
) -> ET.Element:
    note_el = ET.Element("note")
    pitch_el = ET.SubElement(note_el, "pitch")
    ET.SubElement(pitch_el, "step").text = step
    if alter != 0:
        ET.SubElement(pitch_el, "alter").text = str(alter)
    ET.SubElement(pitch_el, "octave").text = str(octave)
    ET.SubElement(note_el, "duration").text = str(duration)
    ET.SubElement(note_el, "voice").text = voice
    ET.SubElement(note_el, "type").text = note_type
    if show_accidental:
        ET.SubElement(note_el, "accidental").text = _ALTER_TO_ACCIDENTAL.get(
            alter, "natural"
        )
    return note_el


def generate_ostinato_trap_musicxml(
    n_bars: int = 16,
    seed: int = 42,
) -> tuple[str, dict[str, Any]]:
    """Generate a deterministic Ostinato Spot-the-Diff MusicXML trap score.

    Produces ``n_bars`` identical 4/4 arpeggio bars except for 2-3 seeded
    1-symbol mutations (accidental flips and eighth->two-sixteenth rhythm
    splits), returning ``(musicxml_str, trap_metadata)`` with exact
    ground-truth mutation site coordinates.
    """
    if n_bars < 4:
        raise ValueError("n_bars must be at least 4 to construct an ostinato trap.")

    rng = random.Random(seed)
    base_pattern = rng.choice(_OSTINATO_PATTERNS)

    # Pick 2 or 3 distinct bars (never bar 1 so the ostinato is established first).
    n_mutations = rng.choice((2, 3)) if n_bars >= 4 else 2
    mutated_bars = sorted(rng.sample(range(2, n_bars + 1), n_mutations))

    # Guarantee both mutation types appear when n_mutations >= 2.
    mutation_types = ["accidental_flip", "rhythm_split"]
    if n_mutations == 3:
        mutation_types.append(rng.choice(("accidental_flip", "rhythm_split")))
    rng.shuffle(mutation_types)

    bar_mutation_plan: dict[int, tuple[int, str]] = {}
    for bar_num, mtype in zip(mutated_bars, mutation_types):
        note_idx = rng.randrange(len(base_pattern))
        bar_mutation_plan[bar_num] = (note_idx, mtype)

    score = ET.Element("score-partwise", {"version": "4.0"})
    part_list = ET.SubElement(score, "part-list")
    score_part = ET.SubElement(part_list, "score-part", {"id": "P1"})
    ET.SubElement(score_part, "part-name").text = "Ostinato Trap"
    part = ET.SubElement(score, "part", {"id": "P1"})

    mutations_meta: list[dict[str, Any]] = []

    for bar_num in range(1, n_bars + 1):
        measure = ET.SubElement(part, "measure", {"number": str(bar_num)})
        if bar_num == 1:
            attrs = ET.SubElement(measure, "attributes")
            ET.SubElement(attrs, "divisions").text = "4"
            key = ET.SubElement(attrs, "key")
            ET.SubElement(key, "fifths").text = "0"
            time = ET.SubElement(attrs, "time")
            ET.SubElement(time, "beats").text = "4"
            ET.SubElement(time, "beat-type").text = "4"
            clef = ET.SubElement(attrs, "clef")
            ET.SubElement(clef, "sign").text = "G"
            ET.SubElement(clef, "line").text = "2"

        plan = bar_mutation_plan.get(bar_num)
        for note_idx, (step, alter, octave) in enumerate(base_pattern):
            if plan is not None and plan[0] == note_idx:
                mtype = plan[1]
                orig_pitch = _format_pitch(step, alter, octave)
                if mtype == "accidental_flip":
                    candidates = [a for a in (-1, 0, 1) if a != alter]
                    new_alter = rng.choice(candidates)
                    new_pitch = _format_pitch(step, new_alter, octave)
                    measure.append(
                        _build_note_element(
                            step,
                            new_alter,
                            octave,
                            duration=2,
                            note_type="eighth",
                            show_accidental=True,
                        )
                    )
                    mutations_meta.append(
                        {
                            "measure": bar_num,
                            "note_index": note_idx,
                            "mutation_type": "accidental_flip",
                            "original": f"{orig_pitch}:eighth",
                            "mutated": f"{new_pitch}:eighth",
                        }
                    )
                else:  # rhythm_split: 1 eighth -> 2 sixteenths (preserves 4/4 bar sum)
                    measure.append(
                        _build_note_element(
                            step,
                            alter,
                            octave,
                            duration=1,
                            note_type="16th",
                            show_accidental=(alter != 0),
                        )
                    )
                    measure.append(
                        _build_note_element(
                            step,
                            alter,
                            octave,
                            duration=1,
                            note_type="16th",
                        )
                    )
                    mutations_meta.append(
                        {
                            "measure": bar_num,
                            "note_index": note_idx,
                            "mutation_type": "rhythm_split",
                            "original": f"{orig_pitch}:eighth",
                            "mutated": f"{orig_pitch}:16th+{orig_pitch}:16th",
                        }
                    )
            else:
                measure.append(
                    _build_note_element(
                        step,
                        alter,
                        octave,
                        duration=2,
                        note_type="eighth",
                        show_accidental=(alter != 0),
                    )
                )

    ET.indent(score, space="  ")
    xml_str = ET.tostring(score, encoding="unicode", xml_declaration=True)
    metadata: dict[str, Any] = {
        "trap_type": "ostinato_spot_the_diff",
        "seed": seed,
        "n_bars": n_bars,
        "base_pattern": [_format_pitch(s, a, o) for s, a, o in base_pattern],
        "mutated_bars": mutated_bars,
        "mutations": mutations_meta,
    }
    return xml_str, metadata


def generate_counterfactual_musicxml(
    base_xml_str: str,
    mutation_rate: float = 0.06,
    seed: int = 42,
) -> tuple[str, list[dict[str, Any]]]:
    """Deterministically mutate pitched notes in any MusicXML ``<score-partwise>``.

    Alters ``round(N * mutation_rate)`` (at least 1 when ``N > 0``) pitched notes
    by flipping an accidental or shifting an octave, returning
    ``(mutated_xml_str, mutation_sites)`` for computing exact Prior-Hallucination
    Rate (PHR).
    """
    if not (0.0 < mutation_rate <= 1.0):
        raise ValueError("mutation_rate must be in (0.0, 1.0].")

    root = ET.fromstring(base_xml_str)
    pitched_entries: list[tuple[str, str, int, int, ET.Element]] = []

    global_idx = 0
    for part in root.findall("part"):
        part_id = part.attrib.get("id", "P1")
        for measure in part.findall("measure"):
            measure_num = measure.attrib.get("number", "1")
            note_idx = 0
            for note in measure.findall("note"):
                pitch = note.find("pitch")
                if pitch is not None and note.find("rest") is None:
                    pitched_entries.append(
                        (part_id, measure_num, note_idx, global_idx, note)
                    )
                    global_idx += 1
                note_idx += 1

    if not pitched_entries:
        xml_str = ET.tostring(root, encoding="unicode", xml_declaration=True)
        return xml_str, []

    rng = random.Random(seed)
    n_mutate = max(1, min(len(pitched_entries), round(len(pitched_entries) * mutation_rate)))
    chosen_indices = sorted(rng.sample(range(len(pitched_entries)), n_mutate))

    mutation_sites: list[dict[str, Any]] = []
    for idx in chosen_indices:
        part_id, measure_num, note_idx, g_idx, note_el = pitched_entries[idx]
        pitch_el = note_el.find("pitch")
        assert pitch_el is not None

        step_el = pitch_el.find("step")
        oct_el = pitch_el.find("octave")
        if step_el is None or step_el.text is None or oct_el is None or oct_el.text is None:
            continue

        step = step_el.text.strip()
        alter_el = pitch_el.find("alter")
        orig_alter = int(alter_el.text.strip()) if (alter_el is not None and alter_el.text) else 0
        orig_oct = int(oct_el.text.strip())
        orig_pitch = _format_pitch(step, orig_alter, orig_oct)

        mtype = rng.choice(("accidental", "octave"))
        if mtype == "accidental":
            candidates = [a for a in (-1, 0, 1) if a != orig_alter]
            new_alter = rng.choice(candidates)
            if alter_el is None:
                alter_el = ET.Element("alter")
                # Insert <alter> immediately after <step> per MusicXML schema.
                step_pos = list(pitch_el).index(step_el)
                pitch_el.insert(step_pos + 1, alter_el)
            alter_el.text = str(new_alter)

            acc_el = note_el.find("accidental")
            if acc_el is None:
                acc_el = ET.SubElement(note_el, "accidental")
            acc_el.text = _ALTER_TO_ACCIDENTAL.get(new_alter, "natural")
            new_pitch = _format_pitch(step, new_alter, orig_oct)
        else:
            if orig_oct <= 2:
                delta_oct = 1
            elif orig_oct >= 6:
                delta_oct = -1
            else:
                delta_oct = rng.choice((-1, 1))
            new_oct = orig_oct + delta_oct
            oct_el.text = str(new_oct)
            new_pitch = _format_pitch(step, orig_alter, new_oct)

        mutation_sites.append(
            {
                "part_id": part_id,
                "measure": measure_num,
                "note_index": note_idx,
                "global_note_index": g_idx,
                "mutation_type": mtype,
                "original": orig_pitch,
                "mutated": new_pitch,
            }
        )

    xml_str = ET.tostring(root, encoding="unicode", xml_declaration=True)
    return xml_str, mutation_sites
