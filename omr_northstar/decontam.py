"""Blind Scrubbing Index for zero-leakage benchmark governance.

Zero external dependencies (Python 3.10+ standard library only).
Implements transposition-invariant and tempo/divisions-invariant symbolic
pitch-interval + IOI-ratio 8-gram hashing as specified in
docs/01-architecture-and-governance.md (Section 5.2).
"""

from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
from collections import defaultdict
from fractions import Fraction
from typing import Any, Mapping

DEFAULT_SCRUB_SALT = "OMR_NS_V1_BLIND_SCRUB"

_STEP_TO_SEMITONE: dict[str, int] = {
    "C": 0,
    "D": 2,
    "E": 4,
    "F": 5,
    "G": 7,
    "A": 9,
    "B": 11,
}


def _strip_ns(tag: str) -> str:
    """Strip optional XML namespace prefix from an ElementTree tag."""
    if "}" in tag:
        return tag.rsplit("}", 1)[1]
    return tag


def _find_child(elem: ET.Element, name: str) -> ET.Element | None:
    for child in elem:
        if _strip_ns(child.tag) == name:
            return child
    return None


def _iter_children(elem: ET.Element, name: str) -> list[ET.Element]:
    return [child for child in elem if _strip_ns(child.tag) == name]


def _parse_positive_fraction(text: str | None) -> Fraction | None:
    if not text:
        return None
    stripped = text.strip()
    if not stripped:
        return None
    try:
        val = Fraction(stripped)
    except (ValueError, ZeroDivisionError):
        return None
    return val if val > 0 else None


def _parse_nonneg_fraction(text: str | None) -> Fraction | None:
    if not text:
        return None
    stripped = text.strip()
    if not stripped:
        return None
    try:
        val = Fraction(stripped)
    except (ValueError, ZeroDivisionError):
        return None
    return val if val >= 0 else None


def _parse_alter(text: str | None) -> int:
    if not text:
        return 0
    stripped = text.strip()
    if not stripped:
        return 0
    try:
        return int(round(Fraction(stripped)))
    except (ValueError, ZeroDivisionError):
        return 0


def _extract_voice_note_streams(musicxml_str: str) -> list[list[tuple[Fraction, int]]]:
    """Extract ordered skyline (onset_beats, midi_pitch) streams per (part, staff, voice)."""
    root = ET.fromstring(musicxml_str)
    streams: list[list[tuple[Fraction, int]]] = []

    for part in _iter_children(root, "part"):
        divisions = Fraction(1)
        cursor = Fraction(0)
        last_onset = Fraction(0)
        voice_events: dict[tuple[str, str], dict[Fraction, int]] = defaultdict(dict)

        for measure in _iter_children(part, "measure"):
            measure_start = cursor
            measure_end = cursor
            for child in measure:
                tag = _strip_ns(child.tag)
                if tag == "attributes":
                    div_el = _find_child(child, "divisions")
                    parsed_div = _parse_positive_fraction(
                        div_el.text if div_el is not None else None
                    )
                    if parsed_div is not None:
                        divisions = parsed_div
                elif tag == "backup":
                    dur_el = _find_child(child, "duration")
                    dur_val = _parse_nonneg_fraction(
                        dur_el.text if dur_el is not None else None
                    )
                    if dur_val is not None:
                        cursor = max(measure_start, cursor - (dur_val / divisions))
                        last_onset = cursor
                elif tag == "forward":
                    dur_el = _find_child(child, "duration")
                    dur_val = _parse_nonneg_fraction(
                        dur_el.text if dur_el is not None else None
                    )
                    if dur_val is not None:
                        cursor += dur_val / divisions
                        last_onset = cursor
                        if cursor > measure_end:
                            measure_end = cursor
                elif tag == "note":
                    is_grace = _find_child(child, "grace") is not None
                    is_chord = _find_child(child, "chord") is not None
                    dur_el = _find_child(child, "duration")
                    dur_val = (
                        _parse_nonneg_fraction(dur_el.text)
                        if (dur_el is not None and not is_grace)
                        else None
                    )
                    dur = (dur_val / divisions) if dur_val is not None else Fraction(0)
                    if is_chord:
                        onset = last_onset
                    else:
                        onset = cursor
                        last_onset = cursor
                        cursor += dur
                        if cursor > measure_end:
                            measure_end = cursor

                    if is_grace or _find_child(child, "rest") is not None:
                        continue

                    pitch_el = _find_child(child, "pitch")
                    if pitch_el is None:
                        continue
                    step_el = _find_child(pitch_el, "step")
                    oct_el = _find_child(pitch_el, "octave")
                    step_text = (
                        step_el.text.strip().upper()
                        if (step_el is not None and step_el.text)
                        else ""
                    )
                    oct_text = (
                        oct_el.text.strip()
                        if (oct_el is not None and oct_el.text)
                        else ""
                    )
                    if not step_text or not oct_text or step_text not in _STEP_TO_SEMITONE:
                        continue
                    try:
                        octave = int(oct_text)
                    except ValueError:
                        continue
                    alter_el = _find_child(pitch_el, "alter")
                    alter = _parse_alter(alter_el.text if alter_el is not None else None)
                    midi = (octave + 1) * 12 + _STEP_TO_SEMITONE[step_text] + alter

                    staff_el = _find_child(child, "staff")
                    staff = (
                        staff_el.text.strip()
                        if (staff_el is not None and staff_el.text and staff_el.text.strip())
                        else "1"
                    )
                    voice_el = _find_child(child, "voice")
                    voice = (
                        voice_el.text.strip()
                        if (voice_el is not None and voice_el.text and voice_el.text.strip())
                        else "1"
                    )
                    onset_map = voice_events[(staff, voice)]
                    prev_midi = onset_map.get(onset)
                    if prev_midi is None or midi > prev_midi:
                        onset_map[onset] = midi

            cursor = measure_end
            last_onset = measure_end

        for key in sorted(voice_events):
            onset_map = voice_events[key]
            if onset_map:
                events = sorted(onset_map.items(), key=lambda ev: ev[0])
                streams.append(events)

    return streams


def _format_ioi_ratio(num_ioi: Fraction, den_ioi: Fraction) -> str:
    """Canonical scale-invariant IOI ratio token."""
    if den_ioi == 0:
        return "0" if num_ioi == 0 else "inf"
    ratio = num_ioi / den_ioi
    return f"{ratio.numerator}/{ratio.denominator}"


def compute_symbolic_8gram_hashes(
    musicxml_str: str,
    *,
    n: int = 8,
    salt: str = DEFAULT_SCRUB_SALT,
) -> list[str]:
    """Extract transposition- and tempo-invariant symbolic n-gram SHA-256 hashes.

    For every contiguous window of ``n`` skyline notes in each
    ``(part, staff, voice)`` stream, encodes the ``n - 1`` semitone pitch
    intervals and ``n - 2`` exact rational inter-onset-interval (IOI) ratios,
    then hashes the canonical representation with salted SHA-256.
    """
    if n < 3:
        raise ValueError("n must be at least 3 to form pitch intervals and IOI ratios.")

    streams = _extract_voice_note_streams(musicxml_str)
    digests: list[str] = []
    prefix = f"{salt}:v1|n={n}|dp="

    for events in streams:
        if len(events) < n:
            continue
        pitch_intervals = [
            str(events[i + 1][1] - events[i][1]) for i in range(len(events) - 1)
        ]
        iois = [events[i + 1][0] - events[i][0] for i in range(len(events) - 1)]
        ioi_ratios = [
            _format_ioi_ratio(iois[i + 1], iois[i]) for i in range(len(iois) - 1)
        ]
        for start in range(len(events) - n + 1):
            dp_str = ",".join(pitch_intervals[start : start + n - 1])
            ioir_str = ",".join(ioi_ratios[start : start + n - 2])
            payload = f"{prefix}{dp_str}|ioir={ioir_str}"
            digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
            digests.append(digest)

    return digests


def build_scrubbing_index(
    work_xml_map: Mapping[str, str],
    *,
    n: int = 8,
    salt: str = DEFAULT_SCRUB_SALT,
) -> dict[str, Any]:
    """Build a privacy-preserving blind scrubbing index from work MusicXML strings.

    Stores only one-way salted SHA-256 n-gram hashes mapped to work IDs, never
    raw symbolic notation, enabling external teams to scrub training corpora
    against sealed test splits without seeing the test data.
    """
    hash_to_works: dict[str, set[str]] = defaultdict(set)
    work_hash_counts: dict[str, int] = {}

    for work_id in sorted(work_xml_map):
        hashes = set(
            compute_symbolic_8gram_hashes(work_xml_map[work_id], n=n, salt=salt)
        )
        work_hash_counts[work_id] = len(hashes)
        for h in hashes:
            hash_to_works[h].add(work_id)

    return {
        "schema_version": "1.0.0",
        "n": n,
        "salt": salt,
        "work_count": len(work_xml_map),
        "hash_to_works": {h: sorted(wids) for h, wids in sorted(hash_to_works.items())},
        "work_hash_counts": work_hash_counts,
    }


def check_corpus_overlap(
    candidate_xml_map: Mapping[str, str],
    scrubbing_index: Mapping[str, Any],
    *,
    min_shared_8grams: int = 2,
) -> dict[str, dict[str, int]]:
    """Check candidate training works against a blind scrubbing index.

    Returns a mapping ``{candidate_work_id: {indexed_work_id: shared_count}}``
    for every candidate-vs-indexed pair sharing at least ``min_shared_8grams``
    unique symbolic n-gram hashes. Clean corpora return ``{}``.
    """
    if min_shared_8grams < 1:
        raise ValueError("min_shared_8grams must be at least 1.")

    n = int(scrubbing_index.get("n", 8))
    salt = str(scrubbing_index.get("salt", DEFAULT_SCRUB_SALT))
    hash_to_works: Mapping[str, list[str]] = scrubbing_index.get("hash_to_works", {})

    overlaps: dict[str, dict[str, int]] = {}
    for cand_id in sorted(candidate_xml_map):
        cand_hashes = set(
            compute_symbolic_8gram_hashes(candidate_xml_map[cand_id], n=n, salt=salt)
        )
        counts: dict[str, int] = defaultdict(int)
        for h in cand_hashes:
            for indexed_work_id in hash_to_works.get(h, ()):
                counts[indexed_work_id] += 1

        leaked = {
            wid: cnt for wid, cnt in sorted(counts.items()) if cnt >= min_shared_8grams
        }
        if leaked:
            overlaps[cand_id] = leaked

    return overlaps
