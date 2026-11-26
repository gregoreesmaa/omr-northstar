"""Blind Scrubbing Index for zero-leakage benchmark governance.

Zero external dependencies (Python 3.10+ standard library only).
Implements transposition-invariant and tempo/divisions-invariant symbolic
pitch-interval + IOI-ratio 8-gram hashing as specified in
docs/01-architecture-and-governance.md (Section 3.2).
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


def _extract_voice_note_streams(musicxml_str: str) -> list[list[tuple[Fraction, int]]]:
    """Extract ordered (onset_beats, midi_pitch) streams per (part, voice)."""
    root = ET.fromstring(musicxml_str)
    streams: list[list[tuple[Fraction, int]]] = []

    for part in _iter_children(root, "part"):
        divisions = 1
        cursor = Fraction(0)
        last_onset = Fraction(0)
        voice_events: dict[str, list[tuple[Fraction, int]]] = defaultdict(list)

        for measure in _iter_children(part, "measure"):
            for child in measure:
                tag = _strip_ns(child.tag)
                if tag == "attributes":
                    div_el = _find_child(child, "divisions")
                    if div_el is not None and div_el.text:
                        parsed_div = int(div_el.text.strip())
                        if parsed_div > 0:
                            divisions = parsed_div
                elif tag == "backup":
                    dur_el = _find_child(child, "duration")
                    if dur_el is not None and dur_el.text:
                        cursor -= Fraction(int(dur_el.text.strip()), divisions)
                elif tag == "forward":
                    dur_el = _find_child(child, "duration")
                    if dur_el is not None and dur_el.text:
                        cursor += Fraction(int(dur_el.text.strip()), divisions)
                elif tag == "note":
                    is_grace = _find_child(child, "grace") is not None
                    is_chord = _find_child(child, "chord") is not None
                    dur_el = _find_child(child, "duration")
                    dur = (
                        Fraction(int(dur_el.text.strip()), divisions)
                        if (dur_el is not None and dur_el.text and not is_grace)
                        else Fraction(0)
                    )
                    if is_chord:
                        onset = last_onset
                    else:
                        onset = cursor
                        last_onset = cursor
                        cursor += dur

                    if is_grace or _find_child(child, "rest") is not None:
                        continue

                    pitch_el = _find_child(child, "pitch")
                    if pitch_el is None:
                        continue
                    step_el = _find_child(pitch_el, "step")
                    oct_el = _find_child(pitch_el, "octave")
                    if (
                        step_el is None
                        or not step_el.text
                        or oct_el is None
                        or not oct_el.text
                    ):
                        continue
                    step = step_el.text.strip().upper()
                    if step not in _STEP_TO_SEMITONE:
                        continue
                    alter_el = _find_child(pitch_el, "alter")
                    alter = (
                        int(round(float(alter_el.text.strip())))
                        if (alter_el is not None and alter_el.text)
                        else 0
                    )
                    octave = int(oct_el.text.strip())
                    midi = (octave + 1) * 12 + _STEP_TO_SEMITONE[step] + alter

                    voice_el = _find_child(child, "voice")
                    voice = (
                        voice_el.text.strip()
                        if (voice_el is not None and voice_el.text)
                        else "1"
                    )
                    voice_events[voice].append((onset, midi))

        for voice in sorted(voice_events):
            events = sorted(voice_events[voice], key=lambda ev: (ev[0], ev[1]))
            if events:
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

    For every contiguous window of ``n`` notes in each ``(part, voice)`` stream,
    encodes the ``n - 1`` semitone pitch intervals and ``n - 2`` exact rational
    inter-onset-interval (IOI) ratios, then hashes the canonical representation
    with salted SHA-256.
    """
    if n < 3:
        raise ValueError("n must be at least 3 to form pitch intervals and IOI ratios.")

    streams = _extract_voice_note_streams(musicxml_str)
    digests: list[str] = []

    for events in streams:
        if len(events) < n:
            continue
        for start in range(len(events) - n + 1):
            window = events[start : start + n]
            pitch_intervals = [
                window[i + 1][1] - window[i][1] for i in range(n - 1)
            ]
            iois = [window[i + 1][0] - window[i][0] for i in range(n - 1)]
            ioi_ratios = [
                _format_ioi_ratio(iois[i + 1], iois[i]) for i in range(n - 2)
            ]
            canonical = (
                f"v1|n={n}|dp={','.join(str(d) for d in pitch_intervals)}"
                f"|ioir={','.join(ioi_ratios)}"
            )
            digest = hashlib.sha256(f"{salt}:{canonical}".encode("utf-8")).hexdigest()
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
