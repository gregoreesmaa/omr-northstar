"""End-to-end pilot fixture runner and reference baselines for OMR-NorthStar.

Zero external dependencies (Python 3.10+ standard library only).
Provides:
  - ``load_pilot_manifest(pilot_dir)``: load and resolve pilot manifest + GT XML paths.
  - ``parse_musicxml_summary(xml_path)``: lightweight stdlib MusicXML 3.1/4.0 summary.
  - ``run_oracle_baseline(pilot_dir)``: exact ground-truth identity baseline (NS == 100.0).
  - ``run_degraded_baseline(pilot_dir, seed=42)``: realistic clean-strong, HW-weak
    baseline demonstrating geometric-mean collapse and paired robustness deltas.
"""

from __future__ import annotations

import json
import pathlib
import random
import xml.etree.ElementTree as ET
from typing import Any, Mapping

from .scoring import PageMetrics, build_report

CANARY_STRING = "OMR_NS_V1_CANARY_DO_NOT_TRAIN"

try:
    from .cer import score_musicxml_pair as _cer_score_musicxml_pair  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - cer.py is optional for standalone pilot smoke runs
    _cer_score_musicxml_pair = None


def load_pilot_manifest(pilot_dir: str | pathlib.Path) -> dict[str, Any]:
    """Load ``manifest.json`` from a pilot fixture directory."""
    root = pathlib.Path(pilot_dir)
    manifest_path = root / "manifest.json"
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def resolve_gt_path(
    pilot_dir: str | pathlib.Path,
    manifest: Mapping[str, Any],
    page_id: str,
) -> pathlib.Path:
    """Resolve the ground-truth MusicXML path for ``page_id``."""
    root = pathlib.Path(pilot_dir)
    gt_map = manifest.get("ground_truth_files")
    if isinstance(gt_map, Mapping) and page_id in gt_map:
        return root / str(gt_map[page_id])
    return root / "gt" / f"{page_id}.musicxml"


def parse_musicxml_summary(xml_path: str | pathlib.Path) -> dict[str, Any]:
    """Parse a MusicXML ``<score-partwise>`` file and extract structural counts."""
    path = pathlib.Path(xml_path)
    tree = ET.parse(path)
    root = tree.getroot()

    parts = root.findall("part")
    n_parts = len(parts)
    n_notes = 0
    n_rests = 0
    n_dynamics = 0
    n_lyrics = 0
    n_staves = 0
    m_gt = 0
    events: list[tuple[str, str, int, str, int, str]] = []

    for part_idx, part in enumerate(parts):
        part_id = part.attrib.get("id", f"P{part_idx + 1}")
        measures = part.findall("measure")
        if part_idx == 0:
            m_gt = len(measures)
        part_staves = 1
        for measure in measures:
            m_num = measure.attrib.get("number", "0")
            for attr in measure.findall("attributes"):
                staves_el = attr.find("staves")
                if staves_el is not None and staves_el.text:
                    part_staves = max(part_staves, int(staves_el.text.strip()))
            n_dynamics += len(measure.findall(".//dynamics"))
            for note in measure.findall("note"):
                if note.find("rest") is not None:
                    n_rests += 1
                    continue
                n_notes += 1
                n_lyrics += len(note.findall("lyric"))
                pitch_el = note.find("pitch")
                step = pitch_el.findtext("step", default="") if pitch_el is not None else ""
                alter = pitch_el.findtext("alter", default="0") if pitch_el is not None else "0"
                octave = pitch_el.findtext("octave", default="0") if pitch_el is not None else "0"
                dur = int(note.findtext("duration", default="0"))
                voice = note.findtext("voice", default="1")
                events.append((part_id, m_num, dur, f"{step}{alter}:{octave}", int(voice), note.findtext("type", default="")))
        n_staves += part_staves

    # Rational-musician baseline transcription cost (strictly positive even for blank staves)
    transcribe_cost = float(max(1.0, 2.0 * n_notes + 1.0 * n_rests + 1.5 * m_gt + 1.0 * (n_dynamics + n_lyrics)))

    return {
        "parseable": root.tag.endswith("score-partwise"),
        "n_parts": n_parts,
        "n_staves": max(1, n_staves),
        "m_gt": max(1, m_gt),
        "n_notes": n_notes,
        "n_rests": n_rests,
        "n_dynamics": n_dynamics,
        "n_lyrics": n_lyrics,
        "transcribe_cost": transcribe_cost,
        "events": events,
    }


def _extract_engraver(rec: Mapping[str, Any]) -> str | None:
    prov = rec.get("provenance")
    if isinstance(prov, Mapping):
        val = prov.get("engraver_or_publisher")
        if val is not None:
            return str(val)
    return None


def _compare_identity_xml(gt_path: pathlib.Path, pred_path: pathlib.Path) -> tuple[bool, float, float, float, float, float]:
    """Compare two MusicXML files; uses ``cer.score_musicxml_pair`` if available, else stdlib exact-event check."""
    if _cer_score_musicxml_pair is not None:
        try:
            res = _cer_score_musicxml_pair(gt_path, pred_path)
            if isinstance(res, Mapping) and {"s_note", "s_struct", "s_expr", "s_play", "wfpp"}.issubset(res.keys()):
                return (
                    bool(res.get("parseable", True)),
                    float(res["s_note"]),
                    float(res["s_struct"]),
                    float(res["s_expr"]),
                    float(res["s_play"]),
                    float(res["wfpp"]),
                )
        except Exception:
            pass

    gt_info = parse_musicxml_summary(gt_path)
    pred_info = parse_musicxml_summary(pred_path)
    identical = (
        gt_info["parseable"]
        and pred_info["parseable"]
        and gt_info["events"] == pred_info["events"]
        and gt_info["m_gt"] == pred_info["m_gt"]
        and gt_info["n_rests"] == pred_info["n_rests"]
    )
    if identical:
        return (True, 1.0, 1.0, 1.0, 1.0, 0.0)
    return (bool(pred_info["parseable"]), 0.0, 0.0, 0.0, 0.0, gt_info["transcribe_cost"])


def run_oracle_baseline(pilot_dir: str | pathlib.Path) -> dict[str, Any]:
    """Feed exact ground-truth MusicXML as predictions; proves NS == 100.0, CFR_max == 0.0, NTSR == 1.0."""
    manifest = load_pilot_manifest(pilot_dir)
    records = manifest.get("pages", [])
    page_metrics: list[PageMetrics] = []

    for rec in records:
        page_id = str(rec["page_id"])
        gt_path = resolve_gt_path(pilot_dir, manifest, page_id)
        summary = parse_musicxml_summary(gt_path)
        parseable, s_note, s_struct, s_expr, s_play, wfpp = _compare_identity_xml(gt_path, gt_path)

        page_metrics.append(
            PageMetrics(
                page_id=page_id,
                work_id=str(rec["work_id"]),
                domain=str(rec["domain"]),
                tier=str(rec["complexity_tier"]),
                s_note=s_note,
                s_struct=s_struct,
                s_expr=s_expr,
                s_play=s_play,
                wfpp=wfpp,
                transcribe_cost=summary["transcribe_cost"],
                parseable=parseable,
                n_gt=summary["n_notes"],
                n_pred=summary["n_notes"],
                k_dropped_systems=0,
                has_repetition_loop=False,
                bsvr=1.0,
                m_gt=summary["m_gt"],
                m_pred=summary["m_gt"],
                paired_anchor_id=rec.get("paired_anchor_id"),
                engraver=_extract_engraver(rec),
                error_buckets={
                    "layout_segmentation_pct": 0.0,
                    "global_context_drift_pct": 0.0,
                    "pitch_accidental_pct": 0.0,
                    "rhythm_meter_pct": 0.0,
                    "polyphony_routing_pct": 0.0,
                    "engraving_expressive_pct": 0.0,
                    "hallucination_truncation_pct": 0.0,
                    "annotation_contamination_pct": 0.0,
                },
                latency_s=0.05,
            )
        )

    return build_report(
        page_metrics,
        system_name="pilot-oracle-baseline",
        split=str(manifest.get("split", "dev-v1")),
        benchmark_version=str(manifest.get("benchmark_version", "1.0.0")),
        adversarial={
            "phr": 0.0,
            "ostinato_tsr": 1.0,
            "q4_q1_ratio": 1.0,
            "non_music_fp_count": 0,
            "cross_page_spanner_f1": 1.0,
            "french_score_staff_acc": 1.0,
        },
        telemetry={
            "peak_memory_gb": 0.1,
            "parameter_count": 0,
        },
    )


def run_degraded_baseline(
    pilot_dir: str | pathlib.Path,
    seed: int = 42,
) -> dict[str, Any]:
    """Simulate a realistic clean-strong, HW-weak OMR system on the pilot fixture pack.

    Demonstrates:
      - High CDR accuracy, moderate ALS accuracy, noticeable RWP photo drop (delta_photo < 0),
        and severe HWM degradation + repetition-loop CF (delta_handwriting < 0).
      - Anti-gaming geometric mean collapse in the North Star Score (0 < NS < 70).
    """
    rng = random.Random(seed)
    manifest = load_pilot_manifest(pilot_dir)
    records = manifest.get("pages", [])
    page_metrics: list[PageMetrics] = []

    # Domain baseline accuracy profiles (s_note, s_struct, s_expr, wfpp_frac)
    domain_profiles: dict[str, tuple[float, float, float, float]] = {
        "CDR": (0.96, 0.96, 0.92, 0.05),
        "ALS": (0.88, 0.87, 0.81, 0.16),
        "RWP": (0.78, 0.77, 0.68, 0.30),
        "HWM": (0.48, 0.50, 0.38, 0.68),
    }
    tier_difficulty_pen: dict[str, float] = {
        "T0": 0.00,
        "T1": 0.015,
        "T2": 0.030,
        "T3": 0.045,
        "T4": 0.060,
        "T5": 0.080,
    }

    for rec in records:
        page_id = str(rec["page_id"])
        domain = str(rec["domain"])
        tier = str(rec["complexity_tier"])
        gt_path = resolve_gt_path(pilot_dir, manifest, page_id)
        summary = parse_musicxml_summary(gt_path)
        n_gt = summary["n_notes"]
        m_gt = summary["m_gt"]
        t_cost = summary["transcribe_cost"]

        base_note, base_struct, base_expr, base_wfpp_frac = domain_profiles[domain]
        t_pen = tier_difficulty_pen.get(tier, 0.02)
        jitter = rng.uniform(-0.012, 0.012)

        # Introduce realistic engraver sensitivity on Protocol A (Verovio slightly lower than Dorico)
        engraver = _extract_engraver(rec)
        eng_pen = 0.025 if (domain == "CDR" and engraver and "Verovio" in engraver) else 0.0

        s_note = max(0.05, min(0.99, base_note - t_pen - eng_pen + jitter))
        s_struct = max(0.05, min(0.99, base_struct - t_pen - eng_pen + rng.uniform(-0.01, 0.01)))
        s_expr = max(0.05, min(0.99, base_expr - t_pen - eng_pen + rng.uniform(-0.015, 0.015)))
        s_play = max(0.05, min(0.99, 0.7 * s_note + 0.3 * s_struct))
        wfpp = max(0.0, t_cost * min(1.15, base_wfpp_frac + t_pen + eng_pen + rng.uniform(-0.01, 0.02)))

        n_pred = n_gt
        m_pred = m_gt
        k_dropped = 0
        has_loop = False
        bsvr = max(0.55, min(1.0, s_struct + 0.06))

        # Negative-control blank staff: clean-strong system avoids hallucinating notes on ALS
        if n_gt == 0:
            s_note, s_struct, s_expr, s_play = 0.96, 0.95, 0.94, 0.96
            wfpp = 0.08 * t_cost
            n_pred = 0

        # Simulate realistic catastrophic repetition loop on brutal T5 HWM page
        if domain == "HWM" and tier == "T5":
            has_loop = True
            n_pred = max(n_gt + 1, int(round(1.85 * max(n_gt, 1))))
            bsvr = 0.48

        # Realistic per-domain error taxonomy breakdown (sums to 100%)
        if domain == "CDR":
            buckets = {
                "layout_segmentation_pct": 5.0,
                "global_context_drift_pct": 5.0,
                "pitch_accidental_pct": 25.0,
                "rhythm_meter_pct": 20.0,
                "polyphony_routing_pct": 15.0,
                "engraving_expressive_pct": 25.0,
                "hallucination_truncation_pct": 5.0,
                "annotation_contamination_pct": 0.0,
            }
        elif domain == "ALS":
            buckets = {
                "layout_segmentation_pct": 12.0,
                "global_context_drift_pct": 10.0,
                "pitch_accidental_pct": 28.0,
                "rhythm_meter_pct": 18.0,
                "polyphony_routing_pct": 12.0,
                "engraving_expressive_pct": 14.0,
                "hallucination_truncation_pct": 4.0,
                "annotation_contamination_pct": 2.0,
            }
        elif domain == "RWP":
            buckets = {
                "layout_segmentation_pct": 24.0,
                "global_context_drift_pct": 14.0,
                "pitch_accidental_pct": 24.0,
                "rhythm_meter_pct": 14.0,
                "polyphony_routing_pct": 8.0,
                "engraving_expressive_pct": 8.0,
                "hallucination_truncation_pct": 5.0,
                "annotation_contamination_pct": 3.0,
            }
        else:  # HWM
            buckets = {
                "layout_segmentation_pct": 16.0,
                "global_context_drift_pct": 14.0,
                "pitch_accidental_pct": 32.0,
                "rhythm_meter_pct": 12.0,
                "polyphony_routing_pct": 6.0,
                "engraving_expressive_pct": 4.0,
                "hallucination_truncation_pct": 12.0,
                "annotation_contamination_pct": 4.0,
            }

        page_metrics.append(
            PageMetrics(
                page_id=page_id,
                work_id=str(rec["work_id"]),
                domain=domain,
                tier=tier,
                s_note=s_note,
                s_struct=s_struct,
                s_expr=s_expr,
                s_play=s_play,
                wfpp=wfpp,
                transcribe_cost=t_cost,
                parseable=True,
                n_gt=n_gt,
                n_pred=n_pred,
                k_dropped_systems=k_dropped,
                has_repetition_loop=has_loop,
                bsvr=bsvr,
                m_gt=m_gt,
                m_pred=m_pred,
                paired_anchor_id=rec.get("paired_anchor_id"),
                engraver=engraver,
                error_buckets=buckets,
                latency_s=round(rng.uniform(0.85, 2.40), 3),
            )
        )

    return build_report(
        page_metrics,
        system_name="pilot-degraded-baseline",
        split=str(manifest.get("split", "dev-v1")),
        benchmark_version=str(manifest.get("benchmark_version", "1.0.0")),
        adversarial={
            "phr": 0.35,
            "ostinato_tsr": 0.40,
            "q4_q1_ratio": 0.62,
            "non_music_fp_count": 0,
            "cross_page_spanner_f1": 0.72,
            "french_score_staff_acc": 0.78,
        },
        telemetry={
            "peak_memory_gb": 4.2,
            "parameter_count": 85_000_000,
        },
    )
