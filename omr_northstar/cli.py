"""Command-line interface for OMR-NorthStar (OMR-NS)."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any, Sequence

from .scoring import DOMAINS, TIERS, PageMetrics, build_report

REQUIRED_MANIFEST_FIELDS: tuple[str, ...] = (
    "page_id",
    "work_id",
    "split",
    "domain",
    "complexity_tier",
    "visual_tier",
    "repertoire_stratum",
    "staff_space_px",
    "geometry",
    "photometry",
    "optics_codec",
    "provenance",
    "license",
)
VALID_SPLITS = {"dev-v1", "test-v1", "fresh"}
VALID_VISUAL_TIERS = {"E", "M", "H", "B"}
VALID_LICENSES = {"Public-Domain", "CC0-1.0", "CC-BY-4.0"}


def validate_manifest_records(records: Sequence[dict[str, Any]]) -> list[str]:
    """Validate a list of page metadata records against OMR-NS schema rules."""
    errors: list[str] = []
    seen_ids: set[str] = set()
    for idx, rec in enumerate(records):
        prefix = f"record[{idx}]"
        if not isinstance(rec, dict):
            errors.append(f"{prefix}: expected object, got {type(rec).__name__}")
            continue
        for req in REQUIRED_MANIFEST_FIELDS:
            if req not in rec:
                errors.append(f"{prefix}: missing required field '{req}'")
        pid = rec.get("page_id")
        if pid in seen_ids:
            errors.append(f"{prefix}: duplicate page_id '{pid}'")
        elif isinstance(pid, str):
            seen_ids.add(pid)
        if rec.get("split") not in VALID_SPLITS:
            errors.append(f"{prefix}: invalid split {rec.get('split')!r}")
        if rec.get("domain") not in DOMAINS:
            errors.append(f"{prefix}: invalid domain {rec.get('domain')!r}")
        if rec.get("complexity_tier") not in TIERS:
            errors.append(f"{prefix}: invalid complexity_tier {rec.get('complexity_tier')!r}")
        if rec.get("visual_tier") not in VALID_VISUAL_TIERS:
            errors.append(f"{prefix}: invalid visual_tier {rec.get('visual_tier')!r}")
        lic = rec.get("license")
        if isinstance(lic, dict):
            if lic.get("music_license") not in VALID_LICENSES:
                errors.append(f"{prefix}: non-permissive music_license {lic.get('music_license')!r}")
            if lic.get("image_license") not in VALID_LICENSES:
                errors.append(f"{prefix}: non-permissive image_license {lic.get('image_license')!r}")
    return errors


def format_scorecard(report: dict[str, Any]) -> str:
    """Render a clean human-readable 4-layer leaderboard scorecard."""
    l1 = report["layer1_headline"]
    l2 = report["layer2_matrix_and_deltas"]
    l4 = report["layer4_forensic_pack"]

    lines = [
        f"=== OMR-NorthStar ({report['benchmark_version']} / {report['split']}) ===",
        f"System: {report['system_name']}",
        "",
        "--- Layer 1: Headline North Star Score ---",
        f"NS Score        : {l1['ns_score']:.2f} / 100.00",
        f"M_geom          : {l1['m_geom']:.2f}",
        f"Worst Stratum Q : {l1['worst_stratum_q']:.2f}",
        f"CFR_max         : {l1['cfr_max'] * 100:.2f}%",
        "",
        "Domain Strata (Q / CFR):",
    ]
    for d in DOMAINS:
        q = l1["stratum_scores"][d]
        cfr = l1["stratum_cfr"][d] * 100.0
        lines.append(f"  {d:4s} : Q={q:6.2f} | CFR={cfr:5.2f}%")

    ax = l1["axes_macro"]
    hu = l1["human_utility"]
    lines.extend(
        [
            "",
            f"Axes Macro      : S_note={ax['s_note']:.4f} | S_struct={ax['s_struct']:.4f} "
            f"| S_expr={ax['s_expr']:.4f} | S_play={ax['s_play']:.4f}",
            f"Human Utility   : median_wFPP={hu['median_wfpp']:.2f} | P90={hu['p90_wfpp']:.2f} "
            f"| P99={hu['p99_wfpp']:.2f} | mean_NTSR={hu['mean_ntsr']:.3f} | NUPR={hu['nupr']*100:.1f}%",
            "",
            "--- Layer 2: Paired Robustness Deltas ---",
        ]
    )
    pd = l2["paired_deltas"]
    for key in ("delta_engraver", "delta_photo", "delta_handwriting"):
        val = pd.get(key)
        sval = f"{val:+.2f}" if val is not None else "N/A"
        lines.append(f"  {key:18s}: {sval}")

    lines.append("")
    lines.append("--- Layer 4: Worst Pages (Top 5) ---")
    for wp in l4["worst_pages"][:5]:
        cf_tag = " [CF]" if wp["catastrophic_failure"] else ""
        lines.append(
            f"  {wp['page_id']} ({wp['domain']}): Q={wp['q_score']:.2f}, wFPP={wp['wfpp']:.2f}{cf_tag}"
        )
    return "\n".join(lines)


def _cmd_score(args: argparse.Namespace) -> int:
    src = pathlib.Path(args.results_json)
    payload = json.loads(src.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        raw_pages = payload
        system_name = args.system_name or src.stem
        split = args.split
        adversarial = None
        telemetry = None
    elif isinstance(payload, dict) and "pages" in payload:
        raw_pages = payload["pages"]
        system_name = args.system_name or str(payload.get("system_name", src.stem))
        split = str(payload.get("split", args.split))
        adversarial = payload.get("adversarial_scorecard")
        telemetry = payload.get("efficiency_telemetry")
    else:
        print("Error: input JSON must be a list of page objects or an object with a 'pages' key.", file=sys.stderr)
        return 1

    pages = [PageMetrics.from_dict(item) for item in raw_pages]
    report = build_report(
        pages,
        system_name=system_name,
        split=split,
        adversarial=adversarial,
        telemetry=telemetry,
    )
    print(format_scorecard(report))

    if args.json_out:
        out_path = pathlib.Path(args.json_out)
        out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


def _cmd_validate_manifest(args: argparse.Namespace) -> int:
    src = pathlib.Path(args.manifest_json)
    payload = json.loads(src.read_text(encoding="utf-8"))
    records = payload if isinstance(payload, list) else payload.get("pages", [])
    errors = validate_manifest_records(records)
    if errors:
        for err in errors:
            print(f"INVALID: {err}", file=sys.stderr)
        return 1
    print(f"OK: validated {len(records)} page metadata record(s).")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="omr-northstar",
        description="OMR-NorthStar (OMR-NS) reference scorer and manifest validator.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_score = sub.add_parser("score", help="Score a page-level results JSON file.")
    p_score.add_argument("results_json", help="Path to page-level results JSON.")
    p_score.add_argument("--system-name", default=None, help="Override system name.")
    p_score.add_argument("--split", default="dev-v1", choices=["dev-v1", "test-v1", "fresh"])
    p_score.add_argument("--json-out", default=None, help="Write canonical report.json to path.")
    p_score.set_defaults(func=_cmd_score)

    p_val = sub.add_parser("validate-manifest", help="Validate a page metadata manifest JSON.")
    p_val.add_argument("manifest_json", help="Path to manifest JSON.")
    p_val.set_defaults(func=_cmd_validate_manifest)

    parsed = parser.parse_args(argv)
    return int(parsed.func(parsed))


if __name__ == "__main__":
    raise SystemExit(main())
