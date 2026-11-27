"""Command-line interface for OMR-NorthStar (OMR-NS)."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any, Sequence

from .compare import compare_reports, format_comparison
from .scoring import DOMAINS, PageMetrics, build_report, compute_page_quality
from .validate import (
    PERMISSIVE_LICENSES,
    REQUIRED_PAGE_FIELDS,
    VALID_VISUAL_TIERS,
    _is_valid_split,
    validate_disjoint_splits,
    validate_manifest,
    validate_page_records,
    validate_report,
)

# Backward-compatible aliases
REQUIRED_MANIFEST_FIELDS = REQUIRED_PAGE_FIELDS
VALID_SPLITS = {"dev-v1", "test-v1", "fresh"}
VALID_LICENSES = set(PERMISSIVE_LICENSES) | {"Public-Domain"}


def validate_manifest_records(records: Sequence[dict[str, Any]]) -> list[str]:
    """Validate a list of page metadata records against OMR-NS schema rules."""
    return validate_page_records(records, allow_legacy_pd=True)


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
    ]
    if l1.get("tier_weighted_ns") is not None:
        lines.append(f"Tier-Weighted NS: {l1['tier_weighted_ns']:.2f} / 100.00")
    if l1.get("ns_ci95") is not None:
        lo, hi = l1["ns_ci95"]
        lines.append(f"NS 95% CI       : [{lo:.2f}, {hi:.2f}]")

    lines.extend(["", "Domain Strata (Q / CFR):"])
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
    payload = json.loads(src.read_text(encoding="utf-8-sig"))
    if isinstance(payload, list):
        raw_pages = payload
        system_name = args.system_name or src.stem
        split = args.split if args.split is not None else "dev-v1"
        adversarial = None
        telemetry = None
    elif isinstance(payload, dict) and "pages" in payload:
        raw_pages = payload["pages"]
        system_name = args.system_name or str(payload.get("system_name", src.stem))
        split = (
            args.split
            if args.split is not None
            else str(payload.get("split", "dev-v1"))
        )
        adversarial = payload.get("adversarial_scorecard")
        telemetry = payload.get("efficiency_telemetry")
    else:
        print(
            "Error: input JSON must be a list of page objects or an object with a 'pages' key.",
            file=sys.stderr,
        )
        return 1

    if not _is_valid_split(split):
        print(
            f"Error: invalid split {split!r}. Expected 'dev-v1', 'test-v1', 'fresh', or 'fresh-YYYY'.",
            file=sys.stderr,
        )
        return 1

    pages = [PageMetrics.from_dict(item) for item in raw_pages]
    report = build_report(
        pages,
        system_name=system_name,
        split=split,
        adversarial=adversarial,
        telemetry=telemetry,
    )
    if args.include_per_page:
        per_page = []
        for p in pages:
            r = compute_page_quality(p)
            per_page.append(
                {
                    "page_id": p.page_id,
                    "work_id": p.work_id,
                    "domain": p.domain,
                    "tier": p.tier,
                    "q_score": round(r.q_score, 4),
                    "wfpp": round(r.effective_wfpp, 4),
                    "ntsr": round(r.ntsr, 4),
                    "catastrophic_failure": r.catastrophic_failure,
                }
            )
        report["layer4_forensic_pack"]["per_page_scores"] = per_page

    print(format_scorecard(report))

    if args.json_out:
        out_path = pathlib.Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    rep_a = json.loads(pathlib.Path(args.report_a).read_text(encoding="utf-8-sig"))
    rep_b = json.loads(pathlib.Path(args.report_b).read_text(encoding="utf-8-sig"))
    diff = compare_reports(
        rep_a,
        rep_b,
        n_perm=args.n_perm,
        seed=args.seed,
        alpha=args.alpha,
    )
    print(format_comparison(diff))
    if args.json_out:
        out_path = pathlib.Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(diff, indent=2) + "\n", encoding="utf-8")
    return 0 if diff["passed"] else 1


def _cmd_validate_manifest(args: argparse.Namespace) -> int:
    src = pathlib.Path(args.manifest_json)
    payload = json.loads(src.read_text(encoding="utf-8-sig"))
    errors = validate_manifest(payload, allow_legacy_pd=args.allow_legacy_pd)
    if errors:
        for err in errors:
            print(f"INVALID: {err}", file=sys.stderr)
        return 1
    count = len(payload) if isinstance(payload, list) else len(payload.get("pages", []))
    print(f"OK: validated {count} page metadata record(s).")
    return 0


def _cmd_validate_splits(args: argparse.Namespace) -> int:
    dev = json.loads(pathlib.Path(args.dev_manifest).read_text(encoding="utf-8-sig"))
    test = json.loads(pathlib.Path(args.test_manifest).read_text(encoding="utf-8-sig"))
    errors: list[str] = []
    errors.extend(f"[dev] {e}" for e in validate_manifest(dev, allow_legacy_pd=args.allow_legacy_pd))
    errors.extend(f"[test] {e}" for e in validate_manifest(test, allow_legacy_pd=args.allow_legacy_pd))
    errors.extend(validate_disjoint_splits(dev, test))
    if errors:
        for err in errors:
            print(f"INVALID: {err}", file=sys.stderr)
        return 1
    dev_n = len(dev) if isinstance(dev, list) else len(dev.get("pages", []))
    test_n = len(test) if isinstance(test, list) else len(test.get("pages", []))
    print(f"OK: splits are valid and disjoint (dev={dev_n} pages, test={test_n} pages).")
    return 0


def _cmd_validate_report(args: argparse.Namespace) -> int:
    src = pathlib.Path(args.report_json)
    payload = json.loads(src.read_text(encoding="utf-8-sig"))
    errors = validate_report(payload)
    if errors:
        for err in errors:
            print(f"INVALID: {err}", file=sys.stderr)
        return 1
    print(
        f"OK: report '{payload.get('system_name')}' ({payload.get('split')}) "
        f"is valid (NS={payload['layer1_headline']['ns_score']:.2f})."
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="omr-northstar",
        description="OMR-NorthStar (OMR-NS) reference scorer, comparator, and validator.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_score = sub.add_parser("score", help="Score a page-level results JSON file.")
    p_score.add_argument("results_json", help="Path to page-level results JSON.")
    p_score.add_argument("--system-name", default=None, help="Override system name.")
    p_score.add_argument("--split", default=None, help="Benchmark split (dev-v1, test-v1, fresh-YYYY).")
    p_score.add_argument("--json-out", default=None, help="Write canonical report.json to path.")
    p_score.add_argument(
        "--include-per-page",
        action="store_true",
        help="Include per_page_scores in layer4_forensic_pack for paired permutation compare().",
    )
    p_score.set_defaults(func=_cmd_score)

    p_cmp = sub.add_parser("compare", help="Compare two report.json files (Layer 4 regression diff).")
    p_cmp.add_argument("report_a", help="Baseline report.json (System A).")
    p_cmp.add_argument("report_b", help="Candidate report.json (System B).")
    p_cmp.add_argument("--n-perm", type=int, default=1000, help="Paired permutation count.")
    p_cmp.add_argument("--seed", type=int, default=42, help="Deterministic RNG seed.")
    p_cmp.add_argument("--alpha", type=float, default=0.05, help="Significance threshold.")
    p_cmp.add_argument("--json-out", default=None, help="Write comparison diff JSON to path.")
    p_cmp.set_defaults(func=_cmd_compare)

    p_val = sub.add_parser("validate-manifest", help="Validate a split manifest JSON and governance rules.")
    p_val.add_argument("manifest_json", help="Path to manifest JSON.")
    p_val.add_argument(
        "--allow-legacy-pd",
        action="store_true",
        help="Allow legacy 'Public-Domain' string alongside SPDX PDM-1.0.",
    )
    p_val.set_defaults(func=_cmd_validate_manifest)

    p_splits = sub.add_parser(
        "validate-splits",
        help="Validate dev and test manifests plus zero-leakage cross-split disjointness.",
    )
    p_splits.add_argument("dev_manifest", help="Path to dev manifest JSON.")
    p_splits.add_argument("test_manifest", help="Path to test manifest JSON.")
    p_splits.add_argument(
        "--allow-legacy-pd",
        action="store_true",
        help="Allow legacy 'Public-Domain' string alongside SPDX PDM-1.0.",
    )
    p_splits.set_defaults(func=_cmd_validate_splits)

    p_rep = sub.add_parser("validate-report", help="Validate a canonical report.json file.")
    p_rep.add_argument("report_json", help="Path to report.json.")
    p_rep.set_defaults(func=_cmd_validate_report)

    parsed = parser.parse_args(argv)
    try:
        return int(parsed.func(parsed))
    except (OSError, json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
