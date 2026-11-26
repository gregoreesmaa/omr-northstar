"""Unit tests for OMR-NorthStar CLI, compare(A, B), governance validators, and JSON Schemas."""

from __future__ import annotations

import json
import pathlib
import tempfile
import unittest
from typing import Any

from omr_northstar.cli import main as cli_main
from omr_northstar.compare import compare_reports, format_comparison
from omr_northstar.scoring import DOMAINS, PageMetrics, build_report, compute_page_quality
from omr_northstar.validate import (
    NC_EVAL_ONLY_LICENSES,
    PERMISSIVE_LICENSES,
    validate_disjoint_splits,
    validate_manifest,
    validate_report,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _make_page_record(
    page_id: str = "ns-dev-0001",
    work_id: str = "work-0001",
    split: str = "dev-v1",
    domain: str = "RWP",
    engraver: str | None = None,
    writer_id: str | None = None,
    music_license: str = "PDM-1.0",
    image_license: str = "CC-BY-4.0",
    **extra: Any,
) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "page_id": page_id,
        "work_id": work_id,
        "split": split,
        "domain": domain,
        "complexity_tier": "T2",
        "visual_tier": "M",
        "visual_severity": "V2",
        "repertoire_stratum": "T2.1",
        "staff_space_px": {"median": 10.5},
        "geometry": {"rotation_deg": 2.0, "keystone_x_deg": 8.0, "keystone_y_deg": -2.0},
        "photometry": {"weber_contrast": 0.65, "illumination_gradient_ev": 1.0},
        "optics_codec": {"blur_sigma_px": 1.0, "bit_depth": 24, "codec": "jpeg", "quality_factor": 85},
        "phenomena_tags": ["mid_bar_clef"],
        "provenance": {
            "source_type": "in_house_phone",
            "engraver_or_publisher": engraver,
            "writer_id": writer_id,
            "gt_verification": "double_blind_human_adjudicated",
        },
        "license": {
            "music_license": music_license,
            "image_license": image_license,
        },
    }
    rec.update(extra)
    return rec


def _make_manifest(
    pages: list[dict[str, Any]],
    *,
    split: str = "dev-v1",
    canary: str | None = "OMR_NS_V1_CANARY_DO_NOT_TRAIN",
    **extra: Any,
) -> dict[str, Any]:
    m: dict[str, Any] = {
        "schema_version": "1.0.0",
        "benchmark_version": "1.0.0",
        "split": split,
        "pages": pages,
    }
    if canary is not None:
        m["canary"] = canary
    m.update(extra)
    return m


def _make_report_with_pages(
    system_name: str,
    domain_scores: dict[str, list[tuple[str, str, float]]],
) -> dict[str, Any]:
    """Build a report + attached per_page_scores from {domain: [(page_id, tier, s_val), ...]}."""
    pages: list[PageMetrics] = []
    per_page: list[dict[str, Any]] = []
    for d, items in domain_scores.items():
        for pid, tier, s_val in items:
            wfpp = max(0.0, (1.0 - s_val) * 100.0)
            pm = PageMetrics(
                page_id=pid,
                work_id=f"w_{pid}",
                domain=d,
                tier=tier,
                s_note=s_val,
                s_struct=s_val,
                s_expr=s_val,
                s_play=s_val,
                wfpp=wfpp,
                transcribe_cost=100.0,
            )
            pages.append(pm)
            res = compute_page_quality(pm)
            per_page.append(
                {
                    "page_id": pid,
                    "work_id": pm.work_id,
                    "domain": d,
                    "tier": tier,
                    "q_score": round(res.q_score, 4),
                    "wfpp": round(res.effective_wfpp, 4),
                    "ntsr": round(res.ntsr, 4),
                    "catastrophic_failure": res.catastrophic_failure,
                }
            )
    rep = build_report(pages, system_name=system_name, split="dev-v1")
    rep["layer4_forensic_pack"]["per_page_scores"] = per_page
    return rep


class TestSchemas(unittest.TestCase):
    """Verify Draft 2020-12 JSON Schemas for page_metadata, report, and manifest."""

    def test_all_three_schemas_exist_and_are_valid(self) -> None:
        schema_dir = REPO_ROOT / "schemas"
        expected = {
            "page_metadata.schema.json",
            "report.schema.json",
            "manifest.schema.json",
        }
        found = {p.name for p in schema_dir.glob("*.json")}
        self.assertTrue(expected.issubset(found))
        for name in expected:
            data = json.loads((schema_dir / name).read_text(encoding="utf-8"))
            self.assertEqual(data["$schema"], "https://json-schema.org/draft/2020-12/schema")
            self.assertEqual(data["type"], "object")
            self.assertIn("properties", data)
            self.assertIn("required", data)

    def test_page_metadata_schema_new_fields(self) -> None:
        data = json.loads((REPO_ROOT / "schemas" / "page_metadata.schema.json").read_text(encoding="utf-8"))
        props = data["properties"]
        for key in (
            "is_negative_control",
            "paired_protocol",
            "human_agreement_score",
            "ambiguous_spans",
            "visual_severity",
            "phenomena_tags",
            "track",
            "eval_only",
        ):
            self.assertIn(key, props)
        self.assertEqual(props["visual_severity"]["enum"], ["V1", "V2", "V3", "V4"])
        self.assertIn("null", props["staff_space_px"]["type"])

    def test_report_schema_covers_build_report_and_extensions(self) -> None:
        data = json.loads((REPO_ROOT / "schemas" / "report.schema.json").read_text(encoding="utf-8"))
        l1_props = data["properties"]["layer1_headline"]["properties"]
        self.assertIn("tier_weighted_ns", l1_props)
        self.assertIn("ns_ci95", l1_props)
        self.assertIn("stress_tests", data["properties"])
        self.assertIn("stress_tests", data["properties"]["layer2_matrix_and_deltas"]["properties"])
        self.assertIn("per_page_scores", data["properties"]["layer4_forensic_pack"]["properties"])

    def test_manifest_schema_envelope_and_tracks(self) -> None:
        data = json.loads((REPO_ROOT / "schemas" / "manifest.schema.json").read_text(encoding="utf-8"))
        props = data["properties"]
        for key in ("schema_version", "benchmark_version", "split", "canary", "track", "eval_only", "pages"):
            self.assertIn(key, props)
        self.assertEqual(props["pages"]["items"]["$ref"], "page_metadata.schema.json")


class TestGovernanceValidators(unittest.TestCase):
    """Test every governance and schema rule in validate.py."""

    def test_core_permissive_rejects_nc_and_proprietary(self) -> None:
        for bad_lic in ("CC-BY-NC-4.0", "CC-BY-NC-SA-4.0", "CC-BY-NC-3.0", "Proprietary", "All-Rights-Reserved"):
            m = _make_manifest([_make_page_record(music_license=bad_lic)])
            errs = validate_manifest(m)
            self.assertTrue(
                any("non-permissive music_license" in e for e in errs),
                f"Expected rejection for {bad_lic}, got {errs}",
            )

    def test_core_permissive_accepts_all_allowlisted_spdx(self) -> None:
        pages = [
            _make_page_record(
                page_id=f"p_{i}",
                work_id=f"w_{i}",
                music_license=lic,
                image_license=lic,
            )
            for i, lic in enumerate(sorted(PERMISSIVE_LICENSES))
        ]
        m = _make_manifest(pages, track="core_permissive")
        self.assertEqual(validate_manifest(m), [])

    def test_research_nc_eval_only_track_requires_eval_only_true(self) -> None:
        # Allowed when track="research_nc_eval_only" and eval_only=True
        pages = [
            _make_page_record(
                page_id=f"nc_{i}",
                work_id=f"w_nc_{i}",
                music_license=lic,
                image_license=lic,
                track="research_nc_eval_only",
                eval_only=True,
            )
            for i, lic in enumerate(sorted(NC_EVAL_ONLY_LICENSES))
        ]
        m_ok = _make_manifest(pages, track="research_nc_eval_only", eval_only=True)
        self.assertEqual(validate_manifest(m_ok), [])

        # Rejected when eval_only is omitted or False
        m_no_flag = _make_manifest(
            [_make_page_record(music_license="CC-BY-NC-4.0", track="research_nc_eval_only")],
            track="research_nc_eval_only",
        )
        errs_no_flag = validate_manifest(m_no_flag)
        self.assertTrue(any("eval_only=true" in e for e in errs_no_flag))

        m_false_flag = _make_manifest(
            [_make_page_record(music_license="CC-BY-NC-4.0", track="research_nc_eval_only", eval_only=False)],
            track="research_nc_eval_only",
            eval_only=False,
        )
        errs_false = validate_manifest(m_false_flag)
        self.assertTrue(any("eval_only=true" in e for e in errs_false))

        # Proprietary is STILL rejected even on research_nc_eval_only
        m_prop = _make_manifest(
            [_make_page_record(music_license="Proprietary", track="research_nc_eval_only", eval_only=True)],
            track="research_nc_eval_only",
            eval_only=True,
        )
        errs_prop = validate_manifest(m_prop)
        self.assertTrue(any("non-permissive music_license" in e for e in errs_prop))

    def test_cdr_engraver_dominance_over_15_pct_rejected(self) -> None:
        # 10 CDR pages: 2 from same engraver = 20% > 15% -> FAIL
        pages_bad = [
            _make_page_record(
                page_id=f"cdr_{i}",
                work_id=f"w_{i}",
                domain="CDR",
                engraver="EngraverA" if i < 2 else f"Engraver_{i}",
            )
            for i in range(10)
        ]
        errs = validate_manifest(_make_manifest(pages_bad))
        self.assertTrue(any("exceeds 15% cap" in e for e in errs), errs)

        # 20 CDR pages: max 3 from any engraver = 15.0% -> PASS
        engravers = ["E1", "E2", "E3", "E4", "E5", "E6", "E7"]
        pages_ok = [
            _make_page_record(
                page_id=f"cdr_{i}",
                work_id=f"w_{i}",
                domain="CDR",
                engraver=engravers[i % len(engravers)],
            )
            for i in range(20)
        ]
        self.assertEqual(validate_manifest(_make_manifest(pages_ok)), [])

    def test_dev_manifest_requires_canary(self) -> None:
        m = _make_manifest([_make_page_record()], split="dev-v1", canary=None)
        errs = validate_manifest(m)
        self.assertTrue(any("canary" in e for e in errs))

        # test-v1 does not require canary
        m_test = _make_manifest([_make_page_record(split="test-v1")], split="test-v1", canary=None)
        self.assertEqual(validate_manifest(m_test), [])

    def test_negative_control_allows_null_staff_space(self) -> None:
        rec = _make_page_record(is_negative_control=True, staff_space_px=None)
        m = _make_manifest([rec])
        self.assertEqual(validate_manifest(m), [])

    def test_disjoint_splits_catches_work_and_writer_overlap(self) -> None:
        dev = _make_manifest(
            [
                _make_page_record(page_id="d1", work_id="shared-work", split="dev-v1", domain="CDR"),
                _make_page_record(page_id="d2", work_id="w-dev-hw", split="dev-v1", domain="HWM", writer_id="copyist-42"),
            ],
            split="dev-v1",
        )
        test = _make_manifest(
            [
                _make_page_record(page_id="t1", work_id="shared-work", split="test-v1", domain="RWP"),
                _make_page_record(page_id="t2", work_id="w-test-hw", split="test-v1", domain="HWM", writer_id="copyist-42"),
            ],
            split="test-v1",
            canary=None,
        )
        errs = validate_disjoint_splits(dev, test)
        self.assertTrue(any("work_id" in e and "shared-work" in e for e in errs))
        self.assertTrue(any("writer_id" in e and "copyist-42" in e for e in errs))

        # Clean disjoint splits pass
        test_clean = _make_manifest(
            [
                _make_page_record(page_id="t1", work_id="unique-work", split="test-v1", domain="RWP"),
                _make_page_record(page_id="t2", work_id="w-test-hw", split="test-v1", domain="HWM", writer_id="copyist-99"),
            ],
            split="test-v1",
            canary=None,
        )
        self.assertEqual(validate_disjoint_splits(dev, test_clean), [])

    def test_validate_report_valid_and_invalid(self) -> None:
        pages = [
            PageMetrics(f"p_{d}", f"w_{d}", d, "T2", 0.9, 0.9, 0.9, 0.9, 10.0, 100.0)
            for d in DOMAINS
        ]
        rep = build_report(pages, system_name="sys-a", split="dev-v1")
        rep["layer1_headline"]["tier_weighted_ns"] = 88.5
        rep["layer1_headline"]["ns_ci95"] = [85.0, 92.0]
        self.assertEqual(validate_report(rep), [])

        bad = json.loads(json.dumps(rep))
        bad["layer1_headline"]["ns_score"] = 120.0
        bad["layer1_headline"]["ns_ci95"] = [95.0, 80.0]
        errs = validate_report(bad)
        self.assertTrue(any("ns_score" in e for e in errs))
        self.assertTrue(any("ns_ci95" in e for e in errs))


class TestCompareReports(unittest.TestCase):
    """Verify Layer 4 compare(A, B) paired permutation regression detection."""

    def test_detects_single_domain_regression_even_when_overall_ns_rises(self) -> None:
        # 8 pages per domain: B improves massively on CDR/ALS/RWP (+15 each),
        # so overall NS rises, but B drops significantly on HWM (-6 each).
        dom_a = {
            "CDR": [(f"cdr_{i}", "T1", 0.70) for i in range(8)],
            "ALS": [(f"als_{i}", "T1", 0.70) for i in range(8)],
            "RWP": [(f"rwp_{i}", "T2", 0.70) for i in range(8)],
            "HWM": [(f"hwm_{i}", "T2", 0.80) for i in range(8)],
        }
        dom_b = {
            "CDR": [(f"cdr_{i}", "T1", 0.95) for i in range(8)],
            "ALS": [(f"als_{i}", "T1", 0.95) for i in range(8)],
            "RWP": [(f"rwp_{i}", "T2", 0.95) for i in range(8)],
            "HWM": [(f"hwm_{i}", "T2", 0.74) for i in range(8)],
        }
        rep_a = _make_report_with_pages("baseline", dom_a)
        rep_b = _make_report_with_pages("candidate", dom_b)

        diff = compare_reports(rep_a, rep_b, n_perm=1000, seed=42)
        self.assertGreater(diff["headline"]["ns_delta"], 0.0)
        self.assertFalse(diff["passed"])
        self.assertTrue(
            any(r["scope"] == "domain" and r["key"] == "HWM" and r["significant"] for r in diff["regressions"]),
            diff["regressions"],
        )
        self.assertLess(diff["domain_deltas"]["HWM"]["p_value"], 0.05)

        table = format_comparison(diff)
        self.assertIn("REGRESSIONS DETECTED", table)
        self.assertIn("HWM", table)

    def test_detects_single_tier_regression_even_when_overall_ns_rises(self) -> None:
        # B improves on T0/T1 across all domains (NS rises), but drops significantly on T3.
        dom_a = {
            d: [(f"{d}_t0_{i}", "T0", 0.70) for i in range(4)]
            + [(f"{d}_t3_{i}", "T3", 0.85) for i in range(4)]
            for d in DOMAINS
        }
        dom_b = {
            d: [(f"{d}_t0_{i}", "T0", 0.95) for i in range(4)]
            + [(f"{d}_t3_{i}", "T3", 0.78) for i in range(4)]
            for d in DOMAINS
        }
        rep_a = _make_report_with_pages("baseline", dom_a)
        rep_b = _make_report_with_pages("candidate", dom_b)

        diff = compare_reports(rep_a, rep_b, n_perm=1000, seed=42)
        self.assertGreater(diff["headline"]["ns_delta"], 0.0)
        self.assertFalse(diff["passed"])
        self.assertTrue(
            any(r["scope"] == "tier" and r["key"] == "T3" and r["significant"] for r in diff["regressions"]),
            diff["regressions"],
        )

    def test_pure_improvement_passes_with_zero_regressions(self) -> None:
        dom_a = {d: [(f"{d}_{i}", "T1", 0.80) for i in range(6)] for d in DOMAINS}
        dom_b = {d: [(f"{d}_{i}", "T1", 0.90) for i in range(6)] for d in DOMAINS}
        rep_a = _make_report_with_pages("baseline", dom_a)
        rep_b = _make_report_with_pages("candidate", dom_b)
        diff = compare_reports(rep_a, rep_b, n_perm=500, seed=7)
        self.assertTrue(diff["passed"])
        self.assertEqual(diff["regressions"], [])
        self.assertGreater(diff["headline"]["ns_delta"], 0.0)


class TestCLISubcommands(unittest.TestCase):
    """End-to-end tests for score, compare, validate-manifest, validate-splits, validate-report."""

    def test_cli_all_subcommands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)

            # 1. score --include-per-page + validate-report
            pages_a = [
                {
                    "page_id": f"{d}_{i}",
                    "work_id": f"w_{d}_{i}",
                    "domain": d,
                    "tier": "T2",
                    "s_note": 0.85,
                    "s_struct": 0.85,
                    "s_expr": 0.85,
                    "s_play": 0.85,
                    "wfpp": 15.0,
                    "transcribe_cost": 100.0,
                }
                for d in DOMAINS
                for i in range(6)
            ]
            pages_b = [
                dict(p, s_note=0.92, s_struct=0.92, s_expr=0.92, s_play=0.92, wfpp=8.0)
                for p in pages_a
            ]
            res_a = root / "res_a.json"
            res_b = root / "res_b.json"
            rep_a = root / "rep_a.json"
            rep_b = root / "rep_b.json"
            diff_out = root / "diff.json"
            res_a.write_text(json.dumps({"system_name": "sys-a", "pages": pages_a}), encoding="utf-8")
            res_b.write_text(json.dumps({"system_name": "sys-b", "pages": pages_b}), encoding="utf-8")

            self.assertEqual(cli_main(["score", str(res_a), "--include-per-page", "--json-out", str(rep_a)]), 0)
            self.assertEqual(cli_main(["score", str(res_b), "--include-per-page", "--json-out", str(rep_b)]), 0)
            self.assertEqual(cli_main(["validate-report", str(rep_a)]), 0)

            # 2. compare (pass when B > A, fail when A > B)
            self.assertEqual(cli_main(["compare", str(rep_a), str(rep_b), "--json-out", str(diff_out)]), 0)
            self.assertTrue(json.loads(diff_out.read_text(encoding="utf-8"))["passed"])
            self.assertEqual(cli_main(["compare", str(rep_b), str(rep_a)]), 1)

            # 3. validate-manifest & validate-splits
            dev_m = root / "dev.json"
            test_m = root / "test.json"
            bad_m = root / "bad.json"
            dev_m.write_text(
                json.dumps(_make_manifest([_make_page_record(page_id="d1", work_id="w1", split="dev-v1")])),
                encoding="utf-8",
            )
            test_m.write_text(
                json.dumps(
                    _make_manifest(
                        [_make_page_record(page_id="t1", work_id="w2", split="test-v1")],
                        split="test-v1",
                        canary=None,
                    )
                ),
                encoding="utf-8",
            )
            bad_m.write_text(
                json.dumps(_make_manifest([_make_page_record(music_license="CC-BY-NC-4.0")])),
                encoding="utf-8",
            )

            self.assertEqual(cli_main(["validate-manifest", str(dev_m)]), 0)
            self.assertEqual(cli_main(["validate-manifest", str(bad_m)]), 1)
            self.assertEqual(cli_main(["validate-splits", str(dev_m), str(test_m)]), 0)


if __name__ == "__main__":
    unittest.main()
