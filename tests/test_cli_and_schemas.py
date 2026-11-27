"""Unit tests for OMR-NorthStar CLI, compare(A, B), governance validators, and JSON Schemas."""

from __future__ import annotations

import json
import math
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest
from typing import Any

import omr_northstar
from omr_northstar.cli import main as cli_main
from omr_northstar.compare import _paired_permutation_pvalue, compare_reports, format_comparison
from omr_northstar.scoring import DOMAINS, PageMetrics, build_report, compute_page_quality
from omr_northstar.validate import (
    NC_EVAL_ONLY_LICENSES,
    PERMISSIVE_LICENSES,
    validate_disjoint_splits,
    validate_manifest,
    validate_page_records,
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
        self.assertIn("nc_gate", data["properties"])
        self.assertNotIn("nc_gate", data["properties"]["layer2_matrix_and_deltas"]["properties"])
        pd_props = data["properties"]["layer2_matrix_and_deltas"]["properties"]["paired_deltas"]["properties"]
        self.assertEqual(pd_props["delta_engraver"]["minimum"], 0)
        self.assertEqual(pd_props["delta_engraver"]["maximum"], 100)
        self.assertEqual(pd_props["delta_photo"]["minimum"], -100)
        self.assertEqual(pd_props["delta_photo"]["maximum"], 100)
        self.assertEqual(pd_props["delta_handwriting"]["minimum"], -100)
        self.assertEqual(pd_props["delta_handwriting"]["maximum"], 100)

    def test_manifest_schema_envelope_and_tracks(self) -> None:
        data = json.loads((REPO_ROOT / "schemas" / "manifest.schema.json").read_text(encoding="utf-8"))
        props = data["properties"]
        for key in (
            "schema_version",
            "manifest_version",
            "benchmark_version",
            "split",
            "canary",
            "track",
            "eval_only",
            "license",
            "ground_truth_files",
            "pages",
        ):
            self.assertIn(key, props)
        self.assertEqual(props["pages"]["items"]["$ref"], "page_metadata.schema.json")
        # Pilot manifest top-level keys are all allowed under additionalProperties: false
        pilot_path = REPO_ROOT / "pilot_dev_v0_1" / "manifest.json"
        if pilot_path.exists():
            pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
            self.assertTrue(set(pilot.keys()).issubset(set(props.keys())))

    def test_page_metadata_schema_requires_staff_space_on_music_pages(self) -> None:
        data = json.loads((REPO_ROOT / "schemas" / "page_metadata.schema.json").read_text(encoding="utf-8"))
        all_of = data.get("allOf", [])
        self.assertTrue(
            any(
                isinstance(rule, dict)
                and rule.get("else", {}).get("required") == ["staff_space_px"]
                for rule in all_of
            )
        )


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

    def test_validate_rejects_null_subobjects_unhashable_nan_and_bypasses(self) -> None:
        # Null / wrong-type page sub-objects must NOT silently bypass
        for field in ("geometry", "photometry", "optics_codec", "provenance", "license"):
            rec = _make_page_record(**{field: None})
            errs = validate_page_records([rec])
            self.assertTrue(any(f"{field} must be an object" in e for e in errs), (field, errs))

        # Unhashable list/dict values must NOT raise TypeError
        unhashable_rec = _make_page_record(
            domain=["RWP"],
            complexity_tier={"t": "T2"},
            visual_tier=["M"],
            visual_severity=["V2"],
            repertoire_stratum=["T2.1"],
            paired_protocol=["A"],
            track=["core_permissive"],
            stress_test_families=[["1A"], "9Z"],
        )
        unhashable_rec["optics_codec"]["bit_depth"] = [24]
        unhashable_rec["optics_codec"]["codec"] = ""
        unhashable_rec["provenance"]["gt_verification"] = ["born_digital_canonical"]
        unhashable_rec["provenance"]["source_type"] = "   "
        unhashable_rec["license"]["music_license"] = ["CC0-1.0"]
        unhashable_rec["license"]["image_license"] = {"lic": "CC0-1.0"}
        errs = validate_page_records([unhashable_rec])
        self.assertTrue(any("invalid domain" in e for e in errs))
        self.assertTrue(any("bit_depth" in e for e in errs))
        self.assertTrue(any("optics_codec.codec" in e for e in errs))
        self.assertTrue(any("provenance.source_type" in e for e in errs))
        self.assertTrue(any("stress_test_families" in e for e in errs))

        # Non-finite NaN / Inf numbers must be rejected
        nan_rec = _make_page_record()
        nan_rec["geometry"]["rotation_deg"] = float("nan")
        nan_rec["photometry"]["illumination_gradient_ev"] = float("inf")
        nan_rec["staff_space_px"]["median"] = float("nan")
        errs_nan = validate_page_records([nan_rec])
        self.assertTrue(any("rotation_deg" in e for e in errs_nan))
        self.assertTrue(any("illumination_gradient_ev" in e for e in errs_nan))
        self.assertTrue(any("staff_space_px.median" in e for e in errs_nan))

        # Null bypasses on staff_space_px (non-NC), strict bool on is_negative_control & eval_only
        null_ss = _make_page_record(staff_space_px=None, is_negative_control=False)
        self.assertTrue(any("staff_space_px" in e for e in validate_page_records([null_ss])))
        bad_bools = _make_page_record(is_negative_control="true", eval_only=1)
        errs_bools = validate_page_records([bad_bools])
        self.assertTrue(any("is_negative_control must be boolean" in e for e in errs_bools))
        self.assertTrue(any("eval_only must be boolean" in e for e in errs_bools))

        # Null bypasses on manifest schema_version, split, and empty benchmark_version
        bad_m = _make_manifest([_make_page_record()], schema_version=None, split=None, benchmark_version="")
        errs_m = validate_manifest(bad_m)
        self.assertTrue(any("schema_version" in e for e in errs_m))
        self.assertTrue(any("invalid split" in e for e in errs_m))
        self.assertTrue(any("benchmark_version" in e for e in errs_m))

        # Null/wrong-type report layers, ns_ci95 out of [0, 100], layer3 sum != 100%, NaN
        pages = [
            PageMetrics(f"p_{d}", f"w_{d}", d, "T2", 0.9, 0.9, 0.9, 0.9, 10.0, 100.0)
            for d in DOMAINS
        ]
        rep = build_report(pages, system_name="sys-a", split="dev-v1")
        for layer in (
            "layer1_headline",
            "layer2_matrix_and_deltas",
            "layer3_error_taxonomy",
            "layer4_forensic_pack",
        ):
            bad_r = json.loads(json.dumps(rep))
            bad_r[layer] = None
            errs_r = validate_report(bad_r)
            self.assertTrue(any(f"{layer} must be an object" in e for e in errs_r), (layer, errs_r))

        bad_r2 = json.loads(json.dumps(rep))
        bad_r2["schema_version"] = None
        bad_r2["benchmark_version"] = " "
        bad_r2["layer1_headline"]["ns_ci95"] = [-5.0, 105.0]
        bad_r2["layer1_headline"]["m_geom"] = float("nan")
        bad_r2["layer3_error_taxonomy"]["pitch_accidental_pct"] = 40.0
        bad_r2["layer3_error_taxonomy"]["rhythm_meter_pct"] = 20.0
        errs_r2 = validate_report(bad_r2)
        self.assertTrue(any("schema_version" in e for e in errs_r2))
        self.assertTrue(any("benchmark_version" in e for e in errs_r2))
        self.assertTrue(any("ns_ci95" in e for e in errs_r2))
        self.assertTrue(any("m_geom" in e for e in errs_r2))
        self.assertTrue(any("sum to 100%" in e for e in errs_r2))

    def test_disjoint_splits_rwp_unseen_capture_device_check(self) -> None:
        dev = _make_manifest(
            [
                _make_page_record(
                    page_id="d1",
                    work_id="w_d1",
                    split="dev-v1",
                    domain="RWP",
                    provenance={
                        "source_type": "in_house_phone",
                        "capture_device_id": "phone-A",
                        "gt_verification": "double_blind_human_adjudicated",
                    },
                ),
                _make_page_record(
                    page_id="d2",
                    work_id="w_d2",
                    split="dev-v1",
                    domain="RWP",
                    provenance={
                        "source_type": "in_house_phone",
                        "capture_device_id": "phone-B",
                        "gt_verification": "double_blind_human_adjudicated",
                    },
                ),
            ],
            split="dev-v1",
        )
        # Only 1 of 3 test devices unseen (33.3% < 50%) -> FAIL
        test_fail = _make_manifest(
            [
                _make_page_record(
                    page_id=f"t{i}",
                    work_id=f"w_t{i}",
                    split="test-v1",
                    domain="RWP",
                    provenance={
                        "source_type": "in_house_phone",
                        "capture_device_id": dev_id,
                        "gt_verification": "double_blind_human_adjudicated",
                    },
                )
                for i, dev_id in enumerate(["phone-A", "phone-B", "phone-C"])
            ],
            split="test-v1",
            canary=None,
        )
        errs_fail = validate_disjoint_splits(dev, test_fail)
        self.assertTrue(any("capture_device_id" in e for e in errs_fail), errs_fail)

        # 1 of 2 test devices unseen (50.0% >= 50%) -> PASS
        test_ok = _make_manifest(
            [
                _make_page_record(
                    page_id=f"t{i}",
                    work_id=f"w_t{i}",
                    split="test-v1",
                    domain="RWP",
                    provenance={
                        "source_type": "in_house_phone",
                        "capture_device_id": dev_id,
                        "gt_verification": "double_blind_human_adjudicated",
                    },
                )
                for i, dev_id in enumerate(["phone-A", "phone-C"])
            ],
            split="test-v1",
            canary=None,
        )
        self.assertEqual(validate_disjoint_splits(dev, test_ok), [])


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

    def test_work_clustered_permutation_prevents_pseudo_replication(self) -> None:
        # 8 pages in HWM all belonging to a SINGLE work_id:
        # Page-level permutation would yield p < 0.01, but work-clustered permutation
        # has n_clusters = 1, so two-sided p >= 0.5 (cannot be < 0.05).
        dom_a = {d: [(f"{d}_{i}", "T1", 0.80) for i in range(8)] for d in DOMAINS}
        dom_b = {
            "CDR": [(f"CDR_{i}", "T1", 0.90) for i in range(8)],
            "ALS": [(f"ALS_{i}", "T1", 0.90) for i in range(8)],
            "RWP": [(f"RWP_{i}", "T1", 0.90) for i in range(8)],
            "HWM": [(f"HWM_{i}", "T1", 0.70) for i in range(8)],
        }
        rep_a = _make_report_with_pages("baseline", dom_a)
        rep_b = _make_report_with_pages("candidate", dom_b)
        for rep in (rep_a, rep_b):
            for p in rep["layer4_forensic_pack"]["per_page_scores"]:
                if p["domain"] == "HWM":
                    p["work_id"] = "single_shared_work"
        diff = compare_reports(rep_a, rep_b, n_perm=500, seed=42)
        self.assertGreaterEqual(diff["domain_deltas"]["HWM"]["p_value"], 0.4)
        self.assertFalse(diff["domain_deltas"]["HWM"]["significant"])
        self.assertEqual(diff["regressions"], [])

    def test_compare_guards_sign_inversion_nan_params_and_tier_fallback(self) -> None:
        # 1. NaN/Inf diffs raise ValueError
        with self.assertRaises(ValueError):
            _paired_permutation_pvalue([1.0, float("nan")], n_perm=100)
        with self.assertRaises(ValueError):
            _paired_permutation_pvalue([1.0, float("inf")], n_perm=100)

        # 2. Invalid alpha / n_perm / unpaired_drop_threshold raise ValueError
        dom = {d: [(f"{d}_0", "T1", 0.85)] for d in DOMAINS}
        rep_a = _make_report_with_pages("a", dom)
        rep_b = _make_report_with_pages("b", dom)
        with self.assertRaises(ValueError):
            compare_reports(rep_a, rep_b, alpha=0.0)
        with self.assertRaises(ValueError):
            compare_reports(rep_a, rep_b, alpha=1.0)
        with self.assertRaises(ValueError):
            compare_reports(rep_a, rep_b, n_perm=0)
        with self.assertRaises(ValueError):
            compare_reports(rep_a, rep_b, unpaired_drop_threshold=-1.0)

        # 3. Sign-inversion guard: if stratum_scores drop (<0) but paired diffs are positive (>0),
        # must NOT flag a significant regression.
        dom_pos_a = {d: [(f"{d}_{i}", "T1", 0.70) for i in range(8)] for d in DOMAINS}
        dom_pos_b = {d: [(f"{d}_{i}", "T1", 0.90) for i in range(8)] for d in DOMAINS}
        ra_inv = _make_report_with_pages("a", dom_pos_a)
        rb_inv = _make_report_with_pages("b", dom_pos_b)
        rb_inv["layer1_headline"]["stratum_scores"]["CDR"] = (
            ra_inv["layer1_headline"]["stratum_scores"]["CDR"] - 5.0
        )
        diff_inv = compare_reports(ra_inv, rb_inv, n_perm=500, seed=42)
        self.assertFalse(
            any(r["scope"] == "domain" and r["key"] == "CDR" for r in diff_inv["regressions"])
        )

        # 4. tier_deltas fallback when matrix is absent & tier mismatches exist (never > 100.0)
        ra_tier = _make_report_with_pages("a", dom_pos_a)
        rb_tier = _make_report_with_pages("b", dom_pos_b)
        ra_tier["layer2_matrix_and_deltas"]["tier_domain_matrix"] = {}
        rb_tier["layer2_matrix_and_deltas"]["tier_domain_matrix"] = {}
        rb_tier["layer4_forensic_pack"]["per_page_scores"][0]["tier"] = "T2"
        diff_tier = compare_reports(ra_tier, rb_tier, n_perm=200, seed=42)
        self.assertLessEqual(diff_tier["tier_deltas"]["T1"]["score_a"], 100.0)
        self.assertLessEqual(diff_tier["tier_deltas"]["T1"]["score_b"], 100.0)

        # 5. n_only_a, n_only_b, split_mismatch recorded
        ra_m = _make_report_with_pages("a", dom)
        rb_m = _make_report_with_pages("b", dom)
        rb_m["split"] = "test-v1"
        ra_m["layer4_forensic_pack"]["per_page_scores"].append(
            {"page_id": "only_a", "work_id": "wa", "domain": "CDR", "tier": "T1", "q_score": 80.0}
        )
        rb_m["layer4_forensic_pack"]["per_page_scores"].append(
            {"page_id": "only_b", "work_id": "wb", "domain": "CDR", "tier": "T1", "q_score": 85.0}
        )
        diff_m = compare_reports(ra_m, rb_m, n_perm=100, seed=42)
        self.assertEqual(diff_m["n_only_a"], 1)
        self.assertEqual(diff_m["n_only_b"], 1)
        self.assertTrue(diff_m["split_mismatch"])


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

    def test_cli_split_override_bom_mkdir_errors_cp1252_and_exports(self) -> None:
        # Top-level package re-exports
        for sym in (
            "compare_reports",
            "format_comparison",
            "validate_manifest",
            "validate_page_records",
            "validate_disjoint_splits",
            "validate_report",
        ):
            self.assertTrue(hasattr(omr_northstar, sym), f"Missing export: {sym}")

        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            pages = [
                {
                    "page_id": f"{d}_0",
                    "work_id": f"w_{d}_0",
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
            ]
            # Write UTF-8 with BOM to verify utf-8-sig reading
            res_bom = root / "res_bom.json"
            res_bom.write_bytes(
                b"\xef\xbb\xbf"
                + json.dumps({"system_name": "sys-bom", "split": "dev-v1", "pages": pages}).encode("utf-8")
            )
            nested_out = root / "nested" / "dir" / "rep.json"

            # --split test-v1 must override "split": "dev-v1" in dict payload
            rc = cli_main(["score", str(res_bom), "--split", "test-v1", "--include-per-page", "--json-out", str(nested_out)])
            self.assertEqual(rc, 0)
            self.assertTrue(nested_out.exists())
            rep_data = json.loads(nested_out.read_text(encoding="utf-8"))
            self.assertEqual(rep_data["split"], "test-v1")

            # Invalid --split returns exit code 1 cleanly
            self.assertEqual(cli_main(["score", str(res_bom), "--split", "not-a-split"]), 1)

            # Missing file & malformed JSON return exit code 1 cleanly (no traceback)
            self.assertEqual(cli_main(["score", str(root / "missing.json")]), 1)
            bad_json = root / "bad.json"
            bad_json.write_text("{not valid json", encoding="utf-8")
            self.assertEqual(cli_main(["validate-report", str(bad_json)]), 1)

            # Windows cp1252 subprocess check for `compare` output
            env = dict(os.environ)
            env["PYTHONIOENCODING"] = "cp1252"
            env["CUDA_VISIBLE_DEVICES"] = ""
            proc = subprocess.run(
                [sys.executable, "-m", "omr_northstar", "compare", str(nested_out), str(nested_out)],
                cwd=str(REPO_ROOT),
                env=env,
                capture_output=True,
                text=True,
                encoding="cp1252",
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn("M_geom Delta:", proc.stdout)


if __name__ == "__main__":
    unittest.main()
