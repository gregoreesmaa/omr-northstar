"""Unit tests for OMR-NorthStar reference scorer, CLI, and JSON Schemas."""

from __future__ import annotations

import json
import pathlib
import tempfile
import unittest

from omr_northstar.cli import format_scorecard, main as cli_main, validate_manifest_records
from omr_northstar.scoring import (
    DOMAINS,
    PageMetrics,
    build_report,
    compute_catastrophic_penalties,
    compute_effective_measure_cost,
    compute_northstar_score,
    compute_ntsr,
    compute_nupr,
    compute_page_quality,
    compute_paired_deltas,
    compute_prior_hallucination_rate,
    compute_stratum_scores,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


class TestNorthStarCalibration(unittest.TestCase):
    """Verify exact anti-gaming calibration examples from docs/04-metrics-and-human-utility.md."""

    def test_clean_only_specialist_collapses(self) -> None:
        scores = {"CDR": 98.0, "ALS": 92.0, "RWP": 35.0, "HWM": 10.0}
        cfr = {"CDR": 0.0, "ALS": 0.02, "RWP": 0.20, "HWM": 0.60}
        res = compute_northstar_score(scores, cfr)
        self.assertAlmostEqual(res.m_geom, 42.1473, places=3)
        self.assertAlmostEqual(res.worst_stratum_q, 10.0, places=4)
        self.assertAlmostEqual(res.cfr_max, 0.60, places=4)
        self.assertAlmostEqual(res.ns_score, 23.8774, places=3)

    def test_printed_photo_strong_hw_weak(self) -> None:
        scores = {"CDR": 95.0, "ALS": 88.0, "RWP": 80.0, "HWM": 30.0}
        cfr = {"CDR": 0.0, "ALS": 0.01, "RWP": 0.05, "HWM": 0.25}
        res = compute_northstar_score(scores, cfr)
        self.assertAlmostEqual(res.m_geom, 66.9277, places=3)
        self.assertAlmostEqual(res.ns_score, 50.4838, places=3)

    def test_balanced_universal_system(self) -> None:
        scores = {"CDR": 90.0, "ALS": 84.0, "RWP": 76.0, "HWM": 68.0}
        cfr = {"CDR": 0.0, "ALS": 0.01, "RWP": 0.02, "HWM": 0.04}
        res = compute_northstar_score(scores, cfr)
        self.assertAlmostEqual(res.m_geom, 79.0609, places=3)
        self.assertAlmostEqual(res.ns_score, 74.7698, places=3)

    def test_missing_stratum_penalized_to_zero(self) -> None:
        page = PageMetrics(
            page_id="p1",
            work_id="w1",
            domain="CDR",
            tier="T0",
            s_note=1.0,
            s_struct=1.0,
            s_expr=1.0,
            s_play=1.0,
            wfpp=0.0,
            transcribe_cost=50.0,
        )
        res = compute_page_quality(page)
        s_scores, s_cfr = compute_stratum_scores([res])
        self.assertEqual(s_scores["CDR"], 100.0)
        self.assertEqual(s_scores["HWM"], 0.0)
        self.assertEqual(s_cfr["HWM"], 1.0)


class TestCatastrophicPenalties(unittest.TestCase):
    """Verify all 5 catastrophic failure triggers and exact penalty multipliers."""

    def _base_page(self, **overrides: object) -> PageMetrics:
        kwargs: dict[str, object] = dict(
            page_id="p1",
            work_id="w1",
            domain="RWP",
            tier="T2",
            s_note=0.90,
            s_struct=0.90,
            s_expr=0.90,
            s_play=0.92,
            wfpp=10.0,
            transcribe_cost=100.0,
            parseable=True,
            n_gt=100,
            n_pred=100,
            k_dropped_systems=0,
            has_repetition_loop=False,
            bsvr=0.95,
            m_gt=16,
            m_pred=16,
        )
        kwargs.update(overrides)
        return PageMetrics(**kwargs)  # type: ignore[arg-type]

    def test_unparseable_zeroes_quality_and_sets_wfpp(self) -> None:
        p = self._base_page(parseable=False)
        res = compute_page_quality(p)
        self.assertTrue(res.catastrophic_failure)
        self.assertIn("unparseable", res.cf_reasons)
        self.assertEqual(res.q_score, 0.0)
        self.assertAlmostEqual(res.effective_wfpp, 120.0)

    def test_empty_output_zeroes_quality(self) -> None:
        p = self._base_page(n_gt=80, n_pred=0)
        res = compute_page_quality(p)
        self.assertTrue(res.catastrophic_failure)
        self.assertIn("empty_output", res.cf_reasons)
        self.assertEqual(res.q_score, 0.0)

    def test_dropped_systems_multiplicative_penalty(self) -> None:
        p0 = self._base_page(k_dropped_systems=0)
        p2 = self._base_page(k_dropped_systems=2)
        r0 = compute_page_quality(p0)
        r2 = compute_page_quality(p2)
        self.assertFalse(r0.catastrophic_failure)
        self.assertTrue(r2.catastrophic_failure)
        self.assertIn("dropped_system", r2.cf_reasons)
        self.assertAlmostEqual(r2.q_score, r0.q_raw * (0.85 ** 2), places=6)

    def test_hallucination_loop_penalty(self) -> None:
        p = self._base_page(n_gt=100, n_pred=300)
        res = compute_page_quality(p)
        self.assertTrue(res.catastrophic_failure)
        self.assertIn("hallucination_loop", res.cf_reasons)
        # LoopPen = min(1, 1.5 * 100 / 300) = 0.5
        self.assertAlmostEqual(res.penalty_multiplier, 0.5, places=6)
        self.assertAlmostEqual(res.q_score, res.q_raw * 0.5, places=6)

    def test_structural_collapse_flags_cf(self) -> None:
        p_bsvr = self._base_page(bsvr=0.40)
        is_cf, reasons, _, _ = compute_catastrophic_penalties(p_bsvr)
        self.assertTrue(is_cf)
        self.assertIn("structural_collapse", reasons)

        p_bars = self._base_page(m_gt=20, m_pred=14)
        is_cf2, reasons2, _, _ = compute_catastrophic_penalties(p_bars)
        self.assertTrue(is_cf2)
        self.assertIn("structural_collapse", reasons2)


class TestHumanUtilityMetrics(unittest.TestCase):
    """Verify Rational Musician Re-Entry Cap, NTSR, and NUPR."""

    def test_effective_measure_cost_cap(self) -> None:
        self.assertAlmostEqual(compute_effective_measure_cost(8.0, 15.0), 8.0)
        self.assertAlmostEqual(compute_effective_measure_cost(60.0, 15.0), 16.5)

    def test_ntsr_and_nupr(self) -> None:
        self.assertAlmostEqual(compute_ntsr(0.0, 100.0), 1.0)
        self.assertAlmostEqual(compute_ntsr(30.0, 100.0), 0.70)
        self.assertAlmostEqual(compute_ntsr(120.0, 100.0), -0.20)
        self.assertAlmostEqual(compute_nupr([0.85, 0.70, 0.0, -0.15]), 0.50)


class TestPairedDeltasAndPHR(unittest.TestCase):
    """Verify paired robustness deltas and Counterfactual Prior-Hallucination Rate."""

    def test_paired_deltas(self) -> None:
        pages = [
            PageMetrics("p_cdr_1", "w1", "CDR", "T2", 0.95, 0.95, 0.95, 0.96, 5.0, 100.0, paired_anchor_id="a1", engraver="Dorico"),
            PageMetrics("p_cdr_2", "w1", "CDR", "T2", 0.90, 0.90, 0.90, 0.91, 10.0, 100.0, paired_anchor_id="a1", engraver="LilyPond"),
            PageMetrics("p_rwp_1", "w1", "RWP", "T2", 0.85, 0.85, 0.85, 0.86, 15.0, 100.0, paired_anchor_id="a1"),
            PageMetrics("p_hwm_1", "w1", "HWM", "T2", 0.75, 0.75, 0.75, 0.76, 25.0, 100.0, paired_anchor_id="a1"),
        ]
        results = [compute_page_quality(p) for p in pages]
        deltas = compute_paired_deltas(results)
        self.assertIsNotNone(deltas["delta_engraver"])
        self.assertAlmostEqual(deltas["delta_engraver"], 5.0, places=4)
        # CDR mean = 92.5, RWP = 85.0 -> delta_photo = -7.5
        self.assertAlmostEqual(deltas["delta_photo"], -7.5, places=4)
        # HWM = 75.0, CDR mean = 92.5 -> delta_handwriting = -17.5
        self.assertAlmostEqual(deltas["delta_handwriting"], -17.5, places=4)

    def test_phr_calculation(self) -> None:
        self.assertAlmostEqual(compute_prior_hallucination_rate(50, 1), 0.02)
        self.assertAlmostEqual(compute_prior_hallucination_rate(50, 0), 0.0)
        with self.assertRaises(ValueError):
            compute_prior_hallucination_rate(0, 0)
        with self.assertRaises(ValueError):
            compute_prior_hallucination_rate(10, 11)


class TestSchemasAndCLI(unittest.TestCase):
    """Verify JSON Schema validity and end-to-end CLI scoring + manifest validation."""

    def test_schemas_are_valid_json_draft_2020_12(self) -> None:
        schema_dir = REPO_ROOT / "schemas"
        schema_files = sorted(schema_dir.glob("*.json"))
        self.assertEqual(len(schema_files), 2)
        for sf in schema_files:
            data = json.loads(sf.read_text(encoding="utf-8"))
            self.assertEqual(
                data.get("$schema"),
                "https://json-schema.org/draft/2020-12/schema",
                f"{sf.name} must declare Draft 2020-12",
            )
            self.assertEqual(data.get("type"), "object")
            self.assertIn("properties", data)
            self.assertIn("required", data)

    def test_report_and_cli_end_to_end(self) -> None:
        sample_pages = [
            {
                "page_id": f"p_{d}",
                "work_id": f"w_{d}",
                "domain": d,
                "tier": "T2",
                "s_note": 0.90,
                "s_struct": 0.85,
                "s_expr": 0.80,
                "s_play": 0.92,
                "wfpp": 12.0,
                "transcribe_cost": 100.0,
                "latency_s": 1.25,
            }
            for d in DOMAINS
        ]
        pages = [PageMetrics.from_dict(p) for p in sample_pages]
        report = build_report(pages, system_name="demo-omr", split="dev-v1")
        self.assertEqual(report["schema_version"], "1.0.0")
        self.assertGreater(report["layer1_headline"]["ns_score"], 80.0)
        text = format_scorecard(report)
        self.assertIn("NS Score", text)
        self.assertIn("demo-omr", text)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            in_file = tmp_path / "results.json"
            out_file = tmp_path / "report.json"
            in_file.write_text(json.dumps({"system_name": "cli-test", "pages": sample_pages}), encoding="utf-8")
            rc = cli_main(["score", str(in_file), "--json-out", str(out_file)])
            self.assertEqual(rc, 0)
            written = json.loads(out_file.read_text(encoding="utf-8"))
            self.assertEqual(written["system_name"], "cli-test")

    def test_manifest_validator(self) -> None:
        valid_rec = {
            "page_id": "ns-dev-0001",
            "work_id": "work-0001",
            "split": "dev-v1",
            "domain": "RWP",
            "complexity_tier": "T2",
            "visual_tier": "M",
            "repertoire_stratum": "T2.1",
            "staff_space_px": {"median": 10.5},
            "geometry": {"rotation_deg": 2.1, "keystone_x_deg": 10.0, "keystone_y_deg": -3.0},
            "photometry": {"weber_contrast": 0.62, "illumination_gradient_ev": 1.1},
            "optics_codec": {"blur_sigma_px": 1.1, "bit_depth": 24, "codec": "jpeg", "quality_factor": 82},
            "provenance": {"source_type": "in_house_phone", "gt_verification": "double_blind_human_adjudicated"},
            "license": {"music_license": "Public-Domain", "image_license": "CC-BY-4.0"},
        }
        self.assertEqual(validate_manifest_records([valid_rec]), [])
        bad_rec = dict(valid_rec, domain="UNKNOWN")
        errs = validate_manifest_records([bad_rec])
        self.assertTrue(any("invalid domain" in e for e in errs))


if __name__ == "__main__":
    unittest.main()
