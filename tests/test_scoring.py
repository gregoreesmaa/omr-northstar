"""Unit tests for OMR-NorthStar reference scorer, CLI, and JSON Schemas."""

from __future__ import annotations

import json
import pathlib
import tempfile
import unittest

from omr_northstar.cli import format_scorecard, main as cli_main, validate_manifest_records
from omr_northstar.scoring import (
    DOMAINS,
    ERROR_BUCKET_KEYS,
    TIER_WEIGHTS,
    TIERS,
    PageMetrics,
    build_report,
    compute_catastrophic_penalties,
    compute_effective_measure_cost,
    compute_negative_control_metrics,
    compute_northstar_score,
    compute_ns_bootstrap_ci,
    compute_ntsr,
    compute_nupr,
    compute_page_quality,
    compute_paired_deltas,
    compute_prior_hallucination_rate,
    compute_stratum_scores,
    compute_tier_weighted_ns,
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
        self.assertGreaterEqual(len(schema_files), 2)
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


class TestBootstrapAndTierWeightedNS(unittest.TestCase):
    """Verify work-clustered bootstrap ns_ci95, 6-tier weighted NS, and edge cases."""

    def test_work_clustered_bootstrap_ci_deterministic_and_bounded(self) -> None:
        pages: list[PageMetrics] = []
        for d_idx, d in enumerate(DOMAINS):
            for w in range(4):
                s_val = 0.70 + 0.05 * w - 0.03 * d_idx
                pages.append(
                    PageMetrics(
                        page_id=f"{d}_w{w}_p1",
                        work_id=f"{d}_w{w}",
                        domain=d,
                        tier="T2",
                        s_note=s_val,
                        s_struct=s_val,
                        s_expr=s_val,
                        s_play=s_val,
                        wfpp=10.0,
                        transcribe_cost=100.0,
                    )
                )
        results = [compute_page_quality(p) for p in pages]
        ci1 = compute_ns_bootstrap_ci(results, n_boot=500, seed=123)
        ci2 = compute_ns_bootstrap_ci(results, n_boot=500, seed=123)
        self.assertIsNotNone(ci1)
        self.assertEqual(ci1, ci2)
        assert ci1 is not None
        self.assertLessEqual(ci1[0], ci1[1])

        report = build_report(pages, system_name="boot-test", n_boot=500, bootstrap_seed=123)
        ns_score = report["layer1_headline"]["ns_score"]
        ns_ci95 = report["layer1_headline"]["ns_ci95"]
        self.assertIsInstance(ns_ci95, list)
        self.assertEqual(len(ns_ci95), 2)
        self.assertLessEqual(ns_ci95[0], ns_score)
        self.assertGreaterEqual(ns_ci95[1], ns_score)

    def test_tier_weighted_ns_full_and_subset_renormalization(self) -> None:
        self.assertAlmostEqual(sum(TIER_WEIGHTS.values()), 1.0, places=9)
        # Subset with only T0 (Q=100) and T2 (Q=60) -> weights 0.05 and 0.25 -> renorm 1/6 and 5/6
        p_t0 = PageMetrics("p0", "w0", "CDR", "T0", 1.0, 1.0, 1.0, 1.0, 0.0, 100.0)
        p_t2 = PageMetrics("p2", "w2", "RWP", "T2", 0.6, 0.6, 0.6, 0.6, 40.0, 100.0)
        res = [compute_page_quality(p_t0), compute_page_quality(p_t2)]
        tw_ns = compute_tier_weighted_ns(res)
        w0 = 0.05 / 0.30
        w2 = 0.25 / 0.30
        m_geom = (100.0 ** w0) * (60.0 ** w2)
        expected = 0.75 * m_geom + 0.25 * 60.0
        self.assertAlmostEqual(tw_ns, expected, places=4)

        # Strict mode penalizes the 4 missing tiers to Q=0.0, CFR=1.0
        tw_ns_strict = compute_tier_weighted_ns(res, strict_missing_tiers=True)
        m_geom_strict = (100.0 ** 0.05) * (60.0 ** 0.25) * (0.1 ** 0.70)
        expected_strict = (0.75 * m_geom_strict + 0.25 * 0.0) * 0.50
        self.assertAlmostEqual(tw_ns_strict, expected_strict, places=4)
        self.assertLess(tw_ns_strict, tw_ns)

        report = build_report([p_t0, p_t2], system_name="tier-test")
        self.assertAlmostEqual(report["layer1_headline"]["tier_weighted_ns"], expected, places=4)

        # Full 6-tier pack: strict and non-strict produce exact identical Spec 4.3 value
        tier_qs = {"T0": 0.96, "T1": 0.90, "T2": 0.84, "T3": 0.78, "T4": 0.72, "T5": 0.64}
        full_pages = [
            PageMetrics(
                f"pf_{t}", f"wf_{t}", "CDR", t,
                q, q, q, q, (1.0 - q) * 100.0, 100.0,
            )
            for t, q in tier_qs.items()
        ]
        full_res = [compute_page_quality(p) for p in full_pages]
        tw_full_default = compute_tier_weighted_ns(full_res, strict_missing_tiers=False)
        tw_full_strict = compute_tier_weighted_ns(full_res, strict_missing_tiers=True)
        self.assertAlmostEqual(tw_full_default, tw_full_strict, places=9)
        m_full = 1.0
        for t in TIERS:
            m_full *= (tier_qs[t] * 100.0) ** TIER_WEIGHTS[t]
        expected_full = 0.75 * m_full + 0.25 * 64.0
        self.assertAlmostEqual(tw_full_strict, expected_full, places=4)

    def test_negative_control_non_music_page_safe(self) -> None:
        # Clean negative-control page (n_gt=0, n_pred=0, m_gt=0, m_pred=0, transcribe_cost=0)
        clean_blank = PageMetrics(
            page_id="neg_ok",
            work_id="w_neg",
            domain="ALS",
            tier="T0",
            s_note=1.0,
            s_struct=1.0,
            s_expr=1.0,
            s_play=1.0,
            wfpp=0.0,
            transcribe_cost=0.0,
            n_gt=0,
            n_pred=0,
            m_gt=0,
            m_pred=0,
            bsvr=1.0,
        )
        res_ok = compute_page_quality(clean_blank)
        self.assertFalse(res_ok.catastrophic_failure)
        self.assertNotIn("empty_output", res_ok.cf_reasons)
        self.assertAlmostEqual(res_ok.ntsr, 1.0)
        self.assertAlmostEqual(res_ok.q_score, 100.0)

        # Hallucinated music on a non-music negative-control page
        hallucinated_blank = PageMetrics(
            page_id="neg_fp",
            work_id="w_neg2",
            domain="ALS",
            tier="T0",
            s_note=0.0,
            s_struct=0.0,
            s_expr=0.0,
            s_play=0.0,
            wfpp=12.5,
            transcribe_cost=0.0,
            n_gt=0,
            n_pred=5,
            m_gt=0,
            m_pred=2,
            bsvr=1.0,
        )
        res_fp = compute_page_quality(hallucinated_blank)
        self.assertTrue(res_fp.catastrophic_failure)
        self.assertIn("hallucination_loop", res_fp.cf_reasons)
        self.assertIn("structural_collapse", res_fp.cf_reasons)
        self.assertEqual(res_fp.q_score, 0.0)
        self.assertLess(res_fp.ntsr, 0.0)

    def test_paired_deltas_groups_duplicate_engravers(self) -> None:
        pages = [
            PageMetrics("p1", "w1", "CDR", "T1", 0.90, 0.90, 0.90, 0.90, 10.0, 100.0, paired_anchor_id="a1", engraver="Dorico"),
            PageMetrics("p2", "w1", "CDR", "T1", 0.94, 0.94, 0.94, 0.94, 6.0, 100.0, paired_anchor_id="a1", engraver="Dorico"),
            PageMetrics("p3", "w1", "CDR", "T1", 0.82, 0.82, 0.82, 0.82, 18.0, 100.0, paired_anchor_id="a1", engraver="LilyPond"),
        ]
        deltas = compute_paired_deltas([compute_page_quality(p) for p in pages])
        # Dorico mean = 92.0, LilyPond = 82.0 -> spread = 10.0 (not max-min=12.0)
        self.assertAlmostEqual(deltas["delta_engraver"], 10.0, places=4)


class TestHardeningAndSpecCompliance(unittest.TestCase):
    """Verify NaN/Inf guards, NC-Gate separation, Layer 3 taxonomy, macro-averaging, and CF refinements."""

    def test_nan_and_inf_inputs_rejected(self) -> None:
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(ValueError):
                PageMetrics("p1", "w1", "CDR", "T1", bad, 0.9, 0.9, 0.9, 5.0, 100.0)
            with self.assertRaises(ValueError):
                PageMetrics("p1", "w1", "CDR", "T1", 0.9, 0.9, 0.9, 0.9, bad, 100.0)
            with self.assertRaises(ValueError):
                PageMetrics("p1", "w1", "CDR", "T1", 0.9, 0.9, 0.9, 0.9, 5.0, 100.0, bsvr=bad)
            with self.assertRaises(ValueError):
                compute_effective_measure_cost(bad, 10.0)
            with self.assertRaises(ValueError):
                compute_ntsr(bad, 100.0)
            with self.assertRaises(ValueError):
                compute_nupr([0.5, bad])
            with self.assertRaises(ValueError):
                compute_northstar_score(
                    {"CDR": bad, "ALS": 80.0, "RWP": 80.0, "HWM": 80.0},
                    {"CDR": 0.0, "ALS": 0.0, "RWP": 0.0, "HWM": 0.0},
                )
            with self.assertRaises(ValueError):
                compute_northstar_score(
                    {"CDR": 80.0, "ALS": 80.0, "RWP": 80.0, "HWM": 80.0},
                    {"CDR": bad, "ALS": 0.0, "RWP": 0.0, "HWM": 0.0},
                )

    def test_unit_interval_bounds_and_ns_clamping(self) -> None:
        for field_name in ("s_note", "s_struct", "s_expr", "s_play", "bsvr"):
            kwargs: dict[str, object] = dict(
                page_id="p1", work_id="w1", domain="CDR", tier="T1",
                s_note=0.9, s_struct=0.9, s_expr=0.9, s_play=0.9,
                wfpp=5.0, transcribe_cost=100.0, bsvr=1.0,
            )
            kwargs[field_name] = 1.25
            with self.assertRaises(ValueError):
                compute_page_quality(PageMetrics(**kwargs))  # type: ignore[arg-type]
            kwargs[field_name] = -0.1
            with self.assertRaises(ValueError):
                compute_page_quality(PageMetrics(**kwargs))  # type: ignore[arg-type]

        # Out-of-range stratum scores are clamped to [0.0, 100.0] before worst_q & m_geom
        res = compute_northstar_score(
            {"CDR": 150.0, "ALS": 90.0, "RWP": 80.0, "HWM": -25.0},
            {"CDR": 0.0, "ALS": 0.0, "RWP": 0.0, "HWM": 0.0},
        )
        self.assertEqual(res.worst_stratum_q, 0.0)
        self.assertLessEqual(res.ns_score, 100.0)
        self.assertGreaterEqual(res.ns_score, 0.0)

    def test_negative_control_separation_and_nc_gate(self) -> None:
        music_pages = [
            PageMetrics(
                f"m_{d}", f"w_{d}", d, "T2",
                0.80, 0.80, 0.80, 0.80, 20.0, 100.0,
                n_gt=50, n_pred=50, m_gt=8, m_pred=8,
            )
            for d in DOMAINS
        ]
        neg_clean = PageMetrics(
            "nc_ok", "w_nc1", "ALS", "T0",
            1.0, 1.0, 1.0, 1.0, 0.0, 0.0,
            n_gt=0, n_pred=0, m_gt=0, m_pred=0,
        )
        # Rest-only hallucination: n_pred=0 but m_pred=2 -> still fails NC-Gate
        neg_rest_fp = PageMetrics(
            "nc_rest", "w_nc2", "ALS", "T0",
            0.0, 0.0, 0.0, 0.0, 5.0, 0.0,
            n_gt=0, n_pred=0, m_gt=0, m_pred=2,
        )
        neg_note_fp = PageMetrics(
            "nc_note", "w_nc3", "RWP", "T0",
            0.0, 0.0, 0.0, 0.0, 10.0, 0.0,
            n_gt=0, n_pred=4, m_gt=0, m_pred=1,
        )

        # Standalone metric function
        nc = compute_negative_control_metrics([neg_clean, neg_rest_fp, neg_note_fp])
        self.assertEqual(nc["n_pages"], 3)
        self.assertEqual(nc["nc_fp"], 4)
        self.assertAlmostEqual(nc["nmhr"], 2.0 / 3.0, places=6)
        self.assertFalse(nc["nc_pass"])

        # Adding negative-control pages must NOT perturb NS, stratum_scores, axes_macro, or human_utility
        rep_music_only = build_report(music_pages, system_name="sys", n_boot=200, bootstrap_seed=7)
        rep_mixed = build_report(
            music_pages + [neg_clean, neg_rest_fp, neg_note_fp],
            system_name="sys",
            n_boot=200,
            bootstrap_seed=7,
        )
        self.assertEqual(rep_music_only["layer1_headline"], rep_mixed["layer1_headline"])
        self.assertEqual(rep_mixed["nc_gate"]["n_pages"], 3)
        self.assertEqual(rep_mixed["nc_gate"]["nc_fp"], 4)
        self.assertFalse(rep_mixed["nc_gate"]["nc_pass"])
        expected_ns_nc = round(rep_mixed["layer1_headline"]["ns_score"] * (1.0 - 0.25 * (2.0 / 3.0)), 4)
        self.assertAlmostEqual(rep_mixed["nc_gate"]["ns_nc"], expected_ns_nc, places=3)
        self.assertEqual(rep_mixed["layer2_matrix_and_deltas"]["adversarial_scorecard"]["non_music_fp_count"], 4)

        # Negative-control-only pack should not crash
        rep_neg_only = build_report([neg_clean], system_name="neg-only")
        self.assertTrue(rep_neg_only["nc_gate"]["nc_pass"])
        self.assertEqual(rep_neg_only["nc_gate"]["n_pages"], 1)
        self.assertIsNone(rep_neg_only["layer1_headline"]["ns_ci95"])

    def test_layer3_error_taxonomy_skips_zero_error_pages_and_sums_to_100(self) -> None:
        zero_buckets = {k: 0.0 for k in ERROR_BUCKET_KEYS}
        err_buckets_1 = {
            "layout_segmentation_pct": 10.0,
            "global_context_drift_pct": 10.0,
            "pitch_accidental_pct": 40.0,
            "rhythm_meter_pct": 20.0,
            "polyphony_routing_pct": 10.0,
            "engraving_expressive_pct": 10.0,
            "hallucination_truncation_pct": 0.0,
            "annotation_contamination_pct": 0.0,
        }
        err_buckets_2 = {
            "layout_segmentation_pct": 0.0,
            "global_context_drift_pct": 20.0,
            "pitch_accidental_pct": 20.0,
            "rhythm_meter_pct": 40.0,
            "polyphony_routing_pct": 0.0,
            "engraving_expressive_pct": 20.0,
            "hallucination_truncation_pct": 0.0,
            "annotation_contamination_pct": 0.0,
        }
        pages = [
            PageMetrics("p1", "w1", "CDR", "T1", 1.0, 1.0, 1.0, 1.0, 0.0, 100.0, error_buckets=zero_buckets),
            PageMetrics("p2", "w2", "ALS", "T1", 0.8, 0.8, 0.8, 0.8, 20.0, 100.0, error_buckets=err_buckets_1),
            PageMetrics("p3", "w3", "RWP", "T1", 0.8, 0.8, 0.8, 0.8, 20.0, 100.0, error_buckets=err_buckets_2),
            PageMetrics("p4", "w4", "HWM", "T1", 1.0, 1.0, 1.0, 1.0, 0.0, 100.0, error_buckets=zero_buckets),
        ]
        rep = build_report(pages, system_name="tax-test")
        l3 = rep["layer3_error_taxonomy"]
        self.assertAlmostEqual(sum(l3.values()), 100.0, places=4)
        self.assertAlmostEqual(l3["pitch_accidental_pct"], 30.0, places=4)
        self.assertAlmostEqual(l3["rhythm_meter_pct"], 30.0, places=4)

        # All zero-error pages -> all 0.0%
        rep_zero = build_report(
            [PageMetrics("z1", "wz1", "CDR", "T1", 1.0, 1.0, 1.0, 1.0, 0.0, 100.0, error_buckets=zero_buckets)],
            system_name="zero-err",
        )
        self.assertEqual(sum(rep_zero["layer3_error_taxonomy"].values()), 0.0)

    def test_multi_page_per_work_macro_averaging_and_cluster_bootstrap(self) -> None:
        # Work A has 3 pages with Q=100; Work B has 1 page with Q=60 -> work macro-avg = 80 (not page avg 90)
        pages = [
            PageMetrics(f"cdr_a_{i}", "work_a", "CDR", "T2", 1.0, 1.0, 1.0, 1.0, 0.0, 100.0)
            for i in range(3)
        ] + [
            PageMetrics("cdr_b_0", "work_b", "CDR", "T2", 0.6, 0.6, 0.6, 0.6, 40.0, 100.0),
            PageMetrics("als_0", "w_als", "ALS", "T2", 0.8, 0.8, 0.8, 0.8, 20.0, 100.0),
            PageMetrics("rwp_0", "w_rwp", "RWP", "T2", 0.8, 0.8, 0.8, 0.8, 20.0, 100.0),
            PageMetrics("hwm_0", "w_hwm", "HWM", "T2", 0.8, 0.8, 0.8, 0.8, 20.0, 100.0),
        ]
        results = [compute_page_quality(p) for p in pages]
        s_scores, _ = compute_stratum_scores(results)
        self.assertAlmostEqual(s_scores["CDR"], 80.0, places=4)

        ci = compute_ns_bootstrap_ci(results, n_boot=400, seed=99)
        self.assertIsNotNone(ci)
        assert ci is not None
        # Since CDR resamples {100, 60} at work level, CI spans [60..85]-ish and is strictly valid
        self.assertLessEqual(ci[0], ci[1])
        self.assertGreaterEqual(ci[0], 0.0)
        self.assertLessEqual(ci[1], 100.0)

    def test_catastrophic_penalty_refinements(self) -> None:
        # 1. Unparseable short-circuits to ("unparseable",) only, even with n_pred=0, m_pred=0, bsvr=0
        p_unparse = PageMetrics(
            "p_bad", "w1", "RWP", "T2",
            0.0, 0.0, 0.0, 0.0, 0.0, 100.0,
            parseable=False, n_gt=50, n_pred=0, m_gt=8, m_pred=0, bsvr=0.0,
        )
        is_cf, reasons, mult, eff_wfpp = compute_catastrophic_penalties(p_unparse)
        self.assertTrue(is_cf)
        self.assertEqual(reasons, ("unparseable",))
        self.assertEqual(mult, 0.0)
        self.assertAlmostEqual(eff_wfpp, 120.0)

        # 2. 1-bar difference on a 2-3 bar excerpt does NOT trigger structural_collapse
        p_short_ok = PageMetrics(
            "p_short", "w2", "CDR", "T0",
            0.9, 0.9, 0.9, 0.9, 5.0, 50.0,
            n_gt=12, n_pred=12, m_gt=3, m_pred=2, bsvr=1.0,
        )
        is_cf_ok, reasons_ok, _, _ = compute_catastrophic_penalties(p_short_ok)
        self.assertFalse(is_cf_ok)
        self.assertNotIn("structural_collapse", reasons_ok)

        p_short_bad = PageMetrics(
            "p_short_bad", "w2", "CDR", "T0",
            0.9, 0.9, 0.9, 0.9, 5.0, 50.0,
            n_gt=12, n_pred=12, m_gt=3, m_pred=5, bsvr=1.0,
        )
        is_cf_bad, reasons_bad, _, _ = compute_catastrophic_penalties(p_short_bad)
        self.assertTrue(is_cf_bad)
        self.assertIn("structural_collapse", reasons_bad)

        # 3. Tacet music page (m_gt > 0, n_gt == 0): 1 spurious note does NOT trigger hallucination_loop
        p_tacet = PageMetrics(
            "p_tacet", "w3", "CDR", "T1",
            0.0, 0.95, 0.95, 0.0, 2.0, 40.0,
            n_gt=0, n_pred=1, m_gt=16, m_pred=16, bsvr=1.0,
        )
        res_tacet = compute_page_quality(p_tacet)
        self.assertFalse(res_tacet.catastrophic_failure)
        self.assertNotIn("hallucination_loop", res_tacet.cf_reasons)
        self.assertGreater(res_tacet.q_score, 0.0)

        # Tacet page with explicit repetition loop DOES trigger hallucination_loop and zero Q
        p_tacet_loop = PageMetrics(
            "p_tacet_loop", "w3", "CDR", "T1",
            0.0, 0.95, 0.95, 0.0, 2.0, 40.0,
            n_gt=0, n_pred=10, m_gt=16, m_pred=16, bsvr=1.0, has_repetition_loop=True,
        )
        res_tacet_loop = compute_page_quality(p_tacet_loop)
        self.assertTrue(res_tacet_loop.catastrophic_failure)
        self.assertIn("hallucination_loop", res_tacet_loop.cf_reasons)
        self.assertEqual(res_tacet_loop.q_score, 0.0)


if __name__ == "__main__":
    unittest.main()
