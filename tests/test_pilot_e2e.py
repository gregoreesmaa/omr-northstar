"""End-to-end smoke tests for the OMR-NorthStar pilot fixture pack and baselines."""

from __future__ import annotations

import json
import pathlib
import unittest
import xml.etree.ElementTree as ET

from omr_northstar.cli import validate_manifest_records
from omr_northstar.pilot import (
    CANARY_STRING,
    load_pilot_manifest,
    parse_musicxml_summary,
    resolve_gt_path,
    run_degraded_baseline,
    run_oracle_baseline,
)
from omr_northstar.scoring import DOMAINS, TIERS

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
PILOT_DIR = REPO_ROOT / "pilot_dev_v0_1"
PAGE_SCHEMA_PATH = REPO_ROOT / "schemas" / "page_metadata.schema.json"
PERMISSIVE_LICENSES = {"Public-Domain", "CC0-1.0", "CC-BY-4.0"}


class TestPilotFixturesAndManifest(unittest.TestCase):
    """Validate pilot_dev_v0_1 MusicXML fixtures and manifest schema compliance."""

    def test_all_xml_fixtures_parse_cleanly(self) -> None:
        manifest = load_pilot_manifest(PILOT_DIR)
        pages = manifest["pages"]
        self.assertEqual(len(pages), 12)

        blank_controls = 0
        for rec in pages:
            gt_path = resolve_gt_path(PILOT_DIR, manifest, rec["page_id"])
            self.assertTrue(gt_path.is_file(), f"Missing ground-truth XML: {gt_path}")
            tree = ET.parse(gt_path)
            root = tree.getroot()
            self.assertEqual(root.tag, "score-partwise")
            summary = parse_musicxml_summary(gt_path)
            self.assertTrue(summary["parseable"])
            self.assertGreaterEqual(summary["m_gt"], 1)
            if summary["n_notes"] == 0:
                blank_controls += 1
                self.assertIn("negative_control_blank_staff", rec.get("notational_tags", []))

        self.assertGreaterEqual(blank_controls, 1, "Expected at least one negative-control blank-staff fixture.")

    def test_manifest_schema_domains_tiers_anchors_canary_and_licenses(self) -> None:
        manifest = load_pilot_manifest(PILOT_DIR)
        schema = json.loads(PAGE_SCHEMA_PATH.read_text(encoding="utf-8"))
        allowed_top_keys = set(schema["properties"].keys())
        required_top_keys = set(schema["required"])

        self.assertEqual(manifest.get("canary"), CANARY_STRING)
        pages = manifest["pages"]
        self.assertEqual(len(pages), 12)

        # CLI validator must return zero errors
        self.assertEqual(validate_manifest_records(pages), [])

        domains_seen: set[str] = set()
        tiers_seen: set[str] = set()
        anchors_by_id: dict[str, set[str]] = {}

        for rec in pages:
            rec_keys = set(rec.keys())
            self.assertTrue(
                required_top_keys.issubset(rec_keys),
                f"{rec['page_id']} missing required keys: {required_top_keys - rec_keys}",
            )
            self.assertTrue(
                rec_keys.issubset(allowed_top_keys),
                f"{rec['page_id']} has unknown keys disallowed by schema: {rec_keys - allowed_top_keys}",
            )

            domains_seen.add(rec["domain"])
            tiers_seen.add(rec["complexity_tier"])

            lic = rec["license"]
            self.assertIn(lic["music_license"], PERMISSIVE_LICENSES)
            self.assertIn(lic["image_license"], PERMISSIVE_LICENSES)

            anchor = rec.get("paired_anchor_id")
            if anchor:
                anchors_by_id.setdefault(anchor, set()).add(rec["domain"])

        self.assertEqual(domains_seen, set(DOMAINS))
        self.assertEqual(tiers_seen, set(TIERS))

        # Verify paired CDR<->RWP and CDR<->HWM anchors exist
        has_cdr_rwp = any({"CDR", "RWP"}.issubset(ds) for ds in anchors_by_id.values())
        has_cdr_hwm = any({"CDR", "HWM"}.issubset(ds) for ds in anchors_by_id.values())
        self.assertTrue(has_cdr_rwp, "Expected at least one paired CDR<->RWP anchor.")
        self.assertTrue(has_cdr_hwm, "Expected at least one paired CDR<->HWM anchor.")

    def test_musical_features_present_across_tiers(self) -> None:
        manifest = load_pilot_manifest(PILOT_DIR)
        xml_texts = {
            rec["page_id"]: resolve_gt_path(PILOT_DIR, manifest, rec["page_id"]).read_text(encoding="utf-8")
            for rec in manifest["pages"]
        }
        # T1 grand staff + chords + dynamics
        self.assertIn("<staves>2</staves>", xml_texts["ns-pilot-002"])
        self.assertIn("<chord/>", xml_texts["ns-pilot-002"])
        self.assertIn("<dynamics>", xml_texts["ns-pilot-002"])
        # T2 counterpoint + lyrics + alto clef
        self.assertIn("<backup>", xml_texts["ns-pilot-004"])
        self.assertIn("<lyric", xml_texts["ns-pilot-004"])
        self.assertIn("<sign>C</sign><line>3</line>", xml_texts["ns-pilot-004"])
        # T3 4-staff + transposing instrument + multi-measure rest
        self.assertIn("<transpose>", xml_texts["ns-pilot-005"])
        self.assertIn("<multiple-rest>2</multiple-rest>", xml_texts["ns-pilot-005"])
        # T4 asymmetric 7/8 + tuplet + mid-measure clef/key change
        self.assertIn("<beats>7</beats><beat-type>8</beat-type>", xml_texts["ns-pilot-009"])
        self.assertIn("<time-modification>", xml_texts["ns-pilot-009"])
        # T5 ostinato mutation + counterfactual accidentals
        self.assertIn("<accidental>sharp</accidental>", xml_texts["ns-pilot-011"])
        self.assertIn("<accidental>flat</accidental>", xml_texts["ns-pilot-011"])


class TestPilotBaselines(unittest.TestCase):
    """Run oracle and degraded baselines end-to-end and verify exact invariants."""

    def test_oracle_baseline_perfect_score(self) -> None:
        report = run_oracle_baseline(PILOT_DIR)
        l1 = report["layer1_headline"]
        l2 = report["layer2_matrix_and_deltas"]

        self.assertEqual(report["schema_version"], "1.0.0")
        self.assertAlmostEqual(l1["ns_score"], 100.0, places=4)
        self.assertAlmostEqual(l1["m_geom"], 100.0, places=4)
        self.assertAlmostEqual(l1["worst_stratum_q"], 100.0, places=4)
        self.assertAlmostEqual(l1["cfr_max"], 0.0, places=4)
        self.assertAlmostEqual(l1["human_utility"]["mean_ntsr"], 1.0, places=4)
        self.assertAlmostEqual(l1["human_utility"]["nupr"], 0.0, places=4)
        for d in DOMAINS:
            self.assertAlmostEqual(l1["stratum_scores"][d], 100.0, places=4)
            self.assertAlmostEqual(l1["stratum_cfr"][d], 0.0, places=4)
        self.assertAlmostEqual(l2["paired_deltas"]["delta_photo"], 0.0, places=4)
        self.assertAlmostEqual(l2["paired_deltas"]["delta_handwriting"], 0.0, places=4)

    def test_degraded_baseline_anti_gaming_and_paired_deltas(self) -> None:
        report = run_degraded_baseline(PILOT_DIR, seed=42)
        report_repeat = run_degraded_baseline(PILOT_DIR, seed=42)
        self.assertEqual(report, report_repeat, "Degraded baseline must be deterministic for fixed seed.")

        l1 = report["layer1_headline"]
        l2 = report["layer2_matrix_and_deltas"]
        strata = l1["stratum_scores"]
        deltas = l2["paired_deltas"]

        self.assertGreater(l1["ns_score"], 0.0)
        self.assertLess(l1["ns_score"], 70.0)
        self.assertGreater(strata["CDR"], strata["HWM"])
        self.assertGreater(strata["CDR"], strata["RWP"])
        self.assertGreater(l1["cfr_max"], 0.0)

        self.assertIsNotNone(deltas["delta_photo"])
        self.assertLess(deltas["delta_photo"], 0.0)
        self.assertIsNotNone(deltas["delta_handwriting"])
        self.assertLess(deltas["delta_handwriting"], 0.0)
        self.assertIsNotNone(deltas["delta_engraver"])
        self.assertGreater(deltas["delta_engraver"], 0.0)


if __name__ == "__main__":
    unittest.main()
