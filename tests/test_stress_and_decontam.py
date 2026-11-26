"""Unit tests for OMR-NorthStar adversarial stress-test scorers, parametric
MusicXML trap generators, and blind symbolic 8-gram scrubbing index.
"""

from __future__ import annotations

import json
import unittest
import xml.etree.ElementTree as ET

from omr_northstar.decontam import (
    build_scrubbing_index,
    check_corpus_overlap,
    compute_symbolic_8gram_hashes,
)
from omr_northstar.stress import (
    generate_counterfactual_musicxml,
    generate_ostinato_trap_musicxml,
    score_annotation_invariance,
    score_counterfactual_trap,
    score_multipage_continuity,
    score_negative_control,
    score_orientation_invariance,
    score_ostinato_trap,
    score_positional_decay,
)


def _make_melody_musicxml(
    notes: list[tuple[str, int, int, int]],
    *,
    divisions: int = 4,
    part_id: str = "P1",
) -> str:
    """Build a minimal valid MusicXML score from (step, alter, octave, duration) tuples."""
    score = ET.Element("score-partwise", {"version": "4.0"})
    part_list = ET.SubElement(score, "part-list")
    sp = ET.SubElement(part_list, "score-part", {"id": part_id})
    ET.SubElement(sp, "part-name").text = "Test"
    part = ET.SubElement(score, "part", {"id": part_id})

    measure = ET.SubElement(part, "measure", {"number": "1"})
    attrs = ET.SubElement(measure, "attributes")
    ET.SubElement(attrs, "divisions").text = str(divisions)

    for step, alter, octave, dur in notes:
        n_el = ET.SubElement(measure, "note")
        p_el = ET.SubElement(n_el, "pitch")
        ET.SubElement(p_el, "step").text = step
        if alter != 0:
            ET.SubElement(p_el, "alter").text = str(alter)
        ET.SubElement(p_el, "octave").text = str(octave)
        ET.SubElement(n_el, "duration").text = str(dur)
        ET.SubElement(n_el, "voice").text = "1"
    return ET.tostring(score, encoding="unicode", xml_declaration=True)


def _transpose_notes(
    notes: list[tuple[str, int, int, int]],
    semitones: int,
    *,
    dur_scale: int = 1,
) -> list[tuple[str, int, int, int]]:
    """Transpose (step, alter, octave, dur) by exact semitones and scale durations."""
    step_to_pc = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
    pc_to_step_alter = {
        0: ("C", 0),
        1: ("C", 1),
        2: ("D", 0),
        3: ("E", -1),
        4: ("E", 0),
        5: ("F", 0),
        6: ("F", 1),
        7: ("G", 0),
        8: ("A", -1),
        9: ("A", 0),
        10: ("B", -1),
        11: ("B", 0),
    }
    out: list[tuple[str, int, int, int]] = []
    for step, alter, octave, dur in notes:
        midi = (octave + 1) * 12 + step_to_pc[step] + alter + semitones
        new_oct = (midi // 12) - 1
        new_step, new_alter = pc_to_step_alter[midi % 12]
        out.append((new_step, new_alter, new_oct, dur * dur_scale))
    return out


class TestStressScorers(unittest.TestCase):
    """Test all 7 executable stress-test scorers (passing & failing edge cases)."""

    def test_ostinato_trap_pass_and_fail(self) -> None:
        # Passing with sequence
        res = score_ostinato_trap(
            ["F#4", "Bb4", "C5"], ["F#4", "Bb4", "C5"], 16, 16, False
        )
        self.assertTrue(res.passed)
        self.assertAlmostEqual(res.tsr, 1.0)
        self.assertTrue(res.exact_bar_count)
        self.assertEqual(res.copy_loop_rate, 0.0)
        self.assertEqual(res.gate_failures, ())

        # Passing with mapping
        res_map = score_ostinato_trap(
            {(4, 2): "F#4", (11, 5): "split"},
            {(4, 2): "F#4", (11, 5): "split"},
            16,
            16,
            False,
        )
        self.assertTrue(res_map.passed)
        self.assertAlmostEqual(res_map.tsr, 1.0)

        # Failing: low TSR, wrong bar count, copy loop triggered
        fail = score_ostinato_trap(["F4", "Bb4", "C5"], ["F#4", "Bb4", "C5"], 15, 16, True)
        self.assertFalse(fail.passed)
        self.assertAlmostEqual(fail.tsr, 2.0 / 3.0)
        self.assertFalse(fail.exact_bar_count)
        self.assertEqual(fail.copy_loop_rate, 1.0)
        self.assertEqual(len(fail.gate_failures), 3)

    def test_counterfactual_trap_pass_and_fail(self) -> None:
        orig = ["C4"] * 50
        mut = ["C#4"] * 50

        # Exact boundary pass: 1/50 = 2% PHR, 49/50 = 98% F1
        pred_boundary = ["C4"] + ["C#4"] * 49
        res = score_counterfactual_trap(pred_boundary, orig, mut)
        self.assertTrue(res.passed)
        self.assertAlmostEqual(res.phr, 0.02)
        self.assertAlmostEqual(res.mutation_site_f1, 0.98)

        # Fail PHR (> 0.02): 2/50 = 4% reverted to canonical original
        pred_prior = ["C4", "C4"] + ["C#4"] * 48
        fail_phr = score_counterfactual_trap(pred_prior, orig, mut)
        self.assertFalse(fail_phr.passed)
        self.assertAlmostEqual(fail_phr.phr, 0.04)
        self.assertAlmostEqual(fail_phr.mutation_site_f1, 0.96)

        # Fail F1 (< 0.95) even with 0% PHR (non-canonical wrong predictions)
        pred_wrong = ["D4"] * 3 + ["C#4"] * 47
        fail_f1 = score_counterfactual_trap(pred_wrong, orig, mut)
        self.assertFalse(fail_f1.passed)
        self.assertAlmostEqual(fail_f1.phr, 0.0)
        self.assertAlmostEqual(fail_f1.mutation_site_f1, 0.94)

        # Length mismatch raises ValueError
        with self.assertRaises(ValueError):
            score_counterfactual_trap(["C#4"], ["C4", "D4"], ["C#4", "D#4"])

    def test_positional_decay_pass_and_fail(self) -> None:
        ok = score_positional_decay(0.96, 0.92, pred_systems=12, gt_systems=12)
        self.assertTrue(ok.passed)
        self.assertAlmostEqual(ok.q4_q1_ratio, 0.92 / 0.96)
        self.assertAlmostEqual(ok.system_completeness, 1.0)

        # Fails Q4/Q1 ratio (< 0.95)
        fail_decay = score_positional_decay(0.96, 0.85, pred_systems=12, gt_systems=12)
        self.assertFalse(fail_decay.passed)
        self.assertLess(fail_decay.q4_q1_ratio, 0.95)

        # Fails dropped system
        fail_drop = score_positional_decay(0.96, 0.95, pred_systems=11, gt_systems=12)
        self.assertFalse(fail_drop.passed)
        self.assertAlmostEqual(fail_drop.system_completeness, 11.0 / 12.0)

        # Fails hallucinated extra system
        fail_extra = score_positional_decay(0.96, 0.95, pred_systems=13, gt_systems=12)
        self.assertFalse(fail_extra.passed)
        self.assertLess(fail_extra.system_completeness, 1.0)

    def test_negative_control_pass_and_fail(self) -> None:
        ok = score_negative_control(0, "blank_staff_paper")
        self.assertTrue(ok.passed)
        self.assertEqual(ok.pred_note_count, 0)

        fail = score_negative_control(3, "title_page")
        self.assertFalse(fail.passed)
        self.assertEqual(fail.pred_note_count, 3)

        with self.assertRaises(ValueError):
            score_negative_control(-1, "text_page")

    def test_orientation_invariance_pass_and_fail(self) -> None:
        # Exact 0.01 drop boundary + one angle slightly higher than 0°
        ok = score_orientation_invariance(0.95, 0.94, 0.955, 0.945)
        self.assertTrue(ok.passed)
        self.assertAlmostEqual(ok.max_drop, 0.01, places=6)

        fail = score_orientation_invariance(0.95, 0.94, 0.91, 0.945)
        self.assertFalse(fail.passed)
        self.assertAlmostEqual(fail.max_drop, 0.04, places=6)

    def test_multipage_continuity_pass_and_fail(self) -> None:
        ok = score_multipage_continuity(
            matched_spanners=19,
            pred_spanners=20,
            gt_spanners=20,
            correct_staff_ids=98,
            total_staff_ids=100,
        )
        self.assertTrue(ok.passed)
        self.assertAlmostEqual(ok.spanner_f1, 0.95)
        self.assertAlmostEqual(ok.staff_tracking_acc, 0.98)

        fail_span = score_multipage_continuity(18, 20, 20, 100, 100)
        self.assertFalse(fail_span.passed)
        self.assertAlmostEqual(fail_span.spanner_f1, 0.90)

        fail_staff = score_multipage_continuity(20, 20, 20, 97, 100)
        self.assertFalse(fail_staff.passed)
        self.assertAlmostEqual(fail_staff.staff_tracking_acc, 0.97)

    def test_annotation_invariance_pass_and_fail(self) -> None:
        ok = score_annotation_invariance(0.95, 0.92, 0.005)
        self.assertTrue(ok.passed)
        self.assertAlmostEqual(ok.delta_f1, 0.03, places=6)

        fail_delta = score_annotation_invariance(0.95, 0.90, 0.001)
        self.assertFalse(fail_delta.passed)
        self.assertAlmostEqual(fail_delta.delta_f1, 0.05, places=6)

        fail_fp = score_annotation_invariance(0.95, 0.94, 0.01)
        self.assertFalse(fail_fp.passed)


class TestParametricTrapGenerators(unittest.TestCase):
    """Test both Parametric MusicXML Trap Generators for validity & determinism."""

    def test_ostinato_generator_valid_xml_and_determinism(self) -> None:
        xml1, meta1 = generate_ostinato_trap_musicxml(n_bars=16, seed=123)
        xml2, meta2 = generate_ostinato_trap_musicxml(n_bars=16, seed=123)
        xml3, meta3 = generate_ostinato_trap_musicxml(n_bars=16, seed=456)

        # Deterministic for identical seed, distinct for different seed
        self.assertEqual(xml1, xml2)
        self.assertEqual(meta1, meta2)
        self.assertNotEqual(xml1, xml3)

        # Valid MusicXML with exact bar count and valid 4/4 duration sums
        root = ET.fromstring(xml1)
        self.assertEqual(root.tag, "score-partwise")
        measures = root.findall("./part/measure")
        self.assertEqual(len(measures), 16)

        for m in measures:
            total_dur = sum(
                int(n.findtext("duration", "0")) for n in m.findall("note")
            )
            self.assertEqual(total_dur, 16, f"Measure {m.attrib['number']} duration != 16")

        # 2-3 exact mutations, none in bar 1, containing both mutation types
        mutations = meta1["mutations"]
        self.assertIn(len(mutations), (2, 3))
        mtypes = {m["mutation_type"] for m in mutations}
        self.assertEqual(mtypes, {"accidental_flip", "rhythm_split"})
        for m in mutations:
            self.assertGreaterEqual(m["measure"], 2)
            self.assertNotEqual(m["original"], m["mutated"])

        with self.assertRaises(ValueError):
            generate_ostinato_trap_musicxml(n_bars=2, seed=1)

    def test_counterfactual_generator_valid_xml_and_phr_roundtrip(self) -> None:
        base_xml, _ = generate_ostinato_trap_musicxml(n_bars=16, seed=7)
        mut_xml1, sites1 = generate_counterfactual_musicxml(
            base_xml, mutation_rate=0.06, seed=99
        )
        mut_xml2, sites2 = generate_counterfactual_musicxml(
            base_xml, mutation_rate=0.06, seed=99
        )
        mut_xml3, sites3 = generate_counterfactual_musicxml(
            base_xml, mutation_rate=0.06, seed=100
        )

        self.assertEqual(mut_xml1, mut_xml2)
        self.assertEqual(sites1, sites2)
        self.assertNotEqual(sites1, sites3)

        # Valid XML
        root_base = ET.fromstring(base_xml)
        root_mut = ET.fromstring(mut_xml1)
        n_pitched = sum(
            1 for n in root_base.findall(".//note") if n.find("pitch") is not None
        )
        expected_k = round(n_pitched * 0.06)
        self.assertEqual(len(sites1), expected_k)
        self.assertEqual(
            len(root_mut.findall(".//note")), len(root_base.findall(".//note"))
        )

        for s in sites1:
            self.assertNotEqual(s["original"], s["mutated"])
            self.assertIn(s["mutation_type"], ("accidental", "octave"))

        # End-to-end integration with score_counterfactual_trap
        orig_seq = [s["original"] for s in sites1]
        mut_seq = [s["mutated"] for s in sites1]

        # Prior-hallucinating model predicts original notes -> PHR = 1.0, F1 = 0.0
        bad_res = score_counterfactual_trap(orig_seq, orig_seq, mut_seq)
        self.assertFalse(bad_res.passed)
        self.assertAlmostEqual(bad_res.phr, 1.0)
        self.assertAlmostEqual(bad_res.mutation_site_f1, 0.0)

        # Visually faithful model predicts mutated notes -> PHR = 0.0, F1 = 1.0
        good_res = score_counterfactual_trap(mut_seq, orig_seq, mut_seq)
        self.assertTrue(good_res.passed)
        self.assertAlmostEqual(good_res.phr, 0.0)
        self.assertAlmostEqual(good_res.mutation_site_f1, 1.0)


class TestBlindScrubbingIndex(unittest.TestCase):
    """Test transposition/tempo-invariant 8-gram hashing & zero-leakage overlap checks."""

    def setUp(self) -> None:
        # 12-note distinctive melody (produces 12 - 8 + 1 = 5 symbolic 8-grams)
        self.melody_a = [
            ("C", 0, 4, 4),
            ("E", 0, 4, 2),
            ("G", 0, 4, 2),
            ("F", 1, 4, 4),
            ("A", 0, 4, 6),
            ("G", 0, 4, 2),
            ("E", -1, 4, 4),
            ("D", 0, 4, 2),
            ("B", 0, 3, 2),
            ("C", 0, 4, 4),
            ("D", 0, 4, 4),
            ("G", 0, 4, 8),
        ]
        # Completely different contour & rhythm
        self.melody_b = [
            ("A", 0, 4, 2),
            ("G", 1, 4, 2),
            ("A", 0, 4, 2),
            ("B", 0, 4, 2),
            ("C", 0, 5, 4),
            ("B", 0, 4, 4),
            ("A", 0, 4, 2),
            ("F", 0, 4, 6),
            ("E", 0, 4, 2),
            ("D", 0, 4, 2),
            ("C", 0, 4, 4),
            ("A", 0, 3, 4),
        ]

    def test_transposition_and_tempo_invariance(self) -> None:
        xml_orig = _make_melody_musicxml(self.melody_a, divisions=4)
        # Transpose up by a perfect fifth (+7 semitones) AND double all durations
        # with a different divisions=12 -> identical pitch intervals & IOI ratios!
        melody_a_transposed = _transpose_notes(self.melody_a, semitones=7, dur_scale=3)
        xml_transposed = _make_melody_musicxml(melody_a_transposed, divisions=12)

        h_orig = compute_symbolic_8gram_hashes(xml_orig, n=8)
        h_trans = compute_symbolic_8gram_hashes(xml_transposed, n=8)

        self.assertEqual(len(h_orig), 5)
        self.assertEqual(h_orig, h_trans)
        # SHA-256 hex digests are 64 chars
        self.assertTrue(all(len(h) == 64 for h in h_orig))

    def test_scrubbing_index_detects_transposed_leak_with_zero_fp(self) -> None:
        xml_test_a = _make_melody_musicxml(self.melody_a, divisions=4)
        xml_cand_leaked = _make_melody_musicxml(
            _transpose_notes(self.melody_a, semitones=-5, dur_scale=2), divisions=8
        )
        xml_cand_clean = _make_melody_musicxml(self.melody_b, divisions=4)

        index = build_scrubbing_index({"test_work_001": xml_test_a})

        # Privacy check: serialized index must not leak raw pitch/XML tags
        serialized = json.dumps(index)
        self.assertNotIn("<pitch>", serialized)
        self.assertNotIn("<score-partwise", serialized)

        overlaps = check_corpus_overlap(
            {
                "train_leaked_transposed": xml_cand_leaked,
                "train_clean_distinct": xml_cand_clean,
            },
            index,
            min_shared_8grams=2,
        )

        self.assertIn("train_leaked_transposed", overlaps)
        self.assertEqual(overlaps["train_leaked_transposed"], {"test_work_001": 5})
        self.assertNotIn("train_clean_distinct", overlaps)

    def test_min_shared_threshold_and_validation(self) -> None:
        xml_test = _make_melody_musicxml(self.melody_a, divisions=4)
        index = build_scrubbing_index({"w1": xml_test})

        # Requiring more shared 8-grams than exist yields no match
        no_match = check_corpus_overlap({"c1": xml_test}, index, min_shared_8grams=10)
        self.assertEqual(no_match, {})

        with self.assertRaises(ValueError):
            compute_symbolic_8gram_hashes(xml_test, n=2)
        with self.assertRaises(ValueError):
            check_corpus_overlap({"c1": xml_test}, index, min_shared_8grams=0)


if __name__ == "__main__":
    unittest.main()
