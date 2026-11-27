"""Unit tests for OMR-NorthStar adversarial stress-test scorers, parametric
MusicXML trap generators, and blind symbolic 8-gram scrubbing index.
"""

from __future__ import annotations

import json
import math
import unittest
import xml.etree.ElementTree as ET

from omr_northstar.cer import parse_musicxml
from omr_northstar.decontam import (
    _extract_voice_note_streams,
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

        # Missing key in pred mapping must not falsely match None GT value
        res_missing = score_ostinato_trap(
            {(2, 0): "F#4"},
            {(1, 0): None, (2, 0): "F#4"},
            16,
            16,
            False,
        )
        self.assertFalse(res_missing.passed)
        self.assertAlmostEqual(res_missing.tsr, 0.5)

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

        # None predictions must never inflate mutation_site_f1 above 1.0
        res_none = score_counterfactual_trap(
            [None, "C#4"],
            ["C4", "C4"],
            [None, "C#4"],
        )
        self.assertEqual(res_none.matched_count, 1)
        self.assertLessEqual(res_none.mutation_site_f1, 1.0)
        self.assertAlmostEqual(res_none.mutation_site_f1, 2.0 / 3.0)

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

        # Invalid F1 values (NaN, Inf, < 0, > 1) must raise ValueError
        for bad_q1, bad_q4 in (
            (math.nan, 0.9),
            (0.9, math.nan),
            (math.inf, 0.9),
            (0.9, -math.inf),
            (-0.1, 0.9),
            (0.9, 1.05),
            (1.2, 0.9),
        ):
            with self.assertRaises(ValueError):
                score_positional_decay(bad_q1, bad_q4, pred_systems=4, gt_systems=4)

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

        # Round-trip both ostinato & counterfactual XML through cer.parse_musicxml
        cer_base = parse_musicxml(base_xml)
        cer_mut = parse_musicxml(mut_xml1)
        self.assertEqual(len(cer_base.measures), 16)
        self.assertEqual(len(cer_mut.measures), 16)
        self.assertEqual(
            sum(len(m.notes) for m in cer_base.measures),
            sum(len(m.notes) for m in cer_mut.measures),
        )

    def test_ostinato_accidental_state_and_cer_roundtrip(self) -> None:
        alter_to_acc = {-2: "flat-flat", -1: "flat", 0: "natural", 1: "sharp", 2: "double-sharp"}
        saw_cancellation = False

        for seed in range(30):
            xml_str, meta = generate_ostinato_trap_musicxml(n_bars=8, seed=seed)
            root = ET.fromstring(xml_str)
            for measure in root.findall("./part/measure"):
                active_alter: dict[tuple[str, int], int] = {}
                for note in measure.findall("note"):
                    pitch = note.find("pitch")
                    assert pitch is not None
                    step = pitch.findtext("step", "").strip()
                    alter = int(pitch.findtext("alter", "0").strip())
                    octave = int(pitch.findtext("octave", "4").strip())
                    prev = active_alter.get((step, octave), 0)
                    acc_el = note.find("accidental")
                    if alter != prev:
                        self.assertIsNotNone(
                            acc_el,
                            f"seed={seed} measure={measure.attrib['number']} missing accidental for {step}{alter}/{octave}",
                        )
                        assert acc_el is not None
                        self.assertEqual(acc_el.text, alter_to_acc[alter])
                        if prev != 0 and alter == 0:
                            saw_cancellation = True
                    else:
                        self.assertIsNone(
                            acc_el,
                            f"seed={seed} measure={measure.attrib['number']} redundant accidental for {step}{alter}/{octave}",
                        )
                    active_alter[(step, octave)] = alter

            # Round-trip through CER parser
            cer_score = parse_musicxml(xml_str)
            self.assertEqual(len(cer_score.measures), 8)
            self.assertEqual(meta["n_bars"], 8)

        self.assertTrue(
            saw_cancellation,
            "Expected at least one natural cancellation after an earlier accidental flip in the same bar.",
        )

    def test_counterfactual_xsd_order_float_alter_and_namespace(self) -> None:
        # 1. XSD order check + CER round-trip: <accidental> must appear after
        # <type>/<dot> and before <stem>/<staff>/<beam>/<notations>/<lyric>.
        xml_with_post_tags = """<?xml version='1.0' encoding='utf-8'?>
<score-partwise version="4.0">
  <part-list><score-part id="P1"><part-name>Test</part-name></score-part></part-list>
  <part id="P1">
    <measure number="1">
      <attributes><divisions>4</divisions></attributes>
      <note>
        <pitch><step>C</step><alter>1</alter><octave>4</octave></pitch>
        <duration>6</duration>
        <voice>1</voice>
        <type>quarter</type>
        <dot/>
        <stem>up</stem>
        <staff>1</staff>
        <beam number="1">begin</beam>
        <notations><articulations><staccato/></articulations></notations>
        <lyric number="1"><text>la</text></lyric>
      </note>
      <note>
        <pitch><step>D</step><octave>4</octave></pitch>
        <duration>2</duration>
        <voice>1</voice>
        <type>eighth</type>
        <stem>up</stem>
      </note>
      <note>
        <pitch><step>E</step><octave>4</octave></pitch>
        <duration>4</duration>
        <voice>1</voice>
        <type>quarter</type>
        <staff>1</staff>
      </note>
    </measure>
  </part>
</score-partwise>"""

        post_tags = {
            "time-modification",
            "stem",
            "notehead",
            "notehead-text",
            "staff",
            "beam",
            "notations",
            "lyric",
            "play",
            "listen",
        }
        saw_accidental_mutation = False
        for seed in range(20):
            mut_xml, sites = generate_counterfactual_musicxml(
                xml_with_post_tags, mutation_rate=1.0, seed=seed
            )
            self.assertEqual(len(sites), 3)
            root = ET.fromstring(mut_xml)
            for note in root.findall(".//note"):
                tags = [child.tag for child in note]
                if "accidental" in tags:
                    saw_accidental_mutation = True
                    acc_idx = tags.index("accidental")
                    for pre in ("type", "dot"):
                        if pre in tags:
                            self.assertGreater(acc_idx, tags.index(pre))
                    for post in post_tags:
                        if post in tags:
                            self.assertLess(acc_idx, tags.index(post))
            # Must round-trip cleanly through CER
            cer_score = parse_musicxml(mut_xml)
            self.assertEqual(len(cer_score.measures[0].notes), 3)
        self.assertTrue(saw_accidental_mutation)

        # 2. Float <alter> (1.0, 0.5) and whitespace-only <alter>
        xml_float_alter = """<?xml version='1.0' encoding='utf-8'?>
<score-partwise version="4.0">
  <part-list><score-part id="P1"><part-name>FloatAlter</part-name></score-part></part-list>
  <part id="P1">
    <measure number="1">
      <attributes><divisions>4</divisions></attributes>
      <note>
        <pitch><step>C</step><alter>1.0</alter><octave>4</octave></pitch>
        <duration>4</duration><voice>1</voice><type>quarter</type><stem>up</stem>
      </note>
      <note>
        <pitch><step>D</step><alter>0.5</alter><octave>4</octave></pitch>
        <duration>4</duration><voice>1</voice><type>quarter</type><stem>up</stem>
      </note>
      <note>
        <pitch><step>E</step><alter>   </alter><octave>4</octave></pitch>
        <duration>4</duration><voice>1</voice><type>quarter</type><staff>1</staff>
      </note>
    </measure>
  </part>
</score-partwise>"""
        mut_float_xml, float_sites = generate_counterfactual_musicxml(
            xml_float_alter, mutation_rate=1.0, seed=7
        )
        self.assertEqual(len(float_sites), 3)
        self.assertEqual(float_sites[0]["original"], "C#4")
        self.assertEqual(float_sites[1]["original"], "D4")
        self.assertEqual(float_sites[2]["original"], "E4")
        ET.fromstring(mut_float_xml)

        # 3. Namespace-agnostic traversal
        ns = "http://www.musicxml.org/ns/musicxml"
        ns_xml = f"""<?xml version='1.0' encoding='utf-8'?>
<score-partwise xmlns="{ns}" version="4.0">
  <part-list><score-part id="P1"><part-name>NS</part-name></score-part></part-list>
  <part id="P1">
    <measure number="1">
      <attributes><divisions>4</divisions></attributes>
      <note>
        <pitch><step>G</step><octave>4</octave></pitch>
        <duration>4</duration>
        <voice>1</voice>
        <type>quarter</type>
        <stem>up</stem>
      </note>
    </measure>
  </part>
</score-partwise>"""
        mut_ns_xml, ns_sites = generate_counterfactual_musicxml(
            ns_xml, mutation_rate=1.0, seed=5
        )
        self.assertEqual(len(ns_sites), 1)
        cer_ns = parse_musicxml(mut_ns_xml)
        self.assertEqual(len(cer_ns.measures[0].notes), 1)


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

    def test_multivoice_backup_no_cross_measure_drift(self) -> None:
        # Measure 1 has voice 1 (4 quarters = 4 beats), <backup> of 4 beats,
        # and a shorter voice 2 (only 2 quarters). Measure 2 continues voice 1.
        # Measure 2 must start at beat 4 (not beat 2), so voice 1's 8-gram
        # matches a single-voice reference score with the same 8 notes.
        v1_notes = [
            ("C", 0, 4, 4),
            ("D", 0, 4, 4),
            ("E", 0, 4, 4),
            ("G", 0, 4, 4),
            ("A", 0, 4, 4),
            ("G", 0, 4, 4),
            ("F", 0, 4, 4),
            ("E", 0, 4, 4),
        ]
        ref_xml = _make_melody_musicxml(v1_notes, divisions=4)

        score = ET.Element("score-partwise", {"version": "4.0"})
        part_list = ET.SubElement(score, "part-list")
        sp = ET.SubElement(part_list, "score-part", {"id": "P1"})
        ET.SubElement(sp, "part-name").text = "MultiVoice"
        part = ET.SubElement(score, "part", {"id": "P1"})

        m1 = ET.SubElement(part, "measure", {"number": "1"})
        attrs = ET.SubElement(m1, "attributes")
        ET.SubElement(attrs, "divisions").text = "4"
        for step, alter, octv, dur in v1_notes[:4]:
            n_el = ET.SubElement(m1, "note")
            p_el = ET.SubElement(n_el, "pitch")
            ET.SubElement(p_el, "step").text = step
            ET.SubElement(p_el, "octave").text = str(octv)
            ET.SubElement(n_el, "duration").text = str(dur)
            ET.SubElement(n_el, "voice").text = "1"
        # Overshooting backup (duration=24 > measure length 16) clamps to measure_start
        bkp = ET.SubElement(m1, "backup")
        ET.SubElement(bkp, "duration").text = "24"
        # Shorter voice 2 (only 2 quarter notes = 8 ticks)
        for step in ("C", "G"):
            n_el = ET.SubElement(m1, "note")
            p_el = ET.SubElement(n_el, "pitch")
            ET.SubElement(p_el, "step").text = step
            ET.SubElement(p_el, "octave").text = "3"
            ET.SubElement(n_el, "duration").text = "4"
            ET.SubElement(n_el, "voice").text = "2"

        m2 = ET.SubElement(part, "measure", {"number": "2"})
        for step, alter, octv, dur in v1_notes[4:]:
            n_el = ET.SubElement(m2, "note")
            p_el = ET.SubElement(n_el, "pitch")
            ET.SubElement(p_el, "step").text = step
            ET.SubElement(p_el, "octave").text = str(octv)
            ET.SubElement(n_el, "duration").text = str(dur)
            ET.SubElement(n_el, "voice").text = "1"

        mv_xml = ET.tostring(score, encoding="unicode", xml_declaration=True)
        self.assertEqual(
            compute_symbolic_8gram_hashes(mv_xml, n=8),
            compute_symbolic_8gram_hashes(ref_xml, n=8),
        )

    def test_chordal_skyline_collapse_eliminates_zero_iois(self) -> None:
        # 8 chords where the skyline (top note) matches self.melody_a[:8]
        skyline = self.melody_a[:8]
        skyline_xml = _make_melody_musicxml(skyline, divisions=4)

        score = ET.Element("score-partwise", {"version": "4.0"})
        part_list = ET.SubElement(score, "part-list")
        sp = ET.SubElement(part_list, "score-part", {"id": "P1"})
        ET.SubElement(sp, "part-name").text = "Chords"
        part = ET.SubElement(score, "part", {"id": "P1"})
        m1 = ET.SubElement(part, "measure", {"number": "1"})
        attrs = ET.SubElement(m1, "attributes")
        ET.SubElement(attrs, "divisions").text = "4"

        for step, alter, octv, dur in skyline:
            # Lower chord tone first
            low = ET.SubElement(m1, "note")
            p_low = ET.SubElement(low, "pitch")
            ET.SubElement(p_low, "step").text = step
            if alter != 0:
                ET.SubElement(p_low, "alter").text = str(alter)
            ET.SubElement(p_low, "octave").text = str(octv - 1)
            ET.SubElement(low, "duration").text = str(dur)
            ET.SubElement(low, "voice").text = "1"

            # Higher skyline tone as <chord/>
            high = ET.SubElement(m1, "note")
            ET.SubElement(high, "chord")
            p_high = ET.SubElement(high, "pitch")
            ET.SubElement(p_high, "step").text = step
            if alter != 0:
                ET.SubElement(p_high, "alter").text = str(alter)
            ET.SubElement(p_high, "octave").text = str(octv)
            ET.SubElement(high, "duration").text = str(dur)
            ET.SubElement(high, "voice").text = "1"

        chord_xml = ET.tostring(score, encoding="unicode", xml_declaration=True)
        streams = _extract_voice_note_streams(chord_xml)
        self.assertEqual(len(streams), 1)
        self.assertEqual(len(streams[0]), 8)
        for i in range(len(streams[0]) - 1):
            self.assertGreater(streams[0][i + 1][0], streams[0][i][0])

        self.assertEqual(
            compute_symbolic_8gram_hashes(chord_xml, n=8),
            compute_symbolic_8gram_hashes(skyline_xml, n=8),
        )

    def test_grand_staff_separates_shared_voice_across_staves(self) -> None:
        # Both staff 1 (RH) and staff 2 (LH) use <voice>1</voice>
        rh = self.melody_a[:8]
        lh = self.melody_b[:8]
        score = ET.Element("score-partwise", {"version": "4.0"})
        part_list = ET.SubElement(score, "part-list")
        sp = ET.SubElement(part_list, "score-part", {"id": "P1"})
        ET.SubElement(sp, "part-name").text = "Piano"
        part = ET.SubElement(score, "part", {"id": "P1"})
        m1 = ET.SubElement(part, "measure", {"number": "1"})
        attrs = ET.SubElement(m1, "attributes")
        ET.SubElement(attrs, "divisions").text = "4"

        rh_total = 0
        for step, alter, octv, dur in rh:
            n_el = ET.SubElement(m1, "note")
            p_el = ET.SubElement(n_el, "pitch")
            ET.SubElement(p_el, "step").text = step
            if alter != 0:
                ET.SubElement(p_el, "alter").text = str(alter)
            ET.SubElement(p_el, "octave").text = str(octv)
            ET.SubElement(n_el, "duration").text = str(dur)
            ET.SubElement(n_el, "voice").text = "1"
            ET.SubElement(n_el, "staff").text = "1"
            rh_total += dur

        bkp = ET.SubElement(m1, "backup")
        ET.SubElement(bkp, "duration").text = str(rh_total)

        for step, alter, octv, dur in lh:
            n_el = ET.SubElement(m1, "note")
            p_el = ET.SubElement(n_el, "pitch")
            ET.SubElement(p_el, "step").text = step
            if alter != 0:
                ET.SubElement(p_el, "alter").text = str(alter)
            ET.SubElement(p_el, "octave").text = str(octv)
            ET.SubElement(n_el, "duration").text = str(dur)
            ET.SubElement(n_el, "voice").text = "1"
            ET.SubElement(n_el, "staff").text = "2"

        piano_xml = ET.tostring(score, encoding="unicode", xml_declaration=True)
        streams = _extract_voice_note_streams(piano_xml)
        self.assertEqual(len(streams), 2)

        expected_hashes = (
            compute_symbolic_8gram_hashes(_make_melody_musicxml(rh, divisions=4), n=8)
            + compute_symbolic_8gram_hashes(_make_melody_musicxml(lh, divisions=4), n=8)
        )
        self.assertEqual(compute_symbolic_8gram_hashes(piano_xml, n=8), expected_hashes)

    def test_safe_numeric_parsing_decimal_and_whitespace(self) -> None:
        xml_decimal = """<?xml version='1.0' encoding='utf-8'?>
<score-partwise version="4.0">
  <part-list><score-part id="P1"><part-name>Dec</part-name></score-part></part-list>
  <part id="P1">
    <measure number="1">
      <attributes><divisions>4.0</divisions></attributes>
      <note><pitch><step>C</step><alter>   </alter><octave>4</octave></pitch><duration>4.0</duration><voice>1</voice></note>
      <note><pitch><step>E</step><alter>0.0</alter><octave>4</octave></pitch><duration>2.0</duration><voice>1</voice></note>
      <note><pitch><step>G</step><octave>4</octave></pitch><duration>2</duration><voice>1</voice></note>
      <note><pitch><step>F</step><alter>1.0</alter><octave>4</octave></pitch><duration>4</duration><voice>1</voice></note>
      <note><pitch><step>A</step><octave>4</octave></pitch><duration>6</duration><voice>1</voice></note>
      <note><pitch><step>G</step><octave>4</octave></pitch><duration>2</duration><voice>1</voice></note>
      <note><pitch><step>E</step><alter>-1.0</alter><octave>4</octave></pitch><duration>4</duration><voice>1</voice></note>
      <note><pitch><step>D</step><octave>4</octave></pitch><duration>2</duration><voice>1</voice></note>
    </measure>
  </part>
</score-partwise>"""
        ref_xml = _make_melody_musicxml(self.melody_a[:8], divisions=4)
        self.assertEqual(
            compute_symbolic_8gram_hashes(xml_decimal, n=8),
            compute_symbolic_8gram_hashes(ref_xml, n=8),
        )


if __name__ == "__main__":
    unittest.main()
