"""Unit tests for the deterministic MusicXML -> CER parser and 2-stage aligner."""

from __future__ import annotations

import tempfile
import unittest
import zipfile
from fractions import Fraction
from pathlib import Path

from omr_northstar.cer import (
    _extract_sounding_events,
    _measure_similarity,
    align_and_score_cer,
    align_measures,
    parse_musicxml,
    parse_musicxml_file,
    score_musicxml_pair,
)
from omr_northstar.scoring import DOMAINS, ERROR_BUCKET_KEYS, build_report, compute_page_quality
from omr_northstar.validate import validate_report


def _wrap_partwise(measures_xml: str, part_id: str = "P1") -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<score-partwise version="4.0">\n'
        f'  <part id="{part_id}">\n'
        f"{measures_xml}\n"
        "  </part>\n"
        "</score-partwise>\n"
    )


def _note_xml(
    step: str,
    octave: int,
    duration: int,
    *,
    alter: int = 0,
    voice: str = "1",
    staff: int = 1,
    chord: bool = False,
    tie_start: bool = False,
    tie_stop: bool = False,
    staccato: bool = False,
    lyric: str | None = None,
) -> str:
    parts = ["<note>"]
    if chord:
        parts.append("<chord/>")
    alt_tag = f"<alter>{alter}</alter>" if alter != 0 else ""
    parts.append(f"<pitch><step>{step}</step>{alt_tag}<octave>{octave}</octave></pitch>")
    parts.append(f"<duration>{duration}</duration>")
    if tie_start:
        parts.append('<tie type="start"/>')
    if tie_stop:
        parts.append('<tie type="stop"/>')
    parts.append(f"<voice>{voice}</voice>")
    parts.append(f"<staff>{staff}</staff>")
    if staccato:
        parts.append("<notations><articulations><staccato/></articulations></notations>")
    if lyric:
        parts.append(f"<lyric><text>{lyric}</text></lyric>")
    parts.append("</note>")
    return "".join(parts)


class TestCERParser(unittest.TestCase):
    """Verify exact rational timing, chords, backup/forward, ties, and context tracking."""

    def test_chords_backup_forward_and_rational_timing(self) -> None:
        m1 = (
            '<measure number="1">'
            "<attributes>"
            "<divisions>2</divisions>"
            "<staves>2</staves>"
            "<key><fifths>1</fifths></key>"
            "<time><beats>4</beats><beat-type>4</beat-type></time>"
            '<clef number="1"><sign>G</sign><line>2</line></clef>'
            '<clef number="2"><sign>F</sign><line>4</line></clef>'
            "</attributes>"
            + _note_xml("G", 4, 4, voice="1", staff=1, staccato=True, lyric="La")
            + _note_xml("B", 4, 4, voice="1", staff=1, chord=True)
            + "<forward><duration>2</duration></forward>"
            + _note_xml("D", 5, 2, voice="1", staff=1)
            + "<backup><duration>8</duration></backup>"
            + _note_xml("G", 2, 8, voice="2", staff=2)
            + '<barline location="right"><bar-style>light-heavy</bar-style></barline>'
            "</measure>"
        )
        score = parse_musicxml(_wrap_partwise(m1))
        # Multi-staff <part> is canonicalized into 2 separate 1-staff part streams
        self.assertEqual(score.parts, ("P1_s1", "P1_s2"))
        self.assertEqual(score.total_measures, 2)
        self.assertEqual(score.num_staves, 2)
        meas_s1, meas_s2 = score.measures
        self.assertEqual(meas_s1.barline_right, "final")
        self.assertEqual(meas_s2.barline_right, "final")
        self.assertEqual(len(meas_s1.notes), 3)
        self.assertEqual(len(meas_s2.notes), 1)

        g4, b4, d5 = meas_s1.notes
        (g2,) = meas_s2.notes
        # divisions=2 -> whole note = 8 ticks
        self.assertEqual(g4.onset, Fraction(0))
        self.assertEqual(g4.duration, Fraction(1, 2))
        self.assertEqual(g4.articulations, ("staccato",))
        self.assertEqual(g4.lyrics, ("La",))
        # Chord shares onset 0 and does not advance cursor
        self.assertTrue(b4.is_chord)
        self.assertEqual(b4.onset, Fraction(0))
        # Forward 2 ticks (1/4 whole) -> D5 onset = 1/2 + 1/4 = 3/4
        self.assertEqual(d5.onset, Fraction(3, 4))
        self.assertEqual(d5.duration, Fraction(1, 4))
        # Backup 8 ticks -> G2 starts at 0 on canonicalized second part stream (staff=1)
        self.assertEqual(g2.onset, Fraction(0))
        self.assertEqual(g2.staff, 1)
        self.assertEqual(meas_s2.note_contexts[0].clef, "F4")
        self.assertEqual(meas_s1.note_contexts[0].key_fifths, 1)
        self.assertEqual(meas_s2.note_contexts[0].key_fifths, 1)

    def test_mid_piece_context_changes_and_ties(self) -> None:
        m1 = (
            '<measure number="1">'
            "<attributes><divisions>1</divisions><key><fifths>0</fifths></key>"
            "<time><beats>2</beats><beat-type>4</beat-type></time>"
            "<clef><sign>G</sign><line>2</line></clef></attributes>"
            + _note_xml("C", 4, 2, tie_start=True)
            + "</measure>"
        )
        m2 = (
            '<measure number="2">'
            "<attributes><key><fifths>-2</fifths></key>"
            "<time><beats>3</beats><beat-type>4</beat-type></time>"
            "<clef><sign>C</sign><line>3</line></clef></attributes>"
            + _note_xml("C", 4, 1, tie_stop=True)
            + _note_xml("D", 4, 2)
            + "</measure>"
        )
        score = parse_musicxml(_wrap_partwise(m1 + m2))
        self.assertEqual(score.total_measures, 2)
        m2_cer = score.measures[1]
        change_kinds = {c[2] for c in m2_cer.context_changes}
        self.assertEqual(change_kinds, {"key", "time", "clef"})
        self.assertEqual(m2_cer.note_contexts[0].clef, "C3")
        self.assertEqual(m2_cer.note_contexts[0].key_fifths, -2)
        self.assertEqual(m2_cer.note_contexts[0].time_num, 3)

        # Tied C4 across m1+m2 resolves into a single sounding event in S_play
        res = align_and_score_cer(score, score)
        self.assertAlmostEqual(res.s_play, 1.0)
        self.assertAlmostEqual(res.s_note, 1.0)
        self.assertAlmostEqual(res.s_struct, 1.0)
        self.assertAlmostEqual(res.s_expr, 1.0)

    def test_backup_cursor_rounding_residual_snaps_to_zero(self) -> None:
        """A 1/40320 whole-note <backup> residual must snap to exact onset 0 on staff 2."""
        # divisions=10080 -> whole note = 40320 ticks, quarter note = 10080 ticks,
        # 3/4 measure = 30240 ticks; <backup> of 30239 leaves a 1/40320 residual.
        attrs = (
            "<attributes>"
            "<divisions>10080</divisions>"
            "<staves>2</staves>"
            "<time><beats>3</beats><beat-type>4</beat-type></time>"
            '<clef number="1"><sign>G</sign><line>2</line></clef>'
            '<clef number="2"><sign>F</sign><line>4</line></clef>'
            "</attributes>"
        )
        m_residual = (
            f'<measure number="1">{attrs}'
            + _note_xml("E", 5, 10080, voice="1", staff=1)
            + _note_xml("D", 5, 10080, voice="1", staff=1)
            + _note_xml("C", 5, 10080, voice="1", staff=1)
            + "<backup><duration>30239</duration></backup>"
            + _note_xml("C", 3, 10080, voice="2", staff=2)
            + _note_xml("G", 2, 10080, voice="2", staff=2)
            + _note_xml("C", 3, 10080, voice="2", staff=2)
            + "</measure>"
        )
        m_exact = (
            f'<measure number="1">{attrs}'
            + _note_xml("E", 5, 10080, voice="1", staff=1)
            + _note_xml("D", 5, 10080, voice="1", staff=1)
            + _note_xml("C", 5, 10080, voice="1", staff=1)
            + "<backup><duration>30240</duration></backup>"
            + _note_xml("C", 3, 10080, voice="2", staff=2)
            + _note_xml("G", 2, 10080, voice="2", staff=2)
            + _note_xml("C", 3, 10080, voice="2", staff=2)
            + "</measure>"
        )
        score_res = parse_musicxml(_wrap_partwise(m_residual))
        score_exact = parse_musicxml(_wrap_partwise(m_exact))

        staff2_notes = score_res.measures[1].notes
        self.assertEqual(
            [n.onset for n in staff2_notes],
            [Fraction(0), Fraction(1, 4), Fraction(1, 2)],
        )
        res = align_and_score_cer(score_res, score_exact)
        self.assertAlmostEqual(res.f1_t1a, 1.0)
        self.assertAlmostEqual(res.f1_t1b, 1.0)
        self.assertAlmostEqual(res.f1_t1c, 1.0)
        self.assertEqual(res.k_dropped_systems, 0)

    def test_rounded_triplet_durations_snap_to_exact_1_24(self) -> None:
        """Rounded triplet ticks (1654/40320, 1693/40320) must snap to exact 1/24 whole notes."""
        # In divisions=10080 (40320 ticks/whole), a 3-note eighth-triplet group sums to
        # 5040 ticks (1/8 whole), often rounded as 1654 + 1693 + 1693 = 5040.
        attrs_rounded = (
            "<attributes>"
            "<divisions>10080</divisions>"
            "<time><beats>3</beats><beat-type>8</beat-type></time>"
            "<clef><sign>G</sign><line>2</line></clef>"
            "</attributes>"
        )
        # 3 triplet groups = 9 notes (total 3/8 whole)
        rounded_notes = (
            _note_xml("C", 5, 1654)
            + _note_xml("D", 5, 1693)
            + _note_xml("E", 5, 1693)
            + _note_xml("F", 5, 1654)
            + _note_xml("G", 5, 1693)
            + _note_xml("A", 5, 1693)
            + _note_xml("G", 5, 1654)
            + _note_xml("F", 5, 1693)
            + _note_xml("E", 5, 1693)
        )
        xml_rounded = _wrap_partwise(f'<measure number="1">{attrs_rounded}{rounded_notes}</measure>')

        # Exact reference using divisions=6 (24 ticks/whole -> 1 tick = 1/24 whole)
        attrs_exact = (
            "<attributes>"
            "<divisions>6</divisions>"
            "<time><beats>3</beats><beat-type>8</beat-type></time>"
            "<clef><sign>G</sign><line>2</line></clef>"
            "</attributes>"
        )
        exact_notes = "".join(
            _note_xml(step, 5, 1)
            for step in ("C", "D", "E", "F", "G", "A", "G", "F", "E")
        )
        xml_exact = _wrap_partwise(f'<measure number="1">{attrs_exact}{exact_notes}</measure>')

        score_rounded = parse_musicxml(xml_rounded)
        score_exact = parse_musicxml(xml_exact)

        parsed_notes = score_rounded.measures[0].notes
        self.assertEqual([n.duration for n in parsed_notes], [Fraction(1, 24)] * 9)
        self.assertEqual(
            [n.onset for n in parsed_notes],
            [Fraction(i, 24) for i in range(9)],
        )

        res = align_and_score_cer(score_rounded, score_exact)
        self.assertAlmostEqual(res.f1_t1a, 1.0)
        self.assertAlmostEqual(res.f1_t1b, 1.0)
        self.assertAlmostEqual(res.f1_t1c, 1.0)
        self.assertAlmostEqual(res.bsvr, 1.0)
        self.assertAlmostEqual(res.mer, 1.0)

    def test_exact_musical_subdivisions_preserved(self) -> None:
        """Exact tuplets and fine subdivisions (1/3..1/128) must not be distorted."""
        # divisions = 13440 -> 4*divisions = 53760, divisible by 3,5,7,12,24,48,32,64,128
        exact_durs = [
            Fraction(1, 3),
            Fraction(1, 6),
            Fraction(1, 12),
            Fraction(1, 24),
            Fraction(1, 48),
            Fraction(1, 5),
            Fraction(1, 10),
            Fraction(1, 20),
            Fraction(1, 7),
            Fraction(1, 14),
            Fraction(1, 32),
            Fraction(1, 64),
            Fraction(1, 128),
        ]
        divisions = 13440  # 4 * divisions = 53760 (divisible by 128, 3, 5, 7)
        attrs = (
            f"<attributes><divisions>{divisions}</divisions>"
            "<time><beats>4</beats><beat-type>4</beat-type></time>"
            "<clef><sign>G</sign><line>2</line></clef></attributes>"
        )
        notes_xml = "".join(
            _note_xml("C", 4, int(d * 4 * divisions))
            for d in exact_durs
        )
        score = parse_musicxml(_wrap_partwise(f'<measure number="1">{attrs}{notes_xml}</measure>'))
        self.assertEqual([n.duration for n in score.measures[0].notes], exact_durs)


class TestTwoStageAlignment(unittest.TestCase):
    """Verify enharmonic distinction, voice-ID permutation invariance, and NW DP recovery."""

    def test_enharmonic_b_sharp_3_vs_c4(self) -> None:
        attrs = (
            "<attributes><divisions>1</divisions><time><beats>4</beats><beat-type>4</beat-type></time>"
            "<clef><sign>G</sign><line>2</line></clef></attributes>"
        )
        gt_xml = _wrap_partwise(
            f'<measure number="1">{attrs}' + _note_xml("C", 4, 4) + "</measure>"
        )
        pred_xml = _wrap_partwise(
            f'<measure number="1">{attrs}' + _note_xml("B", 3, 4, alter=1) + "</measure>"
        )
        res = align_and_score_cer(parse_musicxml(pred_xml), parse_musicxml(gt_xml))
        # B#3 and C4 both have MIDI 60 -> T1a and T1b match 100%, T1c strict engraving fails!
        self.assertAlmostEqual(res.f1_t1a, 1.0)
        self.assertAlmostEqual(res.f1_t1b, 1.0)
        self.assertAlmostEqual(res.f1_t1c, 0.0)
        self.assertAlmostEqual(res.s_play, 1.0)

    def test_voice_id_permutation_invariance(self) -> None:
        attrs = (
            "<attributes><divisions>1</divisions><time><beats>4</beats><beat-type>4</beat-type></time>"
            "<clef><sign>G</sign><line>2</line></clef></attributes>"
        )
        gt_xml = _wrap_partwise(
            f'<measure number="1">{attrs}'
            + _note_xml("E", 5, 4, voice="1")
            + "<backup><duration>4</duration></backup>"
            + _note_xml("C", 4, 4, voice="2")
            + "</measure>"
        )
        # Voices swapped ("7" and "3" instead of "1" and "2")
        pred_xml = _wrap_partwise(
            f'<measure number="1">{attrs}'
            + _note_xml("E", 5, 4, voice="7")
            + "<backup><duration>4</duration></backup>"
            + _note_xml("C", 4, 4, voice="3")
            + "</measure>"
        )
        res = align_and_score_cer(parse_musicxml(pred_xml), parse_musicxml(gt_xml))
        self.assertAlmostEqual(res.f1_t1c, 1.0)
        self.assertAlmostEqual(res.mer, 1.0)

    def test_measure_insertion_and_deletion_needleman_wunsch_recovery(self) -> None:
        attrs = (
            "<attributes><divisions>1</divisions><time><beats>4</beats><beat-type>4</beat-type></time>"
            "<clef><sign>G</sign><line>2</line></clef></attributes>"
        )
        m_c = f'<measure number="1">{attrs}' + _note_xml("C", 4, 4) + "</measure>"
        m_d = '<measure number="2">' + _note_xml("D", 4, 4) + "</measure>"
        m_e = '<measure number="3">' + _note_xml("E", 4, 4) + "</measure>"
        m_g = '<measure number="4">' + _note_xml("G", 4, 4) + "</measure>"
        m_f = '<measure number="5">' + _note_xml("F", 4, 4) + "</measure>"
        m_extra = '<measure number="X">' + _note_xml("A", 5, 4) + "</measure>"

        gt = parse_musicxml(_wrap_partwise(m_c + m_d + m_e + m_g + m_f))

        # 1. Pure measure deletion (m_d dropped): remaining 4 measures recover cleanly
        pred_del = parse_musicxml(_wrap_partwise(m_c + m_e + m_g + m_f))
        aln_del = align_measures(pred_del.measures, gt.measures)
        self.assertEqual(aln_del, [(0, 0), (None, 1), (1, 2), (2, 3), (3, 4)])
        res_del = align_and_score_cer(pred_del, gt)
        self.assertAlmostEqual(res_del.f1_t1c, 8.0 / 9.0, places=6)

        # 2. Pure measure insertion (m_extra inserted): all 5 GT measures recover cleanly
        pred_ins = parse_musicxml(_wrap_partwise(m_c + m_extra + m_d + m_e + m_g + m_f))
        aln_ins = align_measures(pred_ins.measures, gt.measures)
        self.assertEqual(aln_ins, [(0, 0), (1, None), (2, 1), (3, 2), (4, 3), (5, 4)])
        res_ins = align_and_score_cer(pred_ins, gt)
        self.assertAlmostEqual(res_ins.f1_t1c, 10.0 / 11.0, places=6)

        # 3. Combined deletion (m_d) + insertion (m_extra after m_g): 4 of 5 GT notes match
        pred_both = parse_musicxml(_wrap_partwise(m_c + m_e + m_g + m_extra + m_f))
        aln_both = align_measures(pred_both.measures, gt.measures)
        self.assertEqual(aln_both, [(0, 0), (None, 1), (1, 2), (2, 3), (3, None), (4, 4)])
        res_both = align_and_score_cer(pred_both, gt)
        self.assertAlmostEqual(res_both.f1_t1c, 0.80, places=6)

    def test_score_musicxml_pair_end_to_end_into_build_report(self) -> None:
        attrs = (
            "<attributes><divisions>1</divisions><time><beats>4</beats><beat-type>4</beat-type></time>"
            "<clef><sign>G</sign><line>2</line></clef></attributes>"
        )
        xml = _wrap_partwise(
            f'<measure number="1">{attrs}'
            + _note_xml("C", 4, 2)
            + _note_xml("G", 4, 2)
            + "</measure>"
        )
        pages = [
            score_musicxml_pair(
                xml,
                xml,
                page_id=f"p_{d}",
                work_id=f"w_{d}",
                domain=d,
                tier="T1",
            )
            for d in DOMAINS
        ]
        for p in pages:
            q = compute_page_quality(p)
            self.assertAlmostEqual(q.q_score, 100.0)
            self.assertFalse(q.catastrophic_failure)

        report = build_report(pages, system_name="cer-e2e")
        self.assertAlmostEqual(report["layer1_headline"]["ns_score"], 100.0)
        self.assertEqual(report["layer1_headline"]["ns_ci95"], [100.0, 100.0])
        self.assertAlmostEqual(report["layer1_headline"]["tier_weighted_ns"], 100.0)

        # Unparseable prediction gracefully triggers CF without crashing
        bad_pm = score_musicxml_pair(
            "<not-valid-xml",
            xml,
            page_id="p_bad",
            work_id="w_bad",
            domain="HWM",
            tier="T3",
        )
        bad_q = compute_page_quality(bad_pm)
        self.assertFalse(bad_pm.parseable)
        self.assertTrue(bad_q.catastrophic_failure)
        self.assertIn("unparseable", bad_q.cf_reasons)
        self.assertEqual(bad_q.q_score, 0.0)

    def test_multistaff_part_equivalent_to_separate_parts(self) -> None:
        """Bucket-A equivalence: 1 <part><staves>2</staves> == 2 <part><staves>1</staves>."""
        grand_staff_xml = _wrap_partwise(
            '<measure number="1">'
            "<attributes>"
            "<divisions>2</divisions>"
            "<staves>2</staves>"
            "<key><fifths>1</fifths></key>"
            "<time><beats>4</beats><beat-type>4</beat-type></time>"
            '<clef number="1"><sign>G</sign><line>2</line></clef>'
            '<clef number="2"><sign>F</sign><line>4</line></clef>'
            "</attributes>"
            + _note_xml("G", 4, 4, voice="1", staff=1, staccato=True)
            + _note_xml("B", 4, 4, voice="1", staff=1, tie_start=True)
            + "<backup><duration>8</duration></backup>"
            + _note_xml("G", 2, 8, voice="2", staff=2, tie_start=True)
            + "</measure>"
            + '<measure number="2">'
            + '<print new-system="yes"/>'
            + "<attributes><key><fifths>2</fifths></key></attributes>"
            + _note_xml("B", 4, 8, voice="1", staff=1, tie_stop=True)
            + "<backup><duration>8</duration></backup>"
            + _note_xml("G", 2, 8, voice="2", staff=2, tie_stop=True)
            + '<barline location="right"><bar-style>light-heavy</bar-style></barline>'
            + "</measure>"
        )

        two_parts_xml = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<score-partwise version="4.0">\n'
            '  <part id="P1">\n'
            '    <measure number="1">'
            "<attributes>"
            "<divisions>2</divisions>"
            "<staves>1</staves>"
            "<key><fifths>1</fifths></key>"
            "<time><beats>4</beats><beat-type>4</beat-type></time>"
            '<clef><sign>G</sign><line>2</line></clef>'
            "</attributes>"
            + _note_xml("G", 4, 4, voice="1", staff=1, staccato=True)
            + _note_xml("B", 4, 4, voice="1", staff=1, tie_start=True)
            + "</measure>\n"
            + '    <measure number="2">'
            + '<print new-system="yes"/>'
            + "<attributes><key><fifths>2</fifths></key></attributes>"
            + _note_xml("B", 4, 8, voice="1", staff=1, tie_stop=True)
            + '<barline location="right"><bar-style>light-heavy</bar-style></barline>'
            + "</measure>\n"
            "  </part>\n"
            '  <part id="P2">\n'
            '    <measure number="1">'
            "<attributes>"
            "<divisions>2</divisions>"
            "<staves>1</staves>"
            "<key><fifths>1</fifths></key>"
            "<time><beats>4</beats><beat-type>4</beat-type></time>"
            '<clef><sign>F</sign><line>4</line></clef>'
            "</attributes>"
            + _note_xml("G", 2, 8, voice="1", staff=1, tie_start=True)
            + "</measure>\n"
            + '    <measure number="2">'
            + '<print new-system="yes"/>'
            + "<attributes><key><fifths>2</fifths></key></attributes>"
            + _note_xml("G", 2, 8, voice="1", staff=1, tie_stop=True)
            + '<barline location="right"><bar-style>light-heavy</bar-style></barline>'
            + "</measure>\n"
            "  </part>\n"
            "</score-partwise>\n"
        )

        # Test both directions: grand-staff pred vs 2-part GT, and 2-part pred vs grand-staff GT
        for pred_xml, gt_xml in (
            (grand_staff_xml, two_parts_xml),
            (two_parts_xml, grand_staff_xml),
        ):
            pm = score_musicxml_pair(
                pred_xml,
                gt_xml,
                page_id="p_grand",
                work_id="w_grand",
                domain="CDR",
                tier="T1",
            )
            q = compute_page_quality(pm)
            self.assertAlmostEqual(pm.s_note, 1.0)
            self.assertAlmostEqual(pm.s_struct, 1.0)
            self.assertAlmostEqual(pm.s_expr, 1.0)
            self.assertAlmostEqual(pm.s_play, 1.0)
            self.assertFalse(q.catastrophic_failure)
            self.assertAlmostEqual(q.q_score, 100.0)


class TestForwardBSVRAndEdgeCases(unittest.TestCase):
    """Verify MusicXML <forward> BSVR accounting, additive meters, grace chords, and Layer 3 taxonomy."""

    def test_sparse_polyphonic_forward_explicit_and_inferred_staff_bsvr(self) -> None:
        """Valid sparse secondary voices using <forward> must achieve BSVR = 1.0."""
        m1 = (
            '<measure number="1">'
            "<attributes>"
            "<divisions>2</divisions>"
            "<staves>2</staves>"
            "<time><beats>4</beats><beat-type>4</beat-type></time>"
            '<clef number="1"><sign>G</sign><line>2</line></clef>'
            '<clef number="2"><sign>F</sign><line>4</line></clef>'
            "</attributes>"
            # Staff 1, Voice 1: full 4/4 (8 ticks)
            + _note_xml("C", 5, 4, voice="1", staff=1)
            + _note_xml("D", 5, 4, voice="1", staff=1)
            + "<backup><duration>8</duration></backup>"
            # Staff 1, Voice 2: sparse voice using explicit <forward> + note + <forward>
            + "<forward><duration>4</duration><voice>2</voice><staff>1</staff></forward>"
            + _note_xml("A", 4, 2, voice="2", staff=1)
            + "<forward><duration>2</duration><voice>2</voice><staff>1</staff></forward>"
            + "<backup><duration>8</duration></backup>"
            # Staff 2, Voice 5: <forward> with <voice> but omitted <staff> (inferred from note)
            + "<forward><duration>4</duration><voice>5</voice></forward>"
            + _note_xml("C", 3, 4, voice="5", staff=2)
            + "</measure>"
        )
        score = parse_musicxml(_wrap_partwise(m1))
        res = align_and_score_cer(score, score)
        self.assertAlmostEqual(res.bsvr, 1.0)
        self.assertAlmostEqual(res.mer, 1.0)
        self.assertAlmostEqual(res.s_struct, 1.0)

    def test_voiceless_forward_and_empty_forward_segment_handling(self) -> None:
        """Voiceless <forward> in a single-voice segment credits that voice; empty forward-only jumps do not."""
        m1 = (
            '<measure number="1">'
            "<attributes><divisions>2</divisions>"
            "<time><beats>4</beats><beat-type>4</beat-type></time>"
            "<clef><sign>G</sign><line>2</line></clef></attributes>"
            + _note_xml("G", 4, 4, voice="1", staff=1)
            + _note_xml("B", 4, 4, voice="1", staff=1, chord=True)
            + "<forward><duration>2</duration></forward>"
            + _note_xml("D", 5, 2, voice="1", staff=1)
            # Trailing cursor-only forward segment with no notes/rests must not fail BSVR
            + "<backup><duration>8</duration></backup>"
            + "<forward><duration>4</duration></forward>"
            + "</measure>"
        )
        score = parse_musicxml(_wrap_partwise(m1))
        res = align_and_score_cer(score, score)
        self.assertAlmostEqual(res.bsvr, 1.0)
        self.assertAlmostEqual(res.mer, 1.0)

        # Genuine underfull measure without <forward> must still fail BSVR
        m_under = (
            '<measure number="1">'
            "<attributes><divisions>2</divisions>"
            "<time><beats>4</beats><beat-type>4</beat-type></time>"
            "<clef><sign>G</sign><line>2</line></clef></attributes>"
            + _note_xml("G", 4, 4, voice="1", staff=1)
            + _note_xml("D", 5, 2, voice="1", staff=1)
            + "</measure>"
        )
        score_under = parse_musicxml(_wrap_partwise(m_under))
        res_under = align_and_score_cer(score_under, score_under)
        self.assertAlmostEqual(res_under.bsvr, 0.0)

    def test_additive_time_signature_and_grace_chord_onset(self) -> None:
        """Additive meter (2+3/8) sums beats to 5/8, and grace chords share current cursor onset."""
        m1 = (
            '<measure number="1">'
            "<attributes><divisions>2</divisions>"
            "<time><beats>2+3</beats><beat-type>8</beat-type></time>"
            "<clef><sign>G</sign><line>2</line></clef></attributes>"
            + _note_xml("C", 4, 2)  # beat 1: onset 0, dur 1/4
            # Grace chord at onset 1/4 (must NOT inherit last_note_onset=0)
            + "<note><grace/><pitch><step>D</step><octave>4</octave></pitch><voice>1</voice><staff>1</staff></note>"
            + "<note><grace/><chord/><pitch><step>F</step><octave>4</octave></pitch><voice>1</voice><staff>1</staff></note>"
            + _note_xml("E", 4, 3)  # onset 1/4, dur 3/8 -> total 5/8
            + "</measure>"
        )
        score = parse_musicxml(_wrap_partwise(m1))
        meas = score.measures[0]
        self.assertEqual(meas.contexts[1].time_num, 5)
        self.assertEqual(meas.contexts[1].time_den, 8)
        grace_notes = [n for n in meas.notes if n.grace]
        self.assertEqual(len(grace_notes), 2)
        self.assertEqual([n.onset for n in grace_notes], [Fraction(1, 4), Fraction(1, 4)])
        res = align_and_score_cer(score, score)
        self.assertAlmostEqual(res.bsvr, 1.0)
        self.assertAlmostEqual(res.mer, 1.0)

    def test_layer3_error_taxonomy_populated_and_schema_valid(self) -> None:
        """score_musicxml_pair populates Layer 3 error_buckets for build_report()."""
        attrs = (
            "<attributes><divisions>1</divisions><time><beats>4</beats><beat-type>4</beat-type></time>"
            "<clef><sign>G</sign><line>2</line></clef></attributes>"
        )
        gt_xml = _wrap_partwise(
            f'<measure number="1">{attrs}' + _note_xml("C", 4, 2) + _note_xml("E", 4, 2) + "</measure>"
        )
        # Pitch error on note 2 (F4 instead of E4)
        pred_xml = _wrap_partwise(
            f'<measure number="1">{attrs}' + _note_xml("C", 4, 2) + _note_xml("F", 4, 2) + "</measure>"
        )
        pm_err = score_musicxml_pair(
            pred_xml, gt_xml, page_id="p_err", work_id="w1", domain="CDR", tier="T1"
        )
        self.assertIsNotNone(pm_err.error_buckets)
        assert pm_err.error_buckets is not None
        self.assertEqual(set(pm_err.error_buckets.keys()), set(ERROR_BUCKET_KEYS))
        self.assertAlmostEqual(sum(pm_err.error_buckets.values()), 100.0, places=2)
        self.assertGreater(pm_err.error_buckets["pitch_accidental_pct"], 50.0)

        # Exact match yields all-zero error buckets
        pm_ok = score_musicxml_pair(
            gt_xml, gt_xml, page_id="p_ok", work_id="w2", domain="ALS", tier="T1"
        )
        assert pm_ok.error_buckets is not None
        self.assertEqual(sum(pm_ok.error_buckets.values()), 0.0)

        report = build_report(
            [
                pm_err,
                pm_ok,
                score_musicxml_pair(gt_xml, gt_xml, page_id="p_rwp", work_id="w3", domain="RWP", tier="T1"),
                score_musicxml_pair(gt_xml, gt_xml, page_id="p_hwm", work_id="w4", domain="HWM", tier="T1"),
            ],
            system_name="taxonomy-test",
        )
        self.assertEqual(validate_report(report), [])
        self.assertGreater(report["layer3_error_taxonomy"]["pitch_accidental_pct"], 0.0)


class TestRepetitionLoopVsOstinato(unittest.TestCase):
    """Verify _has_repetition_loop spares genuine ostinatos while flagging runaway loops."""

    @staticmethod
    def _bar(num: int, notes: list[tuple[str, int, int]], with_attrs: bool = False) -> str:
        attrs = (
            "<attributes><divisions>1</divisions><time><beats>4</beats><beat-type>4</beat-type></time>"
            "<clef><sign>G</sign><line>2</line></clef></attributes>"
            if with_attrs
            else ""
        )
        body = "".join(_note_xml(step, octv, dur) for step, octv, dur in notes)
        return f'<measure number="{num}">{attrs}{body}</measure>'

    def test_genuine_ostinato_with_minor_read_error_not_flagged(self) -> None:
        """A genuine 6-bar ostinato misread by one pitch must NOT trigger hallucination_loop."""
        pat_gt = [("C", 4, 1), ("E", 4, 1), ("G", 4, 1), ("E", 4, 1)]
        pat_pred = [("C", 4, 1), ("E", 4, 1), ("A", 4, 1), ("E", 4, 1)]
        gt_xml = _wrap_partwise(
            "".join(self._bar(i + 1, pat_gt, with_attrs=(i == 0)) for i in range(6))
        )
        pred_xml = _wrap_partwise(
            "".join(self._bar(i + 1, pat_pred, with_attrs=(i == 0)) for i in range(6))
        )
        res = align_and_score_cer(parse_musicxml(pred_xml), parse_musicxml(gt_xml))
        self.assertFalse(res.has_repetition_loop)

    def test_genuine_ostinato_with_subtle_gt_variation_not_flagged(self) -> None:
        """Smoothing a 1-note variation inside a 6-bar ostinato must NOT trigger hallucination_loop."""
        pat = [("C", 4, 1), ("E", 4, 1), ("G", 4, 1), ("E", 4, 1)]
        pat_var = [("C", 4, 1), ("E", 4, 1), ("A", 4, 1), ("E", 4, 1)]
        gt_bars = [pat, pat, pat, pat, pat_var, pat]
        gt_xml = _wrap_partwise(
            "".join(self._bar(i + 1, b, with_attrs=(i == 0)) for i, b in enumerate(gt_bars))
        )
        pred_xml = _wrap_partwise(
            "".join(self._bar(i + 1, pat, with_attrs=(i == 0)) for i in range(6))
        )
        res = align_and_score_cer(parse_musicxml(pred_xml), parse_musicxml(gt_xml))
        self.assertFalse(res.has_repetition_loop)

    def test_runaway_hallucination_loop_on_varied_score_flagged(self) -> None:
        """Decoder runaway repetition on a varied score MUST trigger hallucination_loop."""
        varied = [
            [("C", 4, 4)],
            [("D", 4, 4)],
            [("E", 4, 4)],
            [("F", 4, 4)],
            [("G", 4, 4)],
            [("A", 4, 4)],
            [("B", 4, 4)],
            [("C", 5, 4)],
        ]
        gt_xml = _wrap_partwise(
            "".join(self._bar(i + 1, b, with_attrs=(i == 0)) for i, b in enumerate(varied))
        )
        # Same-length collapse into 2-bar loop
        collapsed = varied[:2] + [[("E", 4, 4)], [("F", 4, 4)]] * 3
        pred_collapsed = _wrap_partwise(
            "".join(self._bar(i + 1, b, with_attrs=(i == 0)) for i, b in enumerate(collapsed))
        )
        res_col = align_and_score_cer(parse_musicxml(pred_collapsed), parse_musicxml(gt_xml))
        self.assertTrue(res_col.has_repetition_loop)

        # Runaway extra measures repeating 2-bar loop
        runaway = varied[:2] + [[("E", 4, 4)], [("F", 4, 4)]] * 5
        pred_runaway = _wrap_partwise(
            "".join(self._bar(i + 1, b, with_attrs=(i == 0)) for i, b in enumerate(runaway))
        )
        res_run = align_and_score_cer(parse_musicxml(pred_runaway), parse_musicxml(gt_xml))
        self.assertTrue(res_run.has_repetition_loop)

    def test_runaway_loop_extending_genuine_ostinato_flagged(self) -> None:
        """A runaway 12-bar loop on a 6-bar GT ostinato MUST still be flagged."""
        pat = [("C", 4, 1), ("E", 4, 1), ("G", 4, 1), ("E", 4, 1)]
        gt_xml = _wrap_partwise(
            "".join(self._bar(i + 1, pat, with_attrs=(i == 0)) for i in range(6))
        )
        pred_xml = _wrap_partwise(
            "".join(self._bar(i + 1, pat, with_attrs=(i == 0)) for i in range(12))
        )
        res = align_and_score_cer(parse_musicxml(pred_xml), parse_musicxml(gt_xml))
        self.assertTrue(res.has_repetition_loop)

    def test_five_bar_repetition_loop_detected(self) -> None:
        """Longer n-gram (n=5) runaway repetition loop must be flagged."""
        pat = [
            [("C", 4, 4)],
            [("D", 4, 4)],
            [("E", 4, 4)],
            [("F", 4, 4)],
            [("G", 4, 4)],
        ]
        gt_bars = [
            [("C", 4, 4)],
            [("D", 4, 4)],
            [("E", 4, 4)],
            [("F", 4, 4)],
            [("G", 4, 4)],
            [("A", 4, 4)],
            [("B", 4, 4)],
            [("C", 5, 4)],
        ]
        gt_xml = _wrap_partwise(
            "".join(self._bar(i + 1, b, with_attrs=(i == 0)) for i, b in enumerate(gt_bars))
        )
        pred_bars = pat * 3  # 15 measures repeating 5-bar chunk 3x
        pred_xml = _wrap_partwise(
            "".join(self._bar(i + 1, b, with_attrs=(i == 0)) for i, b in enumerate(pred_bars))
        )
        res = align_and_score_cer(parse_musicxml(pred_xml), parse_musicxml(gt_xml))
        self.assertTrue(res.has_repetition_loop)


class TestCERBugFixesAndInvariants(unittest.TestCase):
    """Comprehensive tests for chord invariance, cross-staff BSVR, part bijection, S_play, and parser edge cases."""

    def test_chord_order_invariance_and_articulation_chord_invariance(self) -> None:
        """Reversing chord serialization order and moving articulation across chord notes yields 1.0."""
        attrs = (
            "<attributes><divisions>1</divisions><time><beats>4</beats><beat-type>4</beat-type></time>"
            "<clef><sign>G</sign><line>2</line></clef></attributes>"
        )
        gt_xml = _wrap_partwise(
            f'<measure number="1">{attrs}'
            + _note_xml("C", 4, 4, staccato=True)
            + _note_xml("E", 4, 4, chord=True)
            + _note_xml("G", 4, 4, chord=True)
            + "</measure>"
        )
        pred_xml = _wrap_partwise(
            f'<measure number="1">{attrs}'
            + _note_xml("G", 4, 4)
            + _note_xml("E", 4, 4, chord=True, staccato=True)
            + _note_xml("C", 4, 4, chord=True)
            + "</measure>"
        )
        res = align_and_score_cer(parse_musicxml(pred_xml), parse_musicxml(gt_xml))
        self.assertAlmostEqual(res.f1_t1c, 1.0)
        self.assertAlmostEqual(res.mer, 1.0)
        self.assertAlmostEqual(res.s_expr, 1.0)
        self.assertAlmostEqual(res.bsvr, 1.0)

    def test_cross_staff_chord_and_cross_staff_voice_bsvr(self) -> None:
        """Cross-staff chords clear orphaned is_chord=True and cross-staff voices achieve BSVR=1.0."""
        m1 = (
            '<measure number="1">'
            "<attributes><divisions>1</divisions><staves>2</staves>"
            "<time><beats>4</beats><beat-type>4</beat-type></time>"
            '<clef number="1"><sign>G</sign><line>2</line></clef>'
            '<clef number="2"><sign>F</sign><line>4</line></clef>'
            "</attributes>"
            # Beat 1: cross-staff chord (staff 1 root + staff 2 chord note)
            + _note_xml("E", 4, 1, voice="1", staff=1)
            + _note_xml("C", 3, 1, voice="1", staff=2, chord=True)
            # Beats 2-4: voice 1 alternates between staff 2 and staff 1
            + _note_xml("G", 3, 1, voice="1", staff=2)
            + _note_xml("D", 4, 1, voice="1", staff=1)
            + _note_xml("F", 3, 1, voice="1", staff=2)
            + "</measure>"
        )
        score = parse_musicxml(_wrap_partwise(m1))
        # Staff 2's first note at onset 0 must NOT have orphaned is_chord=True
        s2_notes = score.measures[1].notes
        self.assertFalse(s2_notes[0].is_chord)
        res = align_and_score_cer(score, score)
        self.assertAlmostEqual(res.bsvr, 1.0)
        self.assertAlmostEqual(res.mer, 1.0)
        self.assertAlmostEqual(res.s_struct, 1.0)

    def test_hungarian_part_bijection_and_empty_gt_part_pruning(self) -> None:
        """Reordered parts match via Hungarian bijection, and all-empty hidden GT parts are pruned."""
        p1_xml = (
            '  <part id="P1"><measure number="1">'
            "<attributes><divisions>1</divisions><time><beats>4</beats><beat-type>4</beat-type></time>"
            "<clef><sign>G</sign><line>2</line></clef></attributes>"
            + _note_xml("E", 5, 4)
            + "</measure></part>\n"
        )
        p2_xml = (
            '  <part id="P2"><measure number="1">'
            "<attributes><divisions>1</divisions><time><beats>4</beats><beat-type>4</beat-type></time>"
            "<clef><sign>F</sign><line>4</line></clef></attributes>"
            + _note_xml("C", 3, 4)
            + "</measure></part>\n"
        )
        p_empty_xml = (
            '  <part id="P3"><measure number="1">'
            "<attributes><divisions>1</divisions><time><beats>4</beats><beat-type>4</beat-type></time>"
            "<clef><sign>G</sign><line>2</line></clef></attributes>"
            '<note><rest measure="yes"/><duration>4</duration><voice>1</voice></note>'
            "</measure></part>\n"
        )
        gt_xml = f'<?xml version="1.0"?><score-partwise>{p1_xml}{p2_xml}{p_empty_xml}</score-partwise>'
        # Prediction has P2 and P1 in swapped order and omits the hidden empty part P3
        pred_xml = f'<?xml version="1.0"?><score-partwise>{p2_xml}{p1_xml}</score-partwise>'
        res = align_and_score_cer(parse_musicxml(pred_xml), parse_musicxml(gt_xml))
        self.assertEqual(res.m_gt, 2)
        self.assertEqual(res.m_pred, 2)
        self.assertAlmostEqual(res.f1_t1c, 1.0)
        self.assertAlmostEqual(res.mer, 1.0)
        self.assertAlmostEqual(res.s_struct, 1.0)

    def test_measure_similarity_no_barline_floor_and_pickup_shift_alignment(self) -> None:
        """Unrelated measures get 0.0 similarity (no +0.30 floor), enabling clean pickup alignment."""
        attrs = (
            "<attributes><divisions>1</divisions><time><beats>4</beats><beat-type>4</beat-type></time>"
            "<clef><sign>G</sign><line>2</line></clef></attributes>"
        )
        m0 = f'<measure number="0" implicit="yes">{attrs}' + _note_xml("G", 4, 1) + "</measure>"
        m1 = '<measure number="1">' + _note_xml("C", 4, 4) + "</measure>"
        m2 = '<measure number="2">' + _note_xml("D", 4, 4) + "</measure>"
        gt = parse_musicxml(_wrap_partwise(m0 + m1 + m2))
        pred = parse_musicxml(_wrap_partwise(f'<measure number="1">{attrs}' + _note_xml("C", 4, 4) + "</measure>" + m2))

        self.assertEqual(_measure_similarity(gt.measures[0], gt.measures[1]), 0.0)
        aln = align_measures(pred.measures, gt.measures)
        self.assertEqual(aln, [(None, 0), (0, 1), (1, 2)])

    def test_bsvr_pickup_complement_cadenza_anti_gaming_and_full_rest(self) -> None:
        """Pickup/complement/split/senza-misura/full-measure rest pass BSVR; gaming implicit='yes' fails."""
        attrs_34 = (
            "<attributes><divisions>1</divisions><time><beats>3</beats><beat-type>4</beat-type></time>"
            "<clef><sign>G</sign><line>2</line></clef></attributes>"
        )
        gt_xml = _wrap_partwise(
            # Pickup measure number="0" (1 beat)
            f'<measure number="0">{attrs_34}' + _note_xml("G", 4, 1) + "</measure>"
            # Normal 3/4 measure with full-measure rest encoded as whole-note duration=4
            + '<measure number="1"><note><rest measure="yes"/><duration>4</duration><voice>1</voice></note></measure>'
            # Split measure number="X1"
            + '<measure number="X1">' + _note_xml("C", 4, 1) + "</measure>"
            # Senza-misura cadenza measure
            + '<measure number="2"><attributes><time><senza-misura/></time></attributes>'
            + _note_xml("D", 4, 5) + "</measure>"
            # Return to 3/4 + final complement bar (2 beats)
            + '<measure number="3"><attributes><time><beats>3</beats><beat-type>4</beat-type></time></attributes>'
            + _note_xml("E", 4, 2) + "</measure>"
        )
        gt_score = parse_musicxml(gt_xml)
        self.assertEqual(gt_score.measures[1].rests[0].duration, Fraction(3, 4))
        res_gt = align_and_score_cer(gt_score, gt_score)
        self.assertAlmostEqual(res_gt.bsvr, 1.0)
        self.assertAlmostEqual(res_gt.mer, 1.0)

        # Gaming attempt: pred marks a normal GT measure as implicit="yes" with underfull beats
        normal_gt = parse_musicxml(
            _wrap_partwise(f'<measure number="1">{attrs_34}' + _note_xml("C", 4, 3) + "</measure>")
        )
        gamed_pred = parse_musicxml(
            _wrap_partwise(f'<measure number="1" implicit="yes">{attrs_34}' + _note_xml("C", 4, 1) + "</measure>")
        )
        res_game = align_and_score_cer(gamed_pred, normal_gt)
        self.assertAlmostEqual(res_game.bsvr, 0.0)

        # Non-implicit completely empty predicted measure aligned to non-empty GT fails BSVR
        empty_pred = parse_musicxml(_wrap_partwise(f'<measure number="1">{attrs_34}</measure>'))
        res_empty = align_and_score_cer(empty_pred, normal_gt)
        self.assertAlmostEqual(res_empty.bsvr, 0.0)

    def test_k_dropped_systems_multistaff_and_empty_breaks(self) -> None:
        """Dropping 1 visual system on a 4-staff score counts k_dropped=1, not 4; empty breaks = 1 system."""
        def _part_2sys(pid: str, step: str, include_sys2: bool) -> str:
            attrs = (
                "<attributes><divisions>1</divisions><time><beats>4</beats><beat-type>4</beat-type></time>"
                "<clef><sign>G</sign><line>2</line></clef></attributes>"
            )
            m1 = f'<measure number="1">{attrs}' + _note_xml(step, 4, 4) + "</measure>"
            m2 = '<measure number="2"><print new-system="yes"/>' + _note_xml(step, 5, 4) + "</measure>"
            return f'<part id="{pid}">{m1}{m2 if include_sys2 else ""}</part>'

        gt_4 = (
            '<?xml version="1.0"?><score-partwise>'
            + _part_2sys("P1", "C", True)
            + _part_2sys("P2", "D", True)
            + _part_2sys("P3", "E", True)
            + _part_2sys("P4", "F", True)
            + "</score-partwise>"
        )
        pred_4 = (
            '<?xml version="1.0"?><score-partwise>'
            + _part_2sys("P1", "C", False)
            + _part_2sys("P2", "D", False)
            + _part_2sys("P3", "E", False)
            + _part_2sys("P4", "F", False)
            + "</score-partwise>"
        )
        res = align_and_score_cer(parse_musicxml(pred_4), parse_musicxml(gt_4))
        self.assertEqual(res.k_dropped_systems, 1)

        # Empty system_breaks treated as 1 system [0, m_gt)
        gt_1 = parse_musicxml(
            _wrap_partwise(
                '<measure number="1"><attributes><divisions>1</divisions>'
                "<time><beats>4</beats><beat-type>4</beat-type></time></attributes>"
                + _note_xml("C", 4, 4) + "</measure>"
            )
        )
        pred_empty = parse_musicxml(
            _wrap_partwise(
                '<measure number="1"><attributes><divisions>1</divisions>'
                "<time><beats>4</beats><beat-type>4</beat-type></time></attributes></measure>"
            )
        )
        self.assertEqual(align_and_score_cer(pred_empty, gt_1).k_dropped_systems, 1)

    def test_sounding_events_tie_flush_contiguity_pickup_and_cross_staff(self) -> None:
        """S_play flushes overwritten ties, requires contiguity, uses pickup duration, chains cross-staff ties."""
        # 1. Overwritten tie_start + non-contiguous tie_stop
        xml_ties = _wrap_partwise(
            '<measure number="1"><attributes><divisions>1</divisions>'
            "<time><beats>4</beats><beat-type>4</beat-type></time></attributes>"
            + _note_xml("C", 4, 1, tie_start=True)
            + _note_xml("C", 4, 1, tie_start=True)  # overwrites first -> must flush first
            + "<note><rest/><duration>1</duration><voice>1</voice></note>"
            + _note_xml("C", 4, 1, tie_stop=True)   # non-contiguous -> must not chain
            + "</measure>"
        )
        ev = _extract_sounding_events(parse_musicxml(xml_ties))
        self.assertEqual(sum(ev.values()), 3)
        self.assertEqual(ev[(60, Fraction(0), Fraction(1, 4))], 1)
        self.assertEqual(ev[(60, Fraction(1, 4), Fraction(1, 4))], 1)
        self.assertEqual(ev[(60, Fraction(3, 4), Fraction(1, 4))], 1)

        # 2. Pickup actual duration + cross-staff tie in grand-staff part
        xml_xstaff = _wrap_partwise(
            '<measure number="0" implicit="yes"><attributes><divisions>1</divisions><staves>2</staves>'
            "<time><beats>4</beats><beat-type>4</beat-type></time>"
            '<clef number="1"><sign>G</sign><line>2</line></clef>'
            '<clef number="2"><sign>F</sign><line>4</line></clef></attributes>'
            + _note_xml("C", 4, 1, voice="1", staff=1, tie_start=True)
            + "</measure>"
            + '<measure number="1">'
            + _note_xml("C", 4, 4, voice="1", staff=2, tie_stop=True)
            + "</measure>"
        )
        ev_xs = _extract_sounding_events(parse_musicxml(xml_xstaff))
        self.assertEqual(ev_xs, {(60, Fraction(0), Fraction(5, 4)): 1})

    def test_parser_decimals_zero_divisions_timewise_mxl_transpose_and_expressives(self) -> None:
        """Verify decimals, divisions=0 fallback, timewise, .mxl, transpose, unpitched, wedges, and NFC."""
        # 1. Decimal divisions/alter/duration, divisions=0 fallback, empty child text, negative duration clamp
        xml_dec = _wrap_partwise(
            '<measure number="1">'
            "<attributes><divisions>0</divisions><transpose><chromatic>-2</chromatic><octave-change>-1</octave-change></transpose>"
            "<time><beats>3</beats><beat-type>4</beat-type><beats>2</beats><beat-type>8</beat-type></time>"
            "<clef><sign>G</sign><line>2</line><clef-octave-change>-1</clef-octave-change></clef>"
            "</attributes>"
            "<note><pitch><step>C</step><alter>1.0</alter><octave>5</octave></pitch>"
            "<type>quarter</type><dot/><voice>1</voice><staff> </staff>"
            '<notations><tied type="continue"/></notations></note>'
            "<note><unpitched><display-step>E</display-step><display-octave>4</display-octave></unpitched>"
            "<type>eighth</type><voice>1</voice></note>"
            "<note><rest/><type>half</type><voice>1</voice><notations><fermata/></notations></note>"
            "</measure>"
        )
        sc = parse_musicxml(xml_dec)
        m0 = sc.measures[0]
        # Compound time 3/4 + 2/8 -> 8/8
        self.assertEqual((m0.contexts[1].time_num, m0.contexts[1].time_den), (8, 8))
        self.assertEqual(m0.contexts[1].clef, "G2(-1o)")
        # First note: written C#5 (MIDI 73) transposed by -14 -> sounding MIDI 59; dotted quarter = 3/8
        n0 = m0.notes[0]
        self.assertEqual((n0.step, n0.alter, n0.octave), ("C", 1, 5))
        self.assertEqual(n0.midi, 59)
        self.assertEqual(n0.duration, Fraction(3, 8))
        self.assertTrue(n0.tie_start and n0.tie_stop)
        # Second note: unpitched E4
        self.assertEqual((m0.notes[1].step, m0.notes[1].octave), ("E", 4))
        # Rest has fermata articulation
        self.assertEqual(m0.rests[0].articulations, ("fermata",))

        # 2. <score-timewise> equals <score-partwise>
        tw_xml = (
            '<?xml version="1.0"?><score-timewise><measure number="1">'
            '<part id="P1"><attributes><divisions>1</divisions>'
            "<time><beats>4</beats><beat-type>4</beat-type></time></attributes>"
            + _note_xml("C", 4, 4)
            + "</part></measure></score-timewise>"
        )
        pw_xml = _wrap_partwise(
            '<measure number="1"><attributes><divisions>1</divisions>'
            "<time><beats>4</beats><beat-type>4</beat-type></time></attributes>"
            + _note_xml("C", 4, 4) + "</measure>"
        )
        res_tw = align_and_score_cer(parse_musicxml(tw_xml), parse_musicxml(pw_xml))
        self.assertAlmostEqual(res_tw.f1_t1c, 1.0)

        # 3. parse_musicxml_file supports .mxl ZIP containers
        with tempfile.TemporaryDirectory() as tmpdir:
            mxl_path = Path(tmpdir) / "sample.mxl"
            container_xml = (
                '<?xml version="1.0"?><container><rootfiles>'
                '<rootfile full-path="score.musicxml"/>'
                "</rootfiles></container>"
            )
            with zipfile.ZipFile(mxl_path, "w") as zf:
                zf.writestr("META-INF/container.xml", container_xml)
                zf.writestr("score.musicxml", pw_xml)
            mxl_score = parse_musicxml_file(mxl_path)
            self.assertEqual(mxl_score.total_notes, 1)

        # 4. Cross-measure wedges, octave-shift continue, NFC words, metronome beat-unit, harmony root-alter=0, no spurious ctx
        xml_expr = _wrap_partwise(
            '<measure number="1">'
            "<attributes><divisions>1</divisions><staves>2</staves>"
            "<time><beats>4</beats><beat-type>4</beat-type></time>"
            '<clef number="1"><sign>G</sign><line>2</line></clef></attributes>'
            '<attributes><clef number="2"><sign>G</sign><line>2</line></clef></attributes>'
            '<direction><direction-type><wedge type="crescendo" number="1"/></direction-type><staff>1</staff></direction>'
            '<direction><direction-type><octave-shift type="down" size="8"/></direction-type><staff>1</staff></direction>'
            '<direction><direction-type><words>Cafe\u0301</words></direction-type><staff>1</staff></direction>'
            '<direction><direction-type><metronome><beat-unit>quarter</beat-unit><per-minute>120</per-minute></metronome></direction-type><staff>1</staff></direction>'
            "<harmony><root><root-step>C</root-step><root-alter>0</root-alter></root><kind>major</kind><staff>1</staff></harmony>"
            + _note_xml("C", 5, 4, voice="1", staff=1)
            + "<backup><duration>4</duration></backup>"
            + _note_xml("C", 4, 4, voice="2", staff=2)
            + "</measure>"
            + '<measure number="2">'
            + '<direction><direction-type><octave-shift type="continue" size="8"/></direction-type><staff>1</staff></direction>'
            + _note_xml("D", 5, 4, voice="1", staff=1)
            + '<direction><direction-type><wedge type="stop" number="1"/></direction-type><staff>1</staff></direction>'
            + "<backup><duration>4</duration></backup>"
            + _note_xml("D", 4, 4, voice="2", staff=2)
            + "</measure>"
        )
        sc_expr = parse_musicxml(xml_expr)
        # Staff 2 m0 must NOT emit spurious clef change despite second <attributes> block
        self.assertEqual(sc_expr.measures[2].context_changes, ())
        # Measure 2 staff 1 keeps ottava=1 across <octave-shift type="continue"/>
        self.assertEqual(sc_expr.measures[1].note_contexts[0].ottava, 1)
        # Wedge carried across m1->m2 emits 1 wedge on m1, 0 on m2
        wedges_m1 = [d for d in sc_expr.measures[0].directions if d.kind == "wedge"]
        wedges_m2 = [d for d in sc_expr.measures[1].directions if d.kind == "wedge"]
        self.assertEqual(len(wedges_m1), 1)
        self.assertEqual(len(wedges_m2), 0)
        # NFC words, metronome beat-unit, and root-alter=0 normalization
        dir_map = {(d.kind, d.value) for d in sc_expr.measures[0].directions}
        self.assertIn(("words", "Caf\u00e9"), dir_map)
        self.assertIn(("tempo", "quarter=120"), dir_map)
        self.assertIn(("harmony", "C:major"), dir_map)

    def test_context_drift_taxonomy_negative_page_and_type_error_handling(self) -> None:
        """Clef mismatch populates global_context_drift_pct even with 0 pitch matches; TypeError caught."""
        gt_xml = _wrap_partwise(
            '<measure number="1"><attributes><divisions>1</divisions>'
            "<time><beats>4</beats><beat-type>4</beat-type></time>"
            "<clef><sign>G</sign><line>2</line></clef></attributes>"
            + _note_xml("C", 5, 4) + "</measure>"
        )
        wrong_clef_xml = _wrap_partwise(
            '<measure number="1"><attributes><divisions>1</divisions>'
            "<time><beats>4</beats><beat-type>4</beat-type></time>"
            "<clef><sign>F</sign><line>4</line></clef></attributes>"
            + _note_xml("E", 3, 4) + "</measure>"
        )
        res_ctx = align_and_score_cer(parse_musicxml(wrong_clef_xml), parse_musicxml(gt_xml))
        self.assertGreater(res_ctx.error_buckets["global_context_drift_pct"], 0.0)

        # Negative/empty GT page with hallucinated pred notes -> 100% hallucination_truncation_pct
        empty_gt = parse_musicxml(
            _wrap_partwise(
                '<measure number="1"><attributes><divisions>1</divisions>'
                "<time><beats>4</beats><beat-type>4</beat-type></time></attributes></measure>"
            )
        )
        halluc_pred = parse_musicxml(gt_xml)
        res_neg = align_and_score_cer(halluc_pred, empty_gt)
        self.assertAlmostEqual(res_neg.error_buckets["hallucination_truncation_pct"], 100.0)
        self.assertAlmostEqual(res_neg.error_buckets["pitch_accidental_pct"], 0.0)

        # None pred_xml caught cleanly via TypeError
        pm_none = score_musicxml_pair(
            None,  # type: ignore[arg-type]
            gt_xml,
            page_id="p_none",
            work_id="w_none",
            domain="CDR",
        )
        self.assertFalse(pm_none.parseable)


if __name__ == "__main__":
    unittest.main()
