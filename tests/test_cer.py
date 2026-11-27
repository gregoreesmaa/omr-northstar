"""Unit tests for the deterministic MusicXML -> CER parser and 2-stage aligner."""

from __future__ import annotations

import unittest
from fractions import Fraction

from omr_northstar.cer import (
    align_and_score_cer,
    align_measures,
    parse_musicxml,
    score_musicxml_pair,
)
from omr_northstar.scoring import DOMAINS, build_report, compute_page_quality


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


if __name__ == "__main__":
    unittest.main()
