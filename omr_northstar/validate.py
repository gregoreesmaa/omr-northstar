"""Stdlib manifest, split-disjointness, governance, and report validator.

Zero external dependencies (Python 3.10+ standard library only).
Enforces OMR-NorthStar schema bounds and governance rules:
  - Strict permissive license allowlist (zero CC-BY-NC or proprietary licenses).
  - <= 15% max share per engraver on CDR (when >= 10 CDR pages).
  - Zero work_id overlap across dev/test splits.
  - Zero writer_id / copyist overlap on HWM across splits.
  - Mandatory canary string presence on dev manifests.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Mapping, Sequence

from .scoring import DOMAINS, ERROR_BUCKET_KEYS, TIERS

PERMISSIVE_LICENSES: frozenset[str] = frozenset(
    {
        "CC0-1.0",
        "PDM-1.0",
        "CC-BY-4.0",
        "MIT",
        "BSD-3-Clause",
        "Apache-2.0",
    }
)
NC_EVAL_ONLY_LICENSES: frozenset[str] = frozenset(
    {
        "CC-BY-NC-4.0",
        "CC-BY-NC-SA-4.0",
        "CC-BY-NC-3.0",
    }
)
VALID_TRACKS: frozenset[str] = frozenset({"core_permissive", "research_nc_eval_only"})
LEGACY_PD_ALIASES: frozenset[str] = frozenset({"Public-Domain"})

VALID_VISUAL_TIERS: frozenset[str] = frozenset({"E", "M", "H", "B"})
VALID_VISUAL_SEVERITIES: frozenset[str] = frozenset({"V1", "V2", "V3", "V4"})
VALID_PAIRED_PROTOCOLS: frozenset[str] = frozenset({"A", "B", "C", "D1", "D2", "D3", "E"})
VALID_GT_VERIFICATION: frozenset[str] = frozenset(
    {"born_digital_canonical", "double_blind_human_adjudicated"}
)
VALID_REPERTOIRE_STRATA: frozenset[str] = frozenset(
    {
        "T1.1", "T1.2", "T1.3", "T1.4",
        "T2.1", "T2.2", "T2.3", "T2.4",
        "T3.1", "T3.2", "T3.3", "T3.4", "T3.5", "T3.6", "T3.7",
    }
)
VALID_BIT_DEPTHS: frozenset[int] = frozenset({1, 8, 16, 24, 32})
VALID_STRESS_FAMILIES: frozenset[str] = frozenset(
    {"1A", "1B", "1C", "2A", "2B", "2C", "3A", "3B", "3C"}
)
_FRESH_SPLIT_RE = re.compile(r"^fresh-[0-9]{4}$")

REQUIRED_PAGE_FIELDS: tuple[str, ...] = (
    "page_id",
    "work_id",
    "split",
    "domain",
    "complexity_tier",
    "visual_tier",
    "repertoire_stratum",
    "geometry",
    "photometry",
    "optics_codec",
    "provenance",
    "license",
)

CDR_ENGRAVER_MAX_SHARE: float = 0.15
CDR_ENGRAVER_MIN_PAGES: int = 10


def _is_valid_split(split: Any) -> bool:
    if not isinstance(split, str):
        return False
    return split in {"dev-v1", "test-v1", "fresh"} or bool(_FRESH_SPLIT_RE.match(split))


def _is_number(val: Any) -> bool:
    return isinstance(val, (int, float)) and not isinstance(val, bool)


def _extract_pages(manifest_or_pages: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    if isinstance(manifest_or_pages, Mapping):
        pages = manifest_or_pages.get("pages", [])
        return list(pages) if isinstance(pages, Sequence) and not isinstance(pages, (str, bytes)) else []
    if isinstance(manifest_or_pages, Sequence) and not isinstance(manifest_or_pages, (str, bytes)):
        return list(manifest_or_pages)
    return []


def validate_page_records(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_split: str | None = None,
    expected_track: str | None = None,
    manifest_eval_only: bool | None = None,
    allow_legacy_pd: bool = False,
) -> list[str]:
    """Validate page metadata records and single-split governance rules."""
    errors: list[str] = []
    seen_ids: set[str] = set()
    base_permissive = (
        PERMISSIVE_LICENSES | LEGACY_PD_ALIASES if allow_legacy_pd else PERMISSIVE_LICENSES
    )
    cdr_pages: list[Mapping[str, Any]] = []

    for idx, rec in enumerate(records):
        prefix = f"record[{idx}]"
        if not isinstance(rec, Mapping):
            errors.append(f"{prefix}: expected object, got {type(rec).__name__}")
            continue

        is_neg = bool(rec.get("is_negative_control", False))

        for req in REQUIRED_PAGE_FIELDS:
            if req not in rec:
                errors.append(f"{prefix}: missing required field '{req}'")
        if not is_neg and "staff_space_px" not in rec:
            errors.append(f"{prefix}: missing required field 'staff_space_px'")

        pid = rec.get("page_id")
        if not isinstance(pid, str) or not pid.strip():
            errors.append(f"{prefix}: invalid page_id {pid!r}")
        elif pid in seen_ids:
            errors.append(f"{prefix}: duplicate page_id '{pid}'")
        else:
            seen_ids.add(pid)

        wid = rec.get("work_id")
        if not isinstance(wid, str) or not wid.strip():
            errors.append(f"{prefix}: invalid work_id {wid!r}")

        sp = rec.get("split")
        if not _is_valid_split(sp):
            errors.append(f"{prefix}: invalid split {sp!r}")
        elif expected_split is not None and sp != expected_split:
            errors.append(f"{prefix}: page split {sp!r} does not match manifest split {expected_split!r}")

        dom = rec.get("domain")
        if dom not in DOMAINS:
            errors.append(f"{prefix}: invalid domain {dom!r}")
        elif dom == "CDR":
            cdr_pages.append(rec)

        if rec.get("complexity_tier") not in TIERS:
            errors.append(f"{prefix}: invalid complexity_tier {rec.get('complexity_tier')!r}")

        if rec.get("visual_tier") not in VALID_VISUAL_TIERS:
            errors.append(f"{prefix}: invalid visual_tier {rec.get('visual_tier')!r}")

        if "visual_severity" in rec and rec["visual_severity"] is not None:
            if rec["visual_severity"] not in VALID_VISUAL_SEVERITIES:
                errors.append(f"{prefix}: invalid visual_severity {rec['visual_severity']!r}")

        if rec.get("repertoire_stratum") not in VALID_REPERTOIRE_STRATA:
            errors.append(f"{prefix}: invalid repertoire_stratum {rec.get('repertoire_stratum')!r}")

        if "paired_protocol" in rec and rec["paired_protocol"] is not None:
            if rec["paired_protocol"] not in VALID_PAIRED_PROTOCOLS:
                errors.append(f"{prefix}: invalid paired_protocol {rec['paired_protocol']!r}")

        if "human_agreement_score" in rec and rec["human_agreement_score"] is not None:
            has_val = rec["human_agreement_score"]
            if not _is_number(has_val) or not (0.0 <= float(has_val) <= 1.0):
                errors.append(f"{prefix}: human_agreement_score must be in [0, 1], got {has_val!r}")

        if "staff_space_px" in rec and rec["staff_space_px"] is not None:
            ss = rec["staff_space_px"]
            if not isinstance(ss, Mapping) or "median" not in ss:
                errors.append(f"{prefix}: staff_space_px must be an object with 'median'")
            else:
                med = ss["median"]
                if not _is_number(med) or (float(med) <= 0.0 and not is_neg) or float(med) < 0.0:
                    errors.append(f"{prefix}: invalid staff_space_px.median {med!r}")

        geom = rec.get("geometry")
        if isinstance(geom, Mapping):
            rot = geom.get("rotation_deg")
            if not _is_number(rot) or not (-180.0 <= float(rot) <= 180.0):
                errors.append(f"{prefix}: geometry.rotation_deg out of [-180, 180]: {rot!r}")
            for kfield in ("keystone_x_deg", "keystone_y_deg"):
                kv = geom.get(kfield)
                if not _is_number(kv) or not (-60.0 <= float(kv) <= 60.0):
                    errors.append(f"{prefix}: geometry.{kfield} out of [-60, 60]: {kv!r}")

        phot = rec.get("photometry")
        if isinstance(phot, Mapping):
            wc = phot.get("weber_contrast")
            if not _is_number(wc) or not (0.0 <= float(wc) <= 1.0):
                errors.append(f"{prefix}: photometry.weber_contrast out of [0, 1]: {wc!r}")
            ev = phot.get("illumination_gradient_ev")
            if not _is_number(ev) or float(ev) < 0.0:
                errors.append(f"{prefix}: photometry.illumination_gradient_ev must be >= 0: {ev!r}")

        opt = rec.get("optics_codec")
        if isinstance(opt, Mapping):
            bs = opt.get("blur_sigma_px")
            if not _is_number(bs) or float(bs) < 0.0:
                errors.append(f"{prefix}: optics_codec.blur_sigma_px must be >= 0: {bs!r}")
            bd = opt.get("bit_depth")
            if isinstance(bd, bool) or bd not in VALID_BIT_DEPTHS:
                errors.append(f"{prefix}: optics_codec.bit_depth must be one of {sorted(VALID_BIT_DEPTHS)}, got {bd!r}")

        prov = rec.get("provenance")
        if isinstance(prov, Mapping):
            gtv = prov.get("gt_verification")
            if gtv not in VALID_GT_VERIFICATION:
                errors.append(f"{prefix}: invalid provenance.gt_verification {gtv!r}")

        raw_track = rec.get("track")
        if raw_track is not None and raw_track not in VALID_TRACKS:
            errors.append(f"{prefix}: invalid track {raw_track!r}")
            page_track = expected_track or "core_permissive"
        elif raw_track is not None and expected_track is not None and raw_track != expected_track:
            errors.append(
                f"{prefix}: page track {raw_track!r} does not match manifest track {expected_track!r}"
            )
            page_track = raw_track
        else:
            page_track = raw_track or expected_track or "core_permissive"

        page_eval_only = rec.get("eval_only")
        if page_track == "research_nc_eval_only":
            effective_eval_only = (
                page_eval_only if page_eval_only is not None else manifest_eval_only
            )
            if effective_eval_only is not True:
                errors.append(
                    f"{prefix}: governance: track 'research_nc_eval_only' requires eval_only=true "
                    "(strictly forbidden for training)"
                )
            allowed_licenses = base_permissive | NC_EVAL_ONLY_LICENSES
        else:
            allowed_licenses = base_permissive

        lic = rec.get("license")
        if isinstance(lic, Mapping):
            ml = lic.get("music_license")
            il = lic.get("image_license")
            if ml not in allowed_licenses:
                errors.append(
                    f"{prefix}: non-permissive music_license {ml!r} for track '{page_track}'"
                )
            if il not in allowed_licenses:
                errors.append(
                    f"{prefix}: non-permissive image_license {il!r} for track '{page_track}'"
                )
        elif "license" in rec:
            errors.append(f"{prefix}: license must be an object")

    # Governance rule: <= 15% max share per engraver on CDR (when >= 10 CDR pages)
    if len(cdr_pages) >= CDR_ENGRAVER_MIN_PAGES:
        engraver_counts: Counter[str] = Counter()
        for rec in cdr_pages:
            prov = rec.get("provenance")
            eng = None
            if isinstance(prov, Mapping):
                eng = prov.get("engraver_or_publisher")
            if not eng:
                eng = rec.get("engraver")
            if isinstance(eng, str) and eng.strip():
                engraver_counts[eng.strip()] += 1
        total_cdr = len(cdr_pages)
        for eng, count in sorted(engraver_counts.items()):
            share = count / total_cdr
            if share > CDR_ENGRAVER_MAX_SHARE + 1e-12:
                errors.append(
                    f"governance: CDR engraver '{eng}' share {share * 100:.1f}% "
                    f"({count}/{total_cdr}) exceeds {CDR_ENGRAVER_MAX_SHARE * 100:.0f}% cap"
                )

    return errors


def validate_manifest(
    manifest: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    allow_legacy_pd: bool = False,
) -> list[str]:
    """Validate a full manifest object (or bare list of page records)."""
    errors: list[str] = []
    expected_split: str | None = None
    expected_track: str | None = None
    manifest_eval_only: bool | None = None

    if isinstance(manifest, Mapping):
        for req in ("schema_version", "benchmark_version", "split", "pages"):
            if req not in manifest:
                errors.append(f"manifest: missing required field '{req}'")
        if manifest.get("schema_version") not in (None, "1.0.0"):
            errors.append(f"manifest: unsupported schema_version {manifest.get('schema_version')!r}")
        sp = manifest.get("split")
        if sp is not None:
            if not _is_valid_split(sp):
                errors.append(f"manifest: invalid split {sp!r}")
            else:
                expected_split = sp
                if sp.startswith("dev"):
                    canary = manifest.get("canary")
                    if not isinstance(canary, str) or not canary.strip():
                        errors.append(
                            f"governance: dev manifest ('{sp}') must include a non-empty 'canary' string"
                        )
        raw_track = manifest.get("track")
        if raw_track is not None:
            if raw_track not in VALID_TRACKS:
                errors.append(f"manifest: invalid track {raw_track!r}")
            else:
                expected_track = raw_track
        if "eval_only" in manifest:
            if not isinstance(manifest["eval_only"], bool):
                errors.append(f"manifest: eval_only must be boolean, got {manifest['eval_only']!r}")
            else:
                manifest_eval_only = manifest["eval_only"]
        if expected_track == "research_nc_eval_only" and manifest_eval_only is not True:
            errors.append(
                "governance: manifest with track 'research_nc_eval_only' must set eval_only=true "
                "(strictly forbidden for training)"
            )

        pages = manifest.get("pages")
        if not isinstance(pages, Sequence) or isinstance(pages, (str, bytes)):
            errors.append("manifest: 'pages' must be a list")
            return errors
        errors.extend(
            validate_page_records(
                pages,
                expected_split=expected_split,
                expected_track=expected_track,
                manifest_eval_only=manifest_eval_only,
                allow_legacy_pd=allow_legacy_pd,
            )
        )
        return errors

    if isinstance(manifest, Sequence) and not isinstance(manifest, (str, bytes)):
        return validate_page_records(manifest, allow_legacy_pd=allow_legacy_pd)

    return [f"manifest: expected object or list, got {type(manifest).__name__}"]


def validate_disjoint_splits(
    dev_manifest: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    test_manifest: Mapping[str, Any] | Sequence[Mapping[str, Any]],
) -> list[str]:
    """Enforce zero-leakage disjointness between dev and test manifests.

    Checks:
      - Zero page_id overlap across splits.
      - Zero work_id overlap across splits.
      - Zero writer_id / copyist overlap on HWM across splits.
    """
    errors: list[str] = []
    dev_pages = _extract_pages(dev_manifest)
    test_pages = _extract_pages(test_manifest)

    dev_page_ids = {str(p["page_id"]) for p in dev_pages if isinstance(p, Mapping) and p.get("page_id")}
    test_page_ids = {str(p["page_id"]) for p in test_pages if isinstance(p, Mapping) and p.get("page_id")}
    shared_pages = sorted(dev_page_ids & test_page_ids)
    if shared_pages:
        errors.append(f"governance: overlapping page_id(s) across splits: {', '.join(shared_pages)}")

    dev_works = {str(p["work_id"]) for p in dev_pages if isinstance(p, Mapping) and p.get("work_id")}
    test_works = {str(p["work_id"]) for p in test_pages if isinstance(p, Mapping) and p.get("work_id")}
    shared_works = sorted(dev_works & test_works)
    if shared_works:
        errors.append(f"governance: overlapping work_id(s) across splits: {', '.join(shared_works)}")

    def _hwm_writers(pages: Sequence[Mapping[str, Any]]) -> set[str]:
        writers: set[str] = set()
        for p in pages:
            if not isinstance(p, Mapping) or p.get("domain") != "HWM":
                continue
            prov = p.get("provenance")
            wid = prov.get("writer_id") if isinstance(prov, Mapping) else None
            if not wid:
                wid = p.get("writer_id")
            if isinstance(wid, str) and wid.strip():
                writers.add(wid.strip())
        return writers

    shared_writers = sorted(_hwm_writers(dev_pages) & _hwm_writers(test_pages))
    if shared_writers:
        errors.append(
            f"governance: overlapping HWM writer_id(s) across splits: {', '.join(shared_writers)}"
        )

    return errors


def validate_report(report: Mapping[str, Any]) -> list[str]:
    """Validate a canonical 4-layer report.json dictionary."""
    errors: list[str] = []
    if not isinstance(report, Mapping):
        return [f"report: expected object, got {type(report).__name__}"]

    required_top = (
        "schema_version",
        "benchmark_version",
        "split",
        "system_name",
        "layer1_headline",
        "layer2_matrix_and_deltas",
        "layer3_error_taxonomy",
        "layer4_forensic_pack",
    )
    for req in required_top:
        if req not in report:
            errors.append(f"report: missing required field '{req}'")

    if report.get("schema_version") not in (None, "1.0.0"):
        errors.append(f"report: invalid schema_version {report.get('schema_version')!r}")

    if "split" in report and not _is_valid_split(report["split"]):
        errors.append(f"report: invalid split {report['split']!r}")

    if "system_name" in report and (
        not isinstance(report["system_name"], str) or not report["system_name"].strip()
    ):
        errors.append("report: system_name must be a non-empty string")

    # Layer 1
    l1 = report.get("layer1_headline")
    if isinstance(l1, Mapping):
        for field_name in ("ns_score", "m_geom", "worst_stratum_q"):
            v = l1.get(field_name)
            if not _is_number(v) or not (0.0 <= float(v) <= 100.0):
                errors.append(f"layer1_headline.{field_name}: must be in [0, 100], got {v!r}")
        if l1.get("tier_weighted_ns") is not None:
            tw = l1["tier_weighted_ns"]
            if not _is_number(tw) or not (0.0 <= float(tw) <= 100.0):
                errors.append(f"layer1_headline.tier_weighted_ns: must be in [0, 100], got {tw!r}")
        if l1.get("ns_ci95") is not None:
            ci = l1["ns_ci95"]
            if (
                not isinstance(ci, Sequence)
                or isinstance(ci, (str, bytes))
                or len(ci) != 2
                or not _is_number(ci[0])
                or not _is_number(ci[1])
                or float(ci[0]) > float(ci[1])
            ):
                errors.append(f"layer1_headline.ns_ci95: must be [lo, hi] with lo <= hi, got {ci!r}")
        cfr_max = l1.get("cfr_max")
        if not _is_number(cfr_max) or not (0.0 <= float(cfr_max) <= 1.0):
            errors.append(f"layer1_headline.cfr_max: must be in [0, 1], got {cfr_max!r}")

        ss = l1.get("stratum_scores")
        if not isinstance(ss, Mapping):
            errors.append("layer1_headline.stratum_scores: must be an object")
        else:
            for d in DOMAINS:
                v = ss.get(d)
                if not _is_number(v) or not (0.0 <= float(v) <= 100.0):
                    errors.append(f"layer1_headline.stratum_scores.{d}: must be in [0, 100], got {v!r}")

        scfr = l1.get("stratum_cfr")
        if not isinstance(scfr, Mapping):
            errors.append("layer1_headline.stratum_cfr: must be an object")
        else:
            for d in DOMAINS:
                v = scfr.get(d)
                if not _is_number(v) or not (0.0 <= float(v) <= 1.0):
                    errors.append(f"layer1_headline.stratum_cfr.{d}: must be in [0, 1], got {v!r}")

        ax = l1.get("axes_macro")
        if not isinstance(ax, Mapping):
            errors.append("layer1_headline.axes_macro: must be an object")
        else:
            for k in ("s_note", "s_struct", "s_expr", "s_play"):
                v = ax.get(k)
                if not _is_number(v) or not (0.0 <= float(v) <= 1.0):
                    errors.append(f"layer1_headline.axes_macro.{k}: must be in [0, 1], got {v!r}")

        hu = l1.get("human_utility")
        if not isinstance(hu, Mapping):
            errors.append("layer1_headline.human_utility: must be an object")
        else:
            for k in ("median_wfpp", "p90_wfpp", "p99_wfpp"):
                v = hu.get(k)
                if not _is_number(v) or float(v) < 0.0:
                    errors.append(f"layer1_headline.human_utility.{k}: must be >= 0, got {v!r}")
            ntsr = hu.get("mean_ntsr")
            if not _is_number(ntsr) or float(ntsr) > 1.0:
                errors.append(f"layer1_headline.human_utility.mean_ntsr: must be <= 1, got {ntsr!r}")
            nupr = hu.get("nupr")
            if not _is_number(nupr) or not (0.0 <= float(nupr) <= 1.0):
                errors.append(f"layer1_headline.human_utility.nupr: must be in [0, 1], got {nupr!r}")

    # Layer 2
    l2 = report.get("layer2_matrix_and_deltas")
    if isinstance(l2, Mapping):
        if not isinstance(l2.get("tier_domain_matrix"), Mapping):
            errors.append("layer2_matrix_and_deltas.tier_domain_matrix: must be an object")
        if not isinstance(l2.get("paired_deltas"), Mapping):
            errors.append("layer2_matrix_and_deltas.paired_deltas: must be an object")

    # Layer 3
    l3 = report.get("layer3_error_taxonomy")
    if isinstance(l3, Mapping):
        for k, v in l3.items():
            if k not in ERROR_BUCKET_KEYS:
                errors.append(f"layer3_error_taxonomy: unknown bucket '{k}'")
            elif not _is_number(v) or not (0.0 <= float(v) <= 100.0):
                errors.append(f"layer3_error_taxonomy.{k}: must be in [0, 100], got {v!r}")

    # Layer 4
    l4 = report.get("layer4_forensic_pack")
    if isinstance(l4, Mapping):
        for roster in ("worst_pages", "best_pages"):
            items = l4.get(roster)
            if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
                errors.append(f"layer4_forensic_pack.{roster}: must be a list")
            else:
                for i, item in enumerate(items):
                    if not isinstance(item, Mapping):
                        errors.append(f"layer4_forensic_pack.{roster}[{i}]: must be an object")
                        continue
                    for req in ("page_id", "domain", "q_score", "wfpp", "catastrophic_failure"):
                        if req not in item:
                            errors.append(f"layer4_forensic_pack.{roster}[{i}]: missing '{req}'")
                    if item.get("domain") not in DOMAINS:
                        errors.append(
                            f"layer4_forensic_pack.{roster}[{i}]: invalid domain {item.get('domain')!r}"
                        )

    return errors
