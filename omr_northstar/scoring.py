"""Deterministic reference scorer for OMR-NorthStar (OMR-NS).

Zero external dependencies (Python 3.10+ standard library only).
Implements the exact mathematical definitions from:
  - docs/01-architecture-and-governance.md
  - docs/04-metrics-and-human-utility.md
  - docs/05-adversarial-stress-tests.md
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

DOMAINS: tuple[str, ...] = ("CDR", "ALS", "RWP", "HWM")
TIERS: tuple[str, ...] = ("T0", "T1", "T2", "T3", "T4", "T5")
TIER_WEIGHTS: dict[str, float] = {
    "T0": 0.05,
    "T1": 0.15,
    "T2": 0.25,
    "T3": 0.25,
    "T4": 0.18,
    "T5": 0.12,
}

ERROR_BUCKET_KEYS: tuple[str, ...] = (
    "layout_segmentation_pct",
    "global_context_drift_pct",
    "pitch_accidental_pct",
    "rhythm_meter_pct",
    "polyphony_routing_pct",
    "engraving_expressive_pct",
    "hallucination_truncation_pct",
    "annotation_contamination_pct",
)


@dataclass(frozen=True)
class PageMetrics:
    """Per-page evaluation metrics extracted from CER alignment."""

    page_id: str
    work_id: str
    domain: str
    tier: str
    s_note: float
    s_struct: float
    s_expr: float
    s_play: float
    wfpp: float
    transcribe_cost: float
    parseable: bool = True
    n_gt: int = 1
    n_pred: int = 1
    k_dropped_systems: int = 0
    has_repetition_loop: bool = False
    bsvr: float = 1.0
    m_gt: int = 1
    m_pred: int = 1
    paired_anchor_id: str | None = None
    engraver: str | None = None
    error_buckets: Mapping[str, float] | None = None
    latency_s: float | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PageMetrics":
        return cls(
            page_id=str(data["page_id"]),
            work_id=str(data.get("work_id", data["page_id"])),
            domain=str(data["domain"]),
            tier=str(data.get("tier", "T1")),
            s_note=float(data["s_note"]),
            s_struct=float(data["s_struct"]),
            s_expr=float(data["s_expr"]),
            s_play=float(data.get("s_play", data["s_note"])),
            wfpp=float(data["wfpp"]),
            transcribe_cost=float(data["transcribe_cost"]),
            parseable=bool(data.get("parseable", True)),
            n_gt=int(data.get("n_gt", 1)),
            n_pred=int(data.get("n_pred", 1)),
            k_dropped_systems=int(data.get("k_dropped_systems", 0)),
            has_repetition_loop=bool(data.get("has_repetition_loop", False)),
            bsvr=float(data.get("bsvr", 1.0)),
            m_gt=int(data.get("m_gt", 1)),
            m_pred=int(data.get("m_pred", 1)),
            paired_anchor_id=data.get("paired_anchor_id"),
            engraver=data.get("engraver"),
            error_buckets=data.get("error_buckets"),
            latency_s=(
                float(data["latency_s"])
                if data.get("latency_s") is not None
                else None
            ),
        )


@dataclass(frozen=True)
class PageScoreResult:
    """Computed quality score and catastrophic failure flags for a single page."""

    page: PageMetrics
    q_raw: float
    q_score: float
    ntsr: float
    effective_wfpp: float
    catastrophic_failure: bool
    cf_reasons: tuple[str, ...] = field(default_factory=tuple)
    penalty_multiplier: float = 1.0


@dataclass(frozen=True)
class NorthStarResult:
    """Composite North Star (NS) headline summary."""

    ns_score: float
    m_geom: float
    worst_stratum_q: float
    cfr_max: float


def compute_effective_measure_cost(
    edit_cost: float,
    transcribe_cost: float,
    clear_cost: float = 1.5,
) -> float:
    """Rational musician re-entry cap for a single measure.

    Cost_eff(m) = min(EditCost(m_hat -> m), c_clear + TranscribeCost(m))
    """
    if edit_cost < 0 or transcribe_cost < 0 or clear_cost < 0:
        raise ValueError("Costs must be non-negative.")
    return min(edit_cost, clear_cost + transcribe_cost)


def compute_ntsr(wfpp: float, transcribe_cost: float) -> float:
    """Net Time Saved Ratio: NTSR(P) = 1 - wFPP(P) / TranscribeCost(P).

    Handles non-music negative-control pages (transcribe_cost == 0):
      - wfpp == 0 -> 1.0 (zero false-positive edits)
      - wfpp > 0  -> -wfpp (strictly negative utility)
    """
    if wfpp < 0:
        raise ValueError("wfpp must be non-negative.")
    if transcribe_cost < 0:
        raise ValueError("transcribe_cost must be non-negative.")
    if transcribe_cost == 0.0:
        return 1.0 if wfpp == 0.0 else -float(wfpp)
    return 1.0 - (wfpp / transcribe_cost)


def compute_nupr(ntsr_values: Sequence[float]) -> float:
    """Negative-Utility Page Rate: fraction of pages with NTSR <= 0."""
    if not ntsr_values:
        return 0.0
    neg = sum(1 for v in ntsr_values if v <= 0.0)
    return neg / len(ntsr_values)


def compute_catastrophic_penalties(
    page: PageMetrics,
) -> tuple[bool, tuple[str, ...], float, float]:
    """Evaluate all 5 Catastrophic Failure (CF) conditions and multiplicative penalties.

    Returns:
        (is_cf, reasons, penalty_multiplier, effective_wfpp)
    """
    if page.k_dropped_systems < 0:
        raise ValueError("k_dropped_systems must be non-negative.")
    if page.n_gt < 0 or page.n_pred < 0 or page.m_gt < 0 or page.m_pred < 0:
        raise ValueError("Note and measure counts must be non-negative.")

    reasons: list[str] = []
    mult = 1.0
    effective_wfpp = page.wfpp

    # 1. Unparseable / schema-invalid output
    if not page.parseable:
        reasons.append("unparseable")
        mult = 0.0
        effective_wfpp = max(1.2 * page.transcribe_cost, 1.5 if page.transcribe_cost == 0.0 else 0.0)

    # 2. Skipped / empty page (only when GT actually contains music)
    if page.n_gt > 0 and page.n_pred == 0:
        reasons.append("empty_output")
        mult = 0.0

    # 3. Dropped system(s)
    if page.k_dropped_systems > 0:
        reasons.append("dropped_system")
        mult *= 0.85 ** page.k_dropped_systems

    # 4. Hallucination / repetition loop (also catches music hallucinated on non-music n_gt == 0 pages)
    over_predicted = (
        (page.n_gt > 0 and page.n_pred > 1.5 * page.n_gt)
        or (page.n_gt == 0 and page.n_pred > 0)
    )
    if over_predicted or page.has_repetition_loop:
        reasons.append("hallucination_loop")
        if page.n_gt > 0:
            loop_pen = min(1.0, (1.5 * page.n_gt) / max(page.n_pred, 1))
            mult *= loop_pen
        elif page.n_pred > 0:
            mult = 0.0

    # 5. Structural collapse (safe when m_gt == 0 and m_pred == 0 on non-music pages)
    measure_ratio_bad = (
        (page.m_gt > 0 and abs(page.m_pred - page.m_gt) / page.m_gt > 0.25)
        or (page.m_gt == 0 and page.m_pred > 0)
    )
    is_clean_non_music = (
        page.n_gt == 0 and page.n_pred == 0 and page.m_gt == 0 and page.m_pred == 0
    )
    if (not is_clean_non_music and page.bsvr < 0.50) or measure_ratio_bad:
        reasons.append("structural_collapse")

    return (len(reasons) > 0, tuple(reasons), mult, effective_wfpp)


def compute_page_quality(page: PageMetrics) -> PageScoreResult:
    """Compute per-page quality Q(P) in [0, 100] with catastrophic penalties."""
    is_cf, reasons, mult, effective_wfpp = compute_catastrophic_penalties(page)
    ntsr = compute_ntsr(effective_wfpp, page.transcribe_cost)
    q_raw = 100.0 * (
        0.40 * page.s_note
        + 0.25 * page.s_struct
        + 0.20 * page.s_expr
        + 0.15 * max(0.0, ntsr)
    )
    q_raw = max(0.0, min(100.0, q_raw))
    q_score = max(0.0, min(100.0, q_raw * mult))
    return PageScoreResult(
        page=page,
        q_raw=q_raw,
        q_score=q_score,
        ntsr=ntsr,
        effective_wfpp=effective_wfpp,
        catastrophic_failure=is_cf,
        cf_reasons=reasons,
        penalty_multiplier=mult,
    )


def compute_stratum_scores(
    page_results: Sequence[PageScoreResult],
) -> tuple[dict[str, float], dict[str, float]]:
    """Compute piece-level macro-average Q_s and Catastrophic Failure Rate CFR_s per domain.

    Per anti-gaming rule #3, any missing domain stratum scores Q_s = 0.0 (and CFR_s = 1.0).
    """
    by_domain_work: dict[str, dict[str, list[PageScoreResult]]] = {
        d: defaultdict(list) for d in DOMAINS
    }
    by_domain_pages: dict[str, list[PageScoreResult]] = {d: [] for d in DOMAINS}

    for res in page_results:
        d = res.page.domain
        if d not in by_domain_pages:
            raise ValueError(f"Unknown domain {d!r}; expected one of {DOMAINS}.")
        by_domain_pages[d].append(res)
        by_domain_work[d][res.page.work_id].append(res)

    stratum_scores: dict[str, float] = {}
    stratum_cfr: dict[str, float] = {}

    for d in DOMAINS:
        pages = by_domain_pages[d]
        if not pages:
            stratum_scores[d] = 0.0
            stratum_cfr[d] = 1.0
            continue
        work_means = [
            sum(r.q_score for r in work_pages) / len(work_pages)
            for work_pages in by_domain_work[d].values()
        ]
        stratum_scores[d] = sum(work_means) / len(work_means)
        stratum_cfr[d] = sum(1 for r in pages if r.catastrophic_failure) / len(pages)

    return stratum_scores, stratum_cfr


def compute_tier_scores(
    page_results: Sequence[PageScoreResult],
) -> tuple[dict[str, float | None], dict[str, float | None]]:
    """Compute piece-level macro-average Q_t and CFR_t per complexity tier T0..T5."""
    by_tier_work: dict[str, dict[str, list[PageScoreResult]]] = {
        t: defaultdict(list) for t in TIERS
    }
    by_tier_pages: dict[str, list[PageScoreResult]] = {t: [] for t in TIERS}

    for res in page_results:
        t = res.page.tier
        if t not in by_tier_pages:
            raise ValueError(f"Unknown tier {t!r}; expected one of {TIERS}.")
        by_tier_pages[t].append(res)
        by_tier_work[t][res.page.work_id].append(res)

    tier_scores: dict[str, float | None] = {}
    tier_cfr: dict[str, float | None] = {}
    for t in TIERS:
        pages = by_tier_pages[t]
        if not pages:
            tier_scores[t] = None
            tier_cfr[t] = None
            continue
        work_means = [
            sum(r.q_score for r in work_pages) / len(work_pages)
            for work_pages in by_tier_work[t].values()
        ]
        tier_scores[t] = sum(work_means) / len(work_means)
        tier_cfr[t] = sum(1 for r in pages if r.catastrophic_failure) / len(pages)

    return tier_scores, tier_cfr


def compute_tier_weighted_ns(
    page_results: Sequence[PageScoreResult],
    tier_weights: Mapping[str, float] = TIER_WEIGHTS,
) -> float:
    """Compute 6-tier weighted North Star breakdown renormalized over active tiers.

    Uses weights T0: 0.05, T1: 0.15, T2: 0.25, T3: 0.25, T4: 0.18, T5: 0.12
    renormalized over tiers present in `page_results`, damped by
    (1 - 0.50 * max_active_tier CFR_t).
    """
    tier_scores, tier_cfr = compute_tier_scores(page_results)
    active = [t for t in TIERS if tier_scores[t] is not None]
    if not active:
        return 0.0
    total_w = sum(float(tier_weights[t]) for t in active)
    if total_w <= 0.0:
        return 0.0
    weighted_q = sum(
        (float(tier_weights[t]) / total_w) * float(tier_scores[t])  # type: ignore[arg-type]
        for t in active
    )
    cfr_max = max(float(tier_cfr[t]) for t in active)  # type: ignore[arg-type]
    cfr_max = max(0.0, min(1.0, cfr_max))
    return weighted_q * (1.0 - 0.50 * cfr_max)


def compute_northstar_score(
    stratum_scores: Mapping[str, float],
    stratum_cfr: Mapping[str, float],
) -> NorthStarResult:
    """Compute the composite North Star Score (NS):

    M_geom = (prod_{s in D} max(Q_s, 0.1))^(1/4)
    NS = (0.75 * M_geom + 0.25 * min_s Q_s) * (1 - 0.50 * max_s CFR_s)
    """
    q_vals = [float(stratum_scores.get(d, 0.0)) for d in DOMAINS]
    cfr_vals = [float(stratum_cfr.get(d, 1.0)) for d in DOMAINS]

    prod = 1.0
    for q in q_vals:
        prod *= max(q, 0.1)
    m_geom = prod ** (1.0 / len(DOMAINS))
    worst_q = min(q_vals)
    cfr_max = max(0.0, min(1.0, max(cfr_vals)))
    ns_score = (0.75 * m_geom + 0.25 * worst_q) * (1.0 - 0.50 * cfr_max)

    return NorthStarResult(
        ns_score=ns_score,
        m_geom=m_geom,
        worst_stratum_q=worst_q,
        cfr_max=cfr_max,
    )


def compute_ns_bootstrap_ci(
    page_results: Sequence[PageScoreResult],
    *,
    n_boot: int = 1000,
    seed: int = 42,
    ci: float = 0.95,
) -> tuple[float, float] | None:
    """Deterministic work-clustered bootstrap confidence interval for NS.

    Resamples whole `work_id` clusters with replacement within each domain stratum
    using a seeded `random.Random(seed)` PRNG.
    """
    if not page_results:
        return None
    if n_boot <= 0:
        raise ValueError("n_boot must be strictly positive.")
    if not (0.0 < ci < 1.0):
        raise ValueError("ci must be strictly between 0 and 1.")

    by_domain_work: dict[str, dict[str, list[PageScoreResult]]] = {
        d: defaultdict(list) for d in DOMAINS
    }
    for res in page_results:
        d = res.page.domain
        if d not in by_domain_work:
            raise ValueError(f"Unknown domain {d!r}; expected one of {DOMAINS}.")
        by_domain_work[d][res.page.work_id].append(res)

    # Precompute deterministic sorted work summaries per domain: (mean_q, n_pages, n_cf)
    domain_clusters: dict[str, list[tuple[float, int, int]]] = {}
    for d in DOMAINS:
        clusters: list[tuple[float, int, int]] = []
        for wid in sorted(by_domain_work[d].keys()):
            w_pages = by_domain_work[d][wid]
            mean_q = sum(r.q_score for r in w_pages) / len(w_pages)
            n_cf = sum(1 for r in w_pages if r.catastrophic_failure)
            clusters.append((mean_q, len(w_pages), n_cf))
        domain_clusters[d] = clusters

    rng = random.Random(seed)
    boot_ns: list[float] = []
    for _ in range(n_boot):
        s_scores: dict[str, float] = {}
        s_cfr: dict[str, float] = {}
        for d in DOMAINS:
            clusters = domain_clusters[d]
            k = len(clusters)
            if k == 0:
                s_scores[d] = 0.0
                s_cfr[d] = 1.0
                continue
            q_sum = 0.0
            total_pages = 0
            total_cf = 0
            for _ in range(k):
                mq, npg, ncf = clusters[rng.randrange(k)]
                q_sum += mq
                total_pages += npg
                total_cf += ncf
            s_scores[d] = q_sum / k
            s_cfr[d] = total_cf / total_pages
        boot_ns.append(compute_northstar_score(s_scores, s_cfr).ns_score)

    boot_ns.sort()
    alpha = (1.0 - ci) / 2.0
    lo = _percentile(boot_ns, alpha * 100.0)
    hi = _percentile(boot_ns, (1.0 - alpha) * 100.0)
    return (lo, hi)


def compute_paired_deltas(
    page_results: Sequence[PageScoreResult],
) -> dict[str, float | None]:
    """Compute confounder-free paired robustness deltas from shared paired_anchor_id.

    - delta_engraver: mean (max - min) Q spread across CDR engravers on the same anchor.
    - delta_photo: mean (Q(RWP) - Q(CDR or ALS)) on the same anchor.
    - delta_handwriting: mean (Q(HWM) - Q(CDR)) on the same anchor.
    """
    anchors: dict[str, list[PageScoreResult]] = defaultdict(list)
    for res in page_results:
        if res.page.paired_anchor_id:
            anchors[res.page.paired_anchor_id].append(res)

    engraver_spreads: list[float] = []
    photo_deltas: list[float] = []
    hw_deltas: list[float] = []

    for items in anchors.values():
        cdr_items = [r for r in items if r.page.domain == "CDR"]
        als_items = [r for r in items if r.page.domain == "ALS"]
        rwp_items = [r for r in items if r.page.domain == "RWP"]
        hwm_items = [r for r in items if r.page.domain == "HWM"]

        # Engraver spread within CDR (grouped per engraver to avoid duplicate-render skew)
        by_engraver: dict[str, list[float]] = defaultdict(list)
        for r in cdr_items:
            if r.page.engraver:
                by_engraver[r.page.engraver].append(r.q_score)
        if len(by_engraver) >= 2:
            eng_means = [sum(qs) / len(qs) for qs in by_engraver.values()]
            engraver_spreads.append(max(eng_means) - min(eng_means))

        # Photo delta: RWP - baseline (CDR if present else ALS)
        baseline_for_photo = cdr_items if cdr_items else als_items
        if rwp_items and baseline_for_photo:
            rwp_mean = sum(r.q_score for r in rwp_items) / len(rwp_items)
            base_mean = sum(r.q_score for r in baseline_for_photo) / len(
                baseline_for_photo
            )
            photo_deltas.append(rwp_mean - base_mean)

        # Handwriting delta: HWM - CDR
        if hwm_items and cdr_items:
            hwm_mean = sum(r.q_score for r in hwm_items) / len(hwm_items)
            cdr_mean = sum(r.q_score for r in cdr_items) / len(cdr_items)
            hw_deltas.append(hwm_mean - cdr_mean)

    def _mean_or_none(vals: list[float]) -> float | None:
        return (sum(vals) / len(vals)) if vals else None

    return {
        "delta_engraver": _mean_or_none(engraver_spreads),
        "delta_photo": _mean_or_none(photo_deltas),
        "delta_handwriting": _mean_or_none(hw_deltas),
    }


def compute_prior_hallucination_rate(
    mutated_sites_total: int,
    predicted_as_original_count: int,
) -> float:
    """Counterfactual Classics Prior-Hallucination Rate (PHR).

    PHR = (# mutated sites predicted as the original canonical note) / (# mutated sites)
    """
    if mutated_sites_total <= 0:
        raise ValueError("mutated_sites_total must be strictly positive.")
    if predicted_as_original_count < 0 or predicted_as_original_count > mutated_sites_total:
        raise ValueError("predicted_as_original_count must be in [0, mutated_sites_total].")
    return predicted_as_original_count / mutated_sites_total


def _percentile(sorted_vals: Sequence[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)


def build_report(
    pages: Sequence[PageMetrics],
    *,
    system_name: str = "unnamed-system",
    split: str = "dev-v1",
    benchmark_version: str = "1.0.0",
    adversarial: Mapping[str, Any] | None = None,
    telemetry: Mapping[str, Any] | None = None,
    n_boot: int = 1000,
    bootstrap_seed: int = 42,
) -> dict[str, Any]:
    """Build a canonical 4-layer report.json dictionary."""
    results = [compute_page_quality(p) for p in pages]
    stratum_scores, stratum_cfr = compute_stratum_scores(results)
    ns = compute_northstar_score(stratum_scores, stratum_cfr)
    ns_ci = compute_ns_bootstrap_ci(results, n_boot=n_boot, seed=bootstrap_seed)
    tier_weighted_ns = compute_tier_weighted_ns(results)
    paired_deltas = compute_paired_deltas(results)

    n = len(results)
    if n > 0:
        s_note_m = sum(r.page.s_note for r in results) / n
        s_struct_m = sum(r.page.s_struct for r in results) / n
        s_expr_m = sum(r.page.s_expr for r in results) / n
        s_play_m = sum(r.page.s_play for r in results) / n
        wfpp_sorted = sorted(r.effective_wfpp for r in results)
        ntsr_vals = [r.ntsr for r in results]
        median_wfpp = _percentile(wfpp_sorted, 50.0)
        p90_wfpp = _percentile(wfpp_sorted, 90.0)
        p99_wfpp = _percentile(wfpp_sorted, 99.0)
        mean_ntsr = sum(ntsr_vals) / n
        nupr = compute_nupr(ntsr_vals)
    else:
        s_note_m = s_struct_m = s_expr_m = s_play_m = 0.0
        median_wfpp = p90_wfpp = p99_wfpp = mean_ntsr = nupr = 0.0

    # Layer 2: 6x4 matrix
    matrix: dict[str, dict[str, dict[str, Any]]] = {}
    for t in TIERS:
        matrix[t] = {}
        for d in DOMAINS:
            cell = [r for r in results if r.page.tier == t and r.page.domain == d]
            if cell:
                cell_wfpp = sorted(r.effective_wfpp for r in cell)
                matrix[t][d] = {
                    "mean_q": round(sum(r.q_score for r in cell) / len(cell), 4),
                    "median_wfpp": round(_percentile(cell_wfpp, 50.0), 4),
                    "count": len(cell),
                }
            else:
                matrix[t][d] = {"mean_q": None, "median_wfpp": None, "count": 0}

    # Layer 3: 8-bucket error taxonomy (mean across pages that report buckets)
    bucket_sums: dict[str, float] = {k: 0.0 for k in ERROR_BUCKET_KEYS}
    bucket_counts = 0
    for r in results:
        if r.page.error_buckets:
            bucket_counts += 1
            for k in ERROR_BUCKET_KEYS:
                bucket_sums[k] += float(r.page.error_buckets.get(k, 0.0))
    layer3 = {
        k: round(bucket_sums[k] / bucket_counts, 4) if bucket_counts > 0 else 0.0
        for k in ERROR_BUCKET_KEYS
    }

    # Layer 4: Worst-10 and Best-10 forensic roster
    sorted_by_q = sorted(results, key=lambda r: (r.q_score, r.page.page_id))
    worst_pages = [
        {
            "page_id": r.page.page_id,
            "domain": r.page.domain,
            "q_score": round(r.q_score, 4),
            "wfpp": round(r.effective_wfpp, 4),
            "catastrophic_failure": r.catastrophic_failure,
        }
        for r in sorted_by_q[:10]
    ]
    best_pages = [
        {
            "page_id": r.page.page_id,
            "domain": r.page.domain,
            "q_score": round(r.q_score, 4),
            "wfpp": round(r.effective_wfpp, 4),
            "catastrophic_failure": r.catastrophic_failure,
        }
        for r in reversed(sorted_by_q[-10:])
    ]

    latencies = sorted(r.page.latency_s for r in results if r.page.latency_s is not None)
    eff = {
        "median_latency_s": round(_percentile(latencies, 50.0), 4) if latencies else None,
        "p95_latency_s": round(_percentile(latencies, 95.0), 4) if latencies else None,
        "peak_memory_gb": (telemetry or {}).get("peak_memory_gb"),
        "parameter_count": (telemetry or {}).get("parameter_count"),
    }

    return {
        "schema_version": "1.0.0",
        "benchmark_version": benchmark_version,
        "split": split,
        "system_name": system_name,
        "layer1_headline": {
            "ns_score": round(ns.ns_score, 4),
            "ns_ci95": (
                [round(ns_ci[0], 4), round(ns_ci[1], 4)]
                if ns_ci is not None
                else None
            ),
            "tier_weighted_ns": round(tier_weighted_ns, 4),
            "m_geom": round(ns.m_geom, 4),
            "worst_stratum_q": round(ns.worst_stratum_q, 4),
            "cfr_max": round(ns.cfr_max, 4),
            "stratum_scores": {d: round(stratum_scores[d], 4) for d in DOMAINS},
            "stratum_cfr": {d: round(stratum_cfr[d], 4) for d in DOMAINS},
            "axes_macro": {
                "s_note": round(s_note_m, 4),
                "s_struct": round(s_struct_m, 4),
                "s_expr": round(s_expr_m, 4),
                "s_play": round(s_play_m, 4),
            },
            "human_utility": {
                "median_wfpp": round(median_wfpp, 4),
                "p90_wfpp": round(p90_wfpp, 4),
                "p99_wfpp": round(p99_wfpp, 4),
                "mean_ntsr": round(mean_ntsr, 4),
                "nupr": round(nupr, 4),
            },
        },
        "layer2_matrix_and_deltas": {
            "tier_domain_matrix": matrix,
            "paired_deltas": {
                k: (round(v, 4) if v is not None else None)
                for k, v in paired_deltas.items()
            },
            "adversarial_scorecard": dict(adversarial) if adversarial else {},
        },
        "layer3_error_taxonomy": layer3,
        "layer4_forensic_pack": {
            "worst_pages": worst_pages,
            "best_pages": best_pages,
            "efficiency_telemetry": eff,
        },
    }
