"""OMR-NorthStar (OMR-NS) deterministic reference scoring package."""

from .scoring import (
    DOMAINS,
    TIERS,
    NorthStarResult,
    PageMetrics,
    PageScoreResult,
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

__all__ = [
    "DOMAINS",
    "TIERS",
    "NorthStarResult",
    "PageMetrics",
    "PageScoreResult",
    "build_report",
    "compute_catastrophic_penalties",
    "compute_effective_measure_cost",
    "compute_northstar_score",
    "compute_ntsr",
    "compute_nupr",
    "compute_page_quality",
    "compute_paired_deltas",
    "compute_prior_hallucination_rate",
    "compute_stratum_scores",
]

__version__ = "1.0.0"
