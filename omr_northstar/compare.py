"""Layer 4 pairwise regression diffing (`compare(A, B)`) for OMR-NorthStar.

Zero external dependencies (Python 3.10+ standard library only).
Computes headline NS delta, domain-by-domain and tier-by-tier deltas,
paired two-sided sign-flip permutation p-values when per-page scores are
available, and an automatic `regressions` list flagging any statistically
significant single-domain or single-tier drop even when overall NS rises.
"""

from __future__ import annotations

import random
from typing import Any, Mapping, Sequence

from .scoring import DOMAINS, TIERS


def _extract_per_page_scores(report: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Extract a page_id -> page_score_dict mapping if full per-page scores are present."""
    candidates = (
        report.get("per_page_scores"),
        (report.get("layer4_forensic_pack") or {}).get("per_page_scores"),
        report.get("pages"),
    )
    for raw in candidates:
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)) and raw:
            by_id: dict[str, dict[str, Any]] = {}
            for item in raw:
                if isinstance(item, Mapping) and "page_id" in item and "q_score" in item:
                    by_id[str(item["page_id"])] = dict(item)
            if by_id:
                return by_id
    return {}


def _paired_permutation_pvalue(
    diffs: Sequence[float],
    *,
    n_perm: int = 1000,
    seed: int = 42,
) -> float | None:
    """Two-sided paired sign-flip permutation p-value for mean(diffs) != 0."""
    n = len(diffs)
    if n == 0:
        return None
    if n_perm <= 0:
        raise ValueError("n_perm must be strictly positive.")
    obs = abs(sum(diffs) / n)
    if obs == 0.0:
        return 1.0

    rng = random.Random(seed)
    ge_count = 0
    eps = 1e-12
    for _ in range(n_perm):
        s = 0.0
        for d in diffs:
            s += d if rng.getrandbits(1) else -d
        if abs(s / n) >= obs - eps:
            ge_count += 1
    return (ge_count + 1) / (n_perm + 1)


def _tier_weighted_mean_from_matrix(
    report: Mapping[str, Any],
    tier: str,
) -> float | None:
    """Compute count-weighted mean Q for a tier from layer2 tier_domain_matrix."""
    matrix = (report.get("layer2_matrix_and_deltas") or {}).get("tier_domain_matrix") or {}
    row = matrix.get(tier)
    if not isinstance(row, Mapping):
        return None
    total_count = 0
    weighted_sum = 0.0
    for d in DOMAINS:
        cell = row.get(d)
        if not isinstance(cell, Mapping):
            continue
        cnt = int(cell.get("count", 0) or 0)
        mq = cell.get("mean_q")
        if cnt > 0 and mq is not None:
            weighted_sum += float(mq) * cnt
            total_count += cnt
    if total_count == 0:
        return None
    return weighted_sum / total_count


def compare_reports(
    report_a: Mapping[str, Any],
    report_b: Mapping[str, Any],
    *,
    n_perm: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
    unpaired_drop_threshold: float | None = 2.0,
) -> dict[str, Any]:
    """Compare baseline report A against candidate report B (delta = B - A).

    Automatically populates `regressions` for any statistically significant
    single-domain or single-tier drop (`delta < 0` and `p < alpha`), even when
    overall `NS` improves. When per-page scores are absent, falls back to
    flagging drops exceeding `unpaired_drop_threshold` (if not None).
    """
    l1_a = report_a["layer1_headline"]
    l1_b = report_b["layer1_headline"]

    ns_a = float(l1_a["ns_score"])
    ns_b = float(l1_b["ns_score"])
    ns_delta = round(ns_b - ns_a, 4)

    pages_a = _extract_per_page_scores(report_a)
    pages_b = _extract_per_page_scores(report_b)
    common_ids = sorted(set(pages_a.keys()) & set(pages_b.keys()))
    has_paired = len(common_ids) > 0

    # Overall paired page Q permutation p-value
    all_diffs = [
        float(pages_b[pid]["q_score"]) - float(pages_a[pid]["q_score"])
        for pid in common_ids
    ]
    overall_p = (
        _paired_permutation_pvalue(all_diffs, n_perm=n_perm, seed=seed)
        if has_paired
        else None
    )

    headline: dict[str, Any] = {
        "ns_a": round(ns_a, 4),
        "ns_b": round(ns_b, 4),
        "ns_delta": ns_delta,
        "m_geom_delta": round(float(l1_b["m_geom"]) - float(l1_a["m_geom"]), 4),
        "worst_stratum_q_delta": round(
            float(l1_b["worst_stratum_q"]) - float(l1_a["worst_stratum_q"]), 4
        ),
        "cfr_max_delta": round(float(l1_b["cfr_max"]) - float(l1_a["cfr_max"]), 4),
        "p_value": round(overall_p, 6) if overall_p is not None else None,
        "significant": bool(overall_p is not None and overall_p < alpha),
        "n_paired_pages": len(common_ids),
    }
    if l1_a.get("tier_weighted_ns") is not None and l1_b.get("tier_weighted_ns") is not None:
        headline["tier_weighted_ns_delta"] = round(
            float(l1_b["tier_weighted_ns"]) - float(l1_a["tier_weighted_ns"]), 4
        )

    regressions: list[dict[str, Any]] = []

    # Domain-by-domain comparison
    domain_deltas: dict[str, dict[str, Any]] = {}
    for idx, d in enumerate(DOMAINS):
        qa = float(l1_a["stratum_scores"][d])
        qb = float(l1_b["stratum_scores"][d])
        delta = round(qb - qa, 4)
        cfra = float(l1_a["stratum_cfr"][d])
        cfrb = float(l1_b["stratum_cfr"][d])
        cfr_delta = round(cfrb - cfra, 4)

        d_diffs = [
            float(pages_b[pid]["q_score"]) - float(pages_a[pid]["q_score"])
            for pid in common_ids
            if pages_a[pid].get("domain") == d and pages_b[pid].get("domain") == d
        ]
        p_val = (
            _paired_permutation_pvalue(d_diffs, n_perm=n_perm, seed=seed + 100 + idx)
            if d_diffs
            else None
        )
        sig = bool(p_val is not None and p_val < alpha)

        domain_deltas[d] = {
            "score_a": round(qa, 4),
            "score_b": round(qb, 4),
            "delta": delta,
            "cfr_a": round(cfra, 4),
            "cfr_b": round(cfrb, 4),
            "cfr_delta": cfr_delta,
            "p_value": round(p_val, 6) if p_val is not None else None,
            "significant": sig,
            "n_paired": len(d_diffs),
        }

        if p_val is not None:
            if delta < 0.0 and sig:
                regressions.append(
                    {
                        "scope": "domain",
                        "key": d,
                        "score_a": round(qa, 4),
                        "score_b": round(qb, 4),
                        "delta": delta,
                        "p_value": round(p_val, 6),
                        "significant": True,
                        "reason": f"Statistically significant domain drop on {d} ({delta:+.2f}, p={p_val:.4f} < {alpha})",
                    }
                )
        elif unpaired_drop_threshold is not None and delta <= -unpaired_drop_threshold:
            regressions.append(
                {
                    "scope": "domain",
                    "key": d,
                    "score_a": round(qa, 4),
                    "score_b": round(qb, 4),
                    "delta": delta,
                    "p_value": None,
                    "significant": False,
                    "reason": f"Domain drop on {d} ({delta:+.2f}) exceeds unpaired threshold (-{unpaired_drop_threshold:.2f})",
                }
            )

    # Tier-by-tier comparison
    tier_deltas: dict[str, dict[str, Any]] = {}
    for idx, t in enumerate(TIERS):
        t_diffs = [
            float(pages_b[pid]["q_score"]) - float(pages_a[pid]["q_score"])
            for pid in common_ids
            if pages_a[pid].get("tier") == t and pages_b[pid].get("tier") == t
        ]
        qa = _tier_weighted_mean_from_matrix(report_a, t)
        qb = _tier_weighted_mean_from_matrix(report_b, t)
        if qa is None and t_diffs:
            qa = sum(float(pages_a[pid]["q_score"]) for pid in common_ids if pages_a[pid].get("tier") == t) / len(t_diffs)
        if qb is None and t_diffs:
            qb = sum(float(pages_b[pid]["q_score"]) for pid in common_ids if pages_b[pid].get("tier") == t) / len(t_diffs)

        delta = round(qb - qa, 4) if (qa is not None and qb is not None) else None
        p_val = (
            _paired_permutation_pvalue(t_diffs, n_perm=n_perm, seed=seed + 200 + idx)
            if t_diffs
            else None
        )
        sig = bool(p_val is not None and p_val < alpha)

        tier_deltas[t] = {
            "score_a": round(qa, 4) if qa is not None else None,
            "score_b": round(qb, 4) if qb is not None else None,
            "delta": delta,
            "p_value": round(p_val, 6) if p_val is not None else None,
            "significant": sig,
            "n_paired": len(t_diffs),
        }

        if delta is not None:
            if p_val is not None:
                if delta < 0.0 and sig:
                    regressions.append(
                        {
                            "scope": "tier",
                            "key": t,
                            "score_a": round(qa, 4) if qa is not None else None,
                            "score_b": round(qb, 4) if qb is not None else None,
                            "delta": delta,
                            "p_value": round(p_val, 6),
                            "significant": True,
                            "reason": f"Statistically significant tier drop on {t} ({delta:+.2f}, p={p_val:.4f} < {alpha})",
                        }
                    )
            elif unpaired_drop_threshold is not None and delta <= -unpaired_drop_threshold:
                regressions.append(
                    {
                        "scope": "tier",
                        "key": t,
                        "score_a": round(qa, 4) if qa is not None else None,
                        "score_b": round(qb, 4) if qb is not None else None,
                        "delta": delta,
                        "p_value": None,
                        "significant": False,
                        "reason": f"Tier drop on {t} ({delta:+.2f}) exceeds unpaired threshold (-{unpaired_drop_threshold:.2f})",
                    }
                )

    return {
        "system_a": str(report_a.get("system_name", "system-a")),
        "system_b": str(report_b.get("system_name", "system-b")),
        "split_a": str(report_a.get("split", "")),
        "split_b": str(report_b.get("split", "")),
        "alpha": alpha,
        "n_perm": n_perm,
        "seed": seed,
        "headline": headline,
        "domain_deltas": domain_deltas,
        "tier_deltas": tier_deltas,
        "regressions": regressions,
        "passed": len(regressions) == 0,
    }


def format_comparison(diff: Mapping[str, Any]) -> str:
    """Render a readable terminal regression diff table."""
    h = diff["headline"]
    p_str = f"p={h['p_value']:.4f}" if h["p_value"] is not None else "p=N/A"
    lines = [
        f"=== OMR-NorthStar Comparison: {diff['system_a']} -> {diff['system_b']} ===",
        f"Headline NS : {h['ns_a']:.2f} -> {h['ns_b']:.2f} ({h['ns_delta']:+.2f}, {p_str}, paired={h['n_paired_pages']})",
        f"M_geom Δ    : {h['m_geom_delta']:+.2f} | Worst Q Δ: {h['worst_stratum_q_delta']:+.2f} | CFR_max Δ: {h['cfr_max_delta']:+.4f}",
        "",
        "Domain Deltas:",
        f"  {'Domain':6s}  {'A':>7s}  {'B':>7s}  {'Delta':>8s}  {'p-value':>9s}  {'Sig':>4s}",
    ]
    for d in DOMAINS:
        row = diff["domain_deltas"][d]
        ps = f"{row['p_value']:.4f}" if row["p_value"] is not None else "N/A"
        sig = "YES" if row["significant"] else "no"
        lines.append(
            f"  {d:6s}  {row['score_a']:7.2f}  {row['score_b']:7.2f}  {row['delta']:+8.2f}  {ps:>9s}  {sig:>4s}"
        )

    lines.extend(
        [
            "",
            "Tier Deltas:",
            f"  {'Tier':6s}  {'A':>7s}  {'B':>7s}  {'Delta':>8s}  {'p-value':>9s}  {'Sig':>4s}",
        ]
    )
    for t in TIERS:
        row = diff["tier_deltas"][t]
        sa = f"{row['score_a']:7.2f}" if row["score_a"] is not None else "    N/A"
        sb = f"{row['score_b']:7.2f}" if row["score_b"] is not None else "    N/A"
        sd = f"{row['delta']:+8.2f}" if row["delta"] is not None else "     N/A"
        ps = f"{row['p_value']:.4f}" if row["p_value"] is not None else "N/A"
        sig = "YES" if row["significant"] else "no"
        lines.append(f"  {t:6s}  {sa}  {sb}  {sd}  {ps:>9s}  {sig:>4s}")

    regs = diff["regressions"]
    lines.append("")
    if regs:
        lines.append(f"REGRESSIONS DETECTED ({len(regs)}):")
        for r in regs:
            lines.append(f"  - [{r['scope']}:{r['key']}] {r['reason']}")
    else:
        lines.append("PASS: No regressions detected.")
    return "\n".join(lines)
