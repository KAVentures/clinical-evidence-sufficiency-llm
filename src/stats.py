from __future__ import annotations

import numpy as np
import pandas as pd


def outcome_rates(scores: pd.DataFrame, group_cols: list[str], outcome: str = "unsafe_overconfident") -> pd.DataFrame:
    grouped = scores.groupby(group_cols, dropna=False)[outcome]
    out = grouped.agg(["sum", "count", "mean"]).reset_index()
    out = out.rename(columns={"sum": f"{outcome}_n", "count": "total_n", "mean": "rate"})
    return out


def paired_risk_difference(
    scores: pd.DataFrame,
    item_cols: list[str] | None = None,
    standard_label: str = "standard",
    wrapper_label: str = "evidence_sufficiency",
    outcome: str = "unsafe_overconfident",
) -> float:
    item_cols = item_cols or ["item_id", "model_name", "perturbation_type"]
    wide = scores.pivot_table(index=item_cols, columns="prompt_condition", values=outcome, aggfunc="mean")
    paired = wide.dropna(subset=[standard_label, wrapper_label])
    return float((paired[standard_label] - paired[wrapper_label]).mean())


def clustered_bootstrap_risk_difference(
    scores: pd.DataFrame,
    cluster_col: str = "item_id",
    n_resamples: int = 1000,
    seed: int = 20260704,
    **kwargs,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    clusters = scores[cluster_col].dropna().unique()
    estimates = []
    for _ in range(n_resamples):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        sample = pd.concat([scores[scores[cluster_col] == c] for c in sampled], ignore_index=True)
        estimates.append(paired_risk_difference(sample, **kwargs))
    point = paired_risk_difference(scores, **kwargs)
    lo, hi = np.nanpercentile(estimates, [2.5, 97.5])
    return {"risk_difference": point, "ci_low": float(lo), "ci_high": float(hi), "n_resamples": n_resamples}


def mcnemar_by_pair(
    scores: pd.DataFrame,
    item_cols: list[str] | None = None,
    standard_label: str = "standard",
    wrapper_label: str = "evidence_sufficiency",
    outcome: str = "unsafe_overconfident",
) -> dict[str, float]:
    item_cols = item_cols or ["item_id", "model_name", "perturbation_type"]
    wide = scores.pivot_table(index=item_cols, columns="prompt_condition", values=outcome, aggfunc="max")
    paired = wide.dropna(subset=[standard_label, wrapper_label])
    b = int(((paired[standard_label] == 1) & (paired[wrapper_label] == 0)).sum())
    c = int(((paired[standard_label] == 0) & (paired[wrapper_label] == 1)).sum())
    statistic, p_value = _mcnemar_chi_square_p(b, c)
    return {"standard_unsafe_wrapper_safe": b, "standard_safe_wrapper_unsafe": c, "statistic": statistic, "p_value": p_value}


def gee_logistic(scores: pd.DataFrame, outcome: str = "unsafe_overconfident"):
    import statsmodels.formula.api as smf
    import statsmodels.api as sm

    data = scores.copy()
    data["prompt_wrapper"] = (data["prompt_condition"] == "evidence_sufficiency").astype(int)
    formula = f"{outcome} ~ prompt_wrapper + C(perturbation_type) + C(model_name) + C(dataset) + answer_length_words"
    return smf.gee(formula, groups="item_id", data=data, family=sm.families.Binomial()).fit()


def bh_fdr(p_values: list[float], alpha: float = 0.05) -> pd.DataFrame:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    m = len(p)
    adjusted_ranked = np.minimum.accumulate(((m / np.arange(m, 0, -1)) * ranked[::-1]))[::-1]
    adjusted_ranked = np.minimum(adjusted_ranked, 1.0)
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = adjusted_ranked
    return pd.DataFrame({"p_value": p_values, "p_value_bh": adjusted, "reject_bh": adjusted <= alpha})


def wald_ci_for_rate(n: int, total: int, alpha: float = 0.05) -> tuple[float, float]:
    if total == 0:
        return float("nan"), float("nan")
    from statistics import NormalDist

    z = NormalDist().inv_cdf(1 - alpha / 2)
    p = n / total
    se = np.sqrt(p * (1 - p) / total)
    return max(0.0, p - z * se), min(1.0, p + z * se)


def _mcnemar_chi_square_p(b: int, c: int) -> tuple[float, float]:
    from math import erfc, sqrt

    if b + c == 0:
        return 0.0, 1.0
    statistic = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = erfc(sqrt(statistic / 2))
    return float(statistic), float(p_value)
