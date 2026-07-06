"""Zero-new-API reanalyses addressing reviewer confound concerns.

Runs three analyses on the already-scored panel:

1. GEE variants  - wrapper effect with no covariates, with the prespecified
   covariates, and with answer_length added. Because the wrapper causes length,
   answer_length is a mediator; a large attenuation when it is added means the
   reported effect is partly explained by verbosity rather than reasoning.
2. Length-stratified rates - unsafe-overconfidence rate by answer-length tertile
   within each prompt condition. If longer answers are already safer inside the
   standard arm, the judge is partly reacting to length, not safety.
3. Refusal-vs-answer stratification - paired risk difference split by whether the
   item should be refused/caveated (insufficient-info variants) or answered
   (full-information / harmless-reword variants). Tells us whether the effect is
   concentrated in unanswerable items.

Outputs go to outputs/tables/panel_reanalysis_*.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

STANDARD = "standard"
WRAPPER = "evidence_sufficiency"
OUTCOME = "unsafe_overconfident"

# Perturbation types where a safe model should decline / caveat / seek info
SHOULD_REFUSE = {"decontextualized", "context_uncertainty", "missing_ancillary_tests"}
# Perturbation types that remain answerable from the given text
SHOULD_ANSWER = {"full_information", "reworded"}


def main() -> None:
    tables = ROOT / "outputs/tables"
    flat = pd.read_csv(tables / "panel_scored_outputs_flat.csv")
    flat = flat.drop_duplicates(["perturbation_id", "prompt_condition", "model_name"], keep="last")
    pair = flat[flat["prompt_condition"].isin([STANDARD, WRAPPER])].copy()

    gee = gee_variants(pair)
    (tables / "panel_reanalysis_gee_variants.json").write_text(json.dumps(gee, indent=2), encoding="utf-8")

    length_tbl = length_stratified_rates(pair)
    length_tbl.to_csv(tables / "panel_reanalysis_length_stratified.csv", index=False)

    refans_tbl = refusal_vs_answer(pair)
    refans_tbl.to_csv(tables / "panel_reanalysis_refusal_vs_answer.csv", index=False)

    print("=== GEE variants (wrapper log-odds vs standard) ===")
    print(json.dumps(gee, indent=2))
    print("\n=== Unsafe rate by answer-length tertile within condition ===")
    print(length_tbl.to_string(index=False))
    print("\n=== Paired risk difference: should-refuse vs should-answer ===")
    print(refans_tbl.to_string(index=False))


def paired_wide(scores: pd.DataFrame, value: str = OUTCOME) -> pd.DataFrame:
    wide = scores.pivot_table(
        index=["model_name", "item_id", "perturbation_id"],
        columns="prompt_condition",
        values=value,
        aggfunc="max",
    ).dropna(subset=[STANDARD, WRAPPER])
    return wide


def bootstrap_rd(diffs: np.ndarray, n_resamples: int = 10000, seed: int = 20260704) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(diffs)
    if n == 0:
        return {"risk_difference": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n_pairs": 0}
    boot = np.array([diffs[rng.integers(0, n, n)].mean() for _ in range(n_resamples)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "risk_difference": float(diffs.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n_pairs": int(n),
    }


def gee_variants(scores: pd.DataFrame) -> dict:
    try:
        import statsmodels.api as sm
        from statsmodels.genmod.cov_struct import Exchangeable
        from statsmodels.genmod.families import Binomial
        from statsmodels.genmod.generalized_estimating_equations import GEE
        from statsmodels.tools import add_constant
    except Exception as exc:  # noqa: BLE001
        return {"error": f"statsmodels unavailable: {type(exc).__name__}: {exc}"}

    data = scores.copy()
    data["prompt_wrapper"] = (data["prompt_condition"] == WRAPPER).astype(int)

    specs = {
        "unadjusted": ["prompt_wrapper"],
        "adjusted_prespecified": ["prompt_wrapper", "model_name", "dataset", "perturbation_type"],
        "adjusted_plus_length_MEDIATOR": [
            "prompt_wrapper",
            "model_name",
            "dataset",
            "perturbation_type",
            "answer_length_words",
        ],
    }

    out: dict[str, dict] = {}
    for name, cols in specs.items():
        try:
            frame = data[cols + [OUTCOME, "item_id"]].dropna()
            design = pd.get_dummies(frame[cols], columns=[c for c in cols if frame[c].dtype == object], drop_first=True)
            design = add_constant(design.astype(float), has_constant="add")
            result = GEE(
                frame[OUTCOME].astype(float),
                design,
                groups=frame["item_id"],
                cov_struct=Exchangeable(),
                family=Binomial(),
            ).fit()
            out[name] = {
                "prompt_wrapper_coef": float(result.params["prompt_wrapper"]),
                "prompt_wrapper_odds_ratio": float(np.exp(result.params["prompt_wrapper"])),
                "prompt_wrapper_p_value": float(result.pvalues["prompt_wrapper"]),
                "n_obs": int(len(frame)),
            }
        except Exception as exc:  # noqa: BLE001
            out[name] = {"error": f"{type(exc).__name__}: {exc}"}

    base = out.get("adjusted_prespecified", {}).get("prompt_wrapper_coef")
    med = out.get("adjusted_plus_length_MEDIATOR", {}).get("prompt_wrapper_coef")
    if isinstance(base, float) and isinstance(med, float) and base != 0:
        out["length_attenuation_of_coef"] = {
            "coef_without_length": base,
            "coef_with_length": med,
            "proportion_attenuated": float((base - med) / base),
            "note": "Large positive attenuation => the wrapper effect is partly mediated by answer length.",
        }
    return out


def length_stratified_rates(scores: pd.DataFrame) -> pd.DataFrame:
    df = scores.copy()
    # Tertile edges from the pooled length distribution so bins are comparable across conditions.
    edges = df["answer_length_words"].quantile([0, 1 / 3, 2 / 3, 1.0]).to_numpy()
    edges[0] -= 1  # include the minimum
    df["length_tertile"] = pd.cut(
        df["answer_length_words"], bins=edges, labels=["short", "medium", "long"], include_lowest=True
    )
    rows = []
    for (cond, tertile), grp in df.groupby(["prompt_condition", "length_tertile"], observed=True):
        rows.append(
            {
                "prompt_condition": cond,
                "length_tertile": str(tertile),
                "n": int(len(grp)),
                "median_words": float(grp["answer_length_words"].median()),
                "unsafe_rate": float(grp[OUTCOME].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["prompt_condition", "length_tertile"]).reset_index(drop=True)


def refusal_vs_answer(scores: pd.DataFrame) -> pd.DataFrame:
    df = scores.copy()

    def group_of(pt: str) -> str:
        if pt in SHOULD_REFUSE:
            return "should_refuse"
        if pt in SHOULD_ANSWER:
            return "should_answer"
        return "other"

    df["expected_behavior"] = df["perturbation_type"].map(group_of)
    rows = []
    for behavior, grp in df.groupby("expected_behavior"):
        wide = paired_wide(grp)
        diffs = (wide[STANDARD] - wide[WRAPPER]).to_numpy(dtype=float)
        boot = bootstrap_rd(diffs)
        rows.append(
            {
                "expected_behavior": behavior,
                "perturbation_types": ", ".join(sorted(grp["perturbation_type"].unique())),
                "n_pairs": boot["n_pairs"],
                "standard_unsafe_rate": float(wide[STANDARD].mean()),
                "wrapper_unsafe_rate": float(wide[WRAPPER].mean()),
                "risk_difference": boot["risk_difference"],
                "ci_low": boot["ci_low"],
                "ci_high": boot["ci_high"],
            }
        )
    return pd.DataFrame(rows).sort_values("expected_behavior").reset_index(drop=True)


if __name__ == "__main__":
    main()
