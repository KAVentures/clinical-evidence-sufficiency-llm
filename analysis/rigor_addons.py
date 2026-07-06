"""Free statistical-rigor add-ons (no API). Reads the primary nano scores on the common
panel and produces:
  1. Confirmatory-vs-exploratory analysis registry (which claims are pre-specified primary
     vs secondary vs exploratory).
  2. Multiple-comparison correction (Holm + Benjamini-Hochberg) across all per-model /
     per-dataset / per-perturbation McNemar tests of the wrapper effect.
  3. Formal effect-heterogeneity test: GEE logistic with a model x wrapper interaction, plus
     a joint Wald test that the wrapper effect is constant across models.
  4. Achieved power / minimum detectable effect (MDE) for the paired primary endpoint.

Output: outputs/tables/rigor_addons_report.json
"""
from __future__ import annotations

import json
import sys
from math import sqrt
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.stats import bh_fdr, mcnemar_by_pair  # noqa: E402

SCORES = ROOT / "outputs/scores/requested_panel_openai_judge_scores.jsonl"
REPORT = ROOT / "outputs/tables/rigor_addons_report.json"
FRONTIER = ["gpt-5.5", "claude-opus-4-8", "gemini-3.5-flash", "grok-4.3"]
STD, ES = "standard", "evidence_sufficiency"
OUTCOME = "unsafe_overconfident"
PAIR = ["item_id", "model_name", "perturbation_id"]


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def holm(pvals: list[float], alpha: float = 0.05) -> list[dict]:
    idx = np.argsort(pvals)
    m = len(pvals)
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(idx):
        val = (m - rank) * pvals[i]
        running = max(running, val)
        adj[i] = min(1.0, running)
    return adj


def registry() -> list[dict]:
    return [
        {"analysis": "Primary: wrapper reduces unsafe_overconfident on the common panel (paired RD, McNemar, GEE)",
         "status": "CONFIRMATORY (pre-specified primary endpoint)"},
        {"analysis": "Circularity controls: format_scaffold & neutral_scaffold arms + mechanism decomposition",
         "status": "SECONDARY (pre-specified in revision plan)"},
        {"analysis": "Helpfulness/accuracy tradeoff on answerable full_information items",
         "status": "SECONDARY (pre-specified: safety must not come from over-abstention)"},
        {"analysis": "Cross-judge robustness (claude-sonnet-5) of direction & magnitude",
         "status": "SECONDARY (pre-specified sensitivity analysis)"},
        {"analysis": "Per-model / per-dataset / per-perturbation heterogeneity of the wrapper effect",
         "status": "EXPLORATORY (multiplicity-corrected; hypothesis-generating)"},
        {"analysis": "Contradiction (conflicting-evidence) arm",
         "status": "EXPLORATORY (MedRBench-only; narrower construct)"},
        {"analysis": "Prompt-paraphrase robustness of the wrapper",
         "status": "EXPLORATORY (sensitivity)"},
        {"analysis": "Stochastic stability replicates",
         "status": "EXPLORATORY (sensitivity)"},
    ]


def subgroup_mcnemar(panel: pd.DataFrame) -> list[dict]:
    tests = []
    tests.append(("overall_common_panel", panel))
    for m in FRONTIER:
        tests.append((f"model:{m}", panel[panel["model_name"] == m]))
    for ds in sorted(panel["dataset"].unique()):
        tests.append((f"dataset:{ds}", panel[panel["dataset"] == ds]))
    for pt in sorted(panel["perturbation_type"].unique()):
        tests.append((f"perturbation:{pt}", panel[panel["perturbation_type"] == pt]))
    rows = []
    for name, sub in tests:
        r = mcnemar_by_pair(sub, item_cols=PAIR, outcome=OUTCOME)
        b, c = r["standard_unsafe_wrapper_safe"], r["standard_safe_wrapper_unsafe"]
        n_disc = b + c
        rows.append({"subgroup": name, "b_std_unsafe_es_safe": b, "c_std_safe_es_unsafe": c,
                     "n_discordant": n_disc, "statistic": r["statistic"], "p_value": r["p_value"]})
    return rows


def heterogeneity_test(panel: pd.DataFrame) -> dict:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    data = panel.copy()
    data["wrapper"] = (data["prompt_condition"] == ES).astype(int)
    data["answer_length_words"] = data.get("answer_length_words", 0)
    # main-effects-only vs interaction model; joint Wald that wrapper effect is constant across models
    try:
        full = smf.gee(f"{OUTCOME} ~ wrapper * C(model_name) + C(perturbation_type) + C(dataset)",
                       groups="item_id", data=data, family=sm.families.Binomial()).fit()
        inter_terms = [t for t in full.params.index if "wrapper:" in t]
        # joint Wald test on all interaction coefficients = 0
        R = np.zeros((len(inter_terms), len(full.params)))
        names = list(full.params.index)
        for i, t in enumerate(inter_terms):
            R[i, names.index(t)] = 1.0
        wald = full.wald_test(R, scalar=True)
        return {
            "interaction_terms": {t: {"coef": float(full.params[t]), "p": float(full.pvalues[t])}
                                  for t in inter_terms},
            "joint_wald_stat": float(np.ravel(wald.statistic)[0]),
            "joint_wald_p": float(np.ravel(wald.pvalue)[0]),
            "interpretation": "small joint p => wrapper effect differs across models (heterogeneous)",
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


def power_mde(panel: pd.DataFrame, alpha: float = 0.05, power: float = 0.80) -> dict:
    """Achieved precision + MDE for the paired primary endpoint via a normal approx to the
    paired risk difference (SE from discordant pairs)."""
    wide = panel.pivot_table(index=PAIR, columns="prompt_condition", values=OUTCOME, aggfunc="max")
    wide = wide.dropna(subset=[STD, ES])
    n = len(wide)
    diffs = (wide[STD] - wide[ES]).to_numpy(float)
    rd = float(diffs.mean())
    se = float(diffs.std(ddof=1) / sqrt(n))
    z_a = NormalDist().inv_cdf(1 - alpha / 2)
    z_b = NormalDist().inv_cdf(power)
    # observed achieved power for the observed effect
    achieved_power = float(1 - NormalDist().cdf(z_a - abs(rd) / se) + NormalDist().cdf(-z_a - abs(rd) / se))
    # MDE at target power using the observed per-pair SD scale
    sd_pair = float(diffs.std(ddof=1))
    mde = float((z_a + z_b) * sd_pair / sqrt(n))
    return {"n_pairs": n, "observed_rd": rd, "se": se,
            "ci95": [rd - z_a * se, rd + z_a * se],
            "achieved_power_for_observed_effect": round(achieved_power, 4),
            "mde_at_80pct_power_pp": round(mde * 100, 2)}


def main() -> None:
    scores = pd.DataFrame(read_jsonl(SCORES))
    panel = scores[(scores["model_name"].isin(FRONTIER))
                   & (scores["prompt_condition"].isin([STD, ES]))
                   & (scores["perturbation_type"] != "conflicting_evidence_llm")].copy()

    subs = subgroup_mcnemar(panel)
    pvals = [r["p_value"] for r in subs]
    bh = bh_fdr(pvals)["p_value_bh"].tolist()
    hm = holm(pvals)
    for r, b, h in zip(subs, bh, hm):
        r["p_bh"] = round(float(b), 5)
        r["p_holm"] = round(float(h), 5)
        r["sig_holm_0.05"] = bool(h <= 0.05)
        r["p_value"] = round(r["p_value"], 6)

    report = {
        "analysis_registry": registry(),
        "multiplicity": {
            "n_tests": len(subs),
            "method": "Holm (FWER) + Benjamini-Hochberg (FDR), alpha=0.05",
            "tests": subs,
        },
        "heterogeneity_model_x_wrapper": heterogeneity_test(panel),
        "power": power_mde(panel),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2))
    # console summary
    print("MULTIPLICITY (subgroup McNemar, Holm/BH corrected):")
    for r in subs:
        print(f"  {r['subgroup']:32s} b={r['b_std_unsafe_es_safe']:4d} c={r['c_std_safe_es_unsafe']:3d} "
              f"p={r['p_value']:.2e} p_holm={r['p_holm']:.3g} sig={r['sig_holm_0.05']}")
    print("\nHETEROGENEITY:", json.dumps({k: report["heterogeneity_model_x_wrapper"].get(k)
          for k in ("joint_wald_stat", "joint_wald_p")}, indent=2))
    print("POWER:", json.dumps(report["power"], indent=2))
    print(f"\nWrote {REPORT}")


if __name__ == "__main__":
    main()
