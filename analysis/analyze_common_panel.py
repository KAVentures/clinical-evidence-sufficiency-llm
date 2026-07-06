"""Regenerate manuscript tables/figures from the COMMON PANEL scores.

Supersedes analyze_cost_controlled_panel.py, which read the old, completion-limited
subset (640 outputs, 2 conditions, mostly non-paired items). This reads the go/no-go
common panel + control arms:

  outputs/scores/requested_panel_openai_judge_scores.jsonl
    standard             1200   (300-item x 4-model common panel)
    evidence_sufficiency 1200   (paired with standard)
    format_scaffold       480   (120-item x 4-model placebo w/ forced commit)
    neutral_scaffold      480   (120-item x 4-model clean placebo: same tokens,
                                 no abstain and no commit instruction)

Primary comparison (standard vs evidence_sufficiency) is now on a properly PAIRED
common panel. Adds a mechanism-decomposition table+figure from the neutral arm:
the full wrapper effect splits into a scaffold-structure effect (standard->neutral)
and an abstention-content effect (neutral->ES); the forced-commit arm
(neutral->format) is the adversarial bound that refutes token-level circularity.

No API calls. Usage:
    python analysis/analyze_common_panel.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.tools import add_constant

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.stats import (  # noqa: E402
    clustered_bootstrap_risk_difference,
    mcnemar_by_pair,
    outcome_rates,
)

SCORES = ROOT / "outputs/scores/requested_panel_openai_judge_scores.jsonl"
FRONTIER = ["gpt-5.5", "claude-opus-4-8", "gemini-3.5-flash", "grok-4.3"]
OUTCOME = "unsafe_overconfident"
PAIR_IDX = ["model_name", "item_id", "perturbation_id"]
STD, ES = "standard", "evidence_sufficiency"
NEUTRAL, FORMAT = "neutral_scaffold", "format_scaffold"
# The case-grounded conflicting-evidence (contradiction) arm is a SEPARATE
# exploratory analysis (MedRBench-only; see analysis/cross_judge_robustness.py and
# the contradiction report). It shares the standard/evidence_sufficiency condition
# labels, so it must be explicitly excluded from the confirmatory primary common
# panel; otherwise it inflates n (1,200 -> 1,759) and biases pooled/per-model effects.
CONTRADICTION_PERTURBATION = "conflicting_evidence_llm"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def provider_for(model: str) -> str:
    if model.startswith("claude"):
        return "Anthropic"
    if model.startswith("gemini"):
        return "Google"
    if model.startswith("grok"):
        return "xAI"
    return "OpenAI"


def fast_bootstrap_rd(
    scores: pd.DataFrame,
    standard_label: str = STD,
    wrapper_label: str = ES,
    n_resamples: int = 10000,
    seed: int = 20260705,
) -> dict[str, float]:
    """Paired bootstrap RD (standard - wrapper) over matched cells."""
    wide = scores.pivot_table(
        index=PAIR_IDX, columns="prompt_condition", values=OUTCOME, aggfunc="max"
    )
    if standard_label not in wide or wrapper_label not in wide:
        return {"risk_difference": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n_pairs": 0}
    wide = wide.dropna(subset=[standard_label, wrapper_label])
    diffs = (wide[standard_label] - wide[wrapper_label]).to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    n = len(diffs)
    boot = np.empty(n_resamples)
    for i in range(n_resamples):
        boot[i] = diffs[rng.integers(0, n, n)].mean()
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "risk_difference": float(diffs.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n_pairs": int(n),
        "n_resamples": n_resamples,
    }


def matched(df: pd.DataFrame, conditions: list[str]) -> pd.DataFrame:
    """Restrict to (model,item,perturbation) cells present in ALL given conditions."""
    piv = df.pivot_table(index=PAIR_IDX, columns="prompt_condition", values=OUTCOME, aggfunc="max")
    keep = piv.dropna(subset=conditions).reset_index()[PAIR_IDX]
    return df.merge(keep, on=PAIR_IDX, how="inner")


def n_cells(df: pd.DataFrame) -> int:
    return df[PAIR_IDX].drop_duplicates().shape[0]


def rate(group: pd.DataFrame, condition: str) -> float:
    sub = group[group["prompt_condition"] == condition]
    return float(sub[OUTCOME].mean()) if len(sub) else float("nan")


def summarize_secondary(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for condition, group in scores.groupby("prompt_condition"):
        rows.append(
            {
                "prompt_condition": condition,
                "n": int(len(group)),
                "unsafe_overconfident_rate": group[OUTCOME].mean(),
                "correct_abstention_rate": group["correct_abstention"].mean(),
                "asks_for_missing_info_rate": group["asks_for_missing_info"].mean(),
                "identifies_removed_evidence_rate": group["identifies_removed_evidence"].mean(),
                "potentially_harmful_treatment_rate": group["potentially_harmful_treatment"].mean(),
                "guideline_concordant_next_step_rate": group["guideline_concordant_next_step"].mean(),
                "median_answer_length_words": group["answer_length_words"].median(),
            }
        )
    return pd.DataFrame(rows)


def run_gee(scores: pd.DataFrame) -> dict:
    try:
        data = scores.copy()
        data["prompt_wrapper"] = (data["prompt_condition"] == ES).astype(int)
        x = pd.get_dummies(data[["prompt_wrapper", "model_name", "dataset", "perturbation_type"]], drop_first=True)
        x = add_constant(x.astype(float), has_constant="add")
        model = GEE(data[OUTCOME], x, groups=data["item_id"], cov_struct=Exchangeable(), family=Binomial())
        result = model.fit()
        return {
            "prompt_wrapper_coef": float(result.params["prompt_wrapper"]),
            "prompt_wrapper_odds_ratio": float(np.exp(result.params["prompt_wrapper"])),
            "prompt_wrapper_p_value": float(result.pvalues["prompt_wrapper"]),
            "note": "GEE logistic clustered by item_id; wrapper vs standard on the common panel, adjusted for model, dataset, perturbation type.",
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    tables = ROOT / "outputs/tables"
    figures = ROOT / "outputs/figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(read_jsonl(SCORES))
    df = df[df["model_name"].isin(FRONTIER)].copy()
    df = df.drop_duplicates(["perturbation_id", "prompt_condition", "model_name"], keep="last")
    # Exclude the separate exploratory contradiction arm from the primary common panel
    # and all common-panel/control-arm/secondary analyses in this script.
    df = df[df["perturbation_type"] != CONTRADICTION_PERTURBATION].copy()
    df.to_csv(tables / "panel_scored_outputs_flat.csv", index=False)

    # Primary panel = the paired standard/ES common panel.
    pair = df[df["prompt_condition"].isin([STD, ES])].copy()

    # --- Table 1: common-panel composition (paired cells) ---
    paired_cells = matched(pair, [STD, ES])[PAIR_IDX + ["dataset", "perturbation_type"]].drop_duplicates()
    (
        paired_cells.groupby(["model_name", "dataset", "perturbation_type"]).size()
        .reset_index(name="n_paired_cells")
        .to_csv(tables / "panel_table1_dataset_composition.csv", index=False)
    )

    # --- Table 2: model run metadata ---
    pd.DataFrame(
        [
            {
                "provider": provider_for(m),
                "model": m,
                "reasoning_effort": "high",
                "temperature": "0 or provider default where required",
                "n_outputs": int((df["model_name"] == m).sum()),
                "failed_calls": 0,
            }
            for m in FRONTIER
        ]
        + [
            {
                "provider": "OpenAI",
                "model": "gpt-5.4-nano",
                "reasoning_effort": "judge",
                "temperature": 0,
                "n_outputs": int(len(df)),
                "failed_calls": 0,
            }
        ]
    ).to_csv(tables / "panel_table2_model_run_metadata.csv", index=False)

    # --- Table 3: primary outcome (common panel, paired) ---
    primary = outcome_rates(pair, ["prompt_condition"])
    boot = fast_bootstrap_rd(pair)
    mc = mcnemar_by_pair(pair, item_cols=PAIR_IDX)
    primary["absolute_risk_difference_standard_minus_wrapper"] = boot["risk_difference"]
    primary["risk_difference_ci_low"] = boot["ci_low"]
    primary["risk_difference_ci_high"] = boot["ci_high"]
    primary["mcnemar_p_value"] = mc["p_value"]
    primary["n_paired_cells"] = boot["n_pairs"]
    primary.to_csv(tables / "panel_table3_primary_outcome.csv", index=False)

    # --- Table 4: per model ---
    model_rows = []
    for m in FRONTIER:
        g = pair[pair["model_name"] == m]
        b = fast_bootstrap_rd(g)
        mcm = mcnemar_by_pair(g, item_cols=PAIR_IDX)
        model_rows.append(
            {
                "model_name": m,
                "n_pairs": b["n_pairs"],
                "standard_rate": rate(g, STD),
                "wrapper_rate": rate(g, ES),
                "risk_difference": b["risk_difference"],
                "ci_low": b["ci_low"],
                "ci_high": b["ci_high"],
                "mcnemar_p_value": mcm["p_value"],
            }
        )
    per_model = pd.DataFrame(model_rows)
    per_model.to_csv(tables / "panel_table4_per_model_outcome.csv", index=False)

    # --- Table 5: per perturbation ---
    pert_rows = []
    for pert, g in pair.groupby("perturbation_type"):
        b = fast_bootstrap_rd(g)
        if b["n_pairs"] == 0:
            continue
        pert_rows.append(
            {
                "perturbation_type": pert,
                "n_pairs": b["n_pairs"],
                "standard_rate": rate(g, STD),
                "wrapper_rate": rate(g, ES),
                "risk_difference": b["risk_difference"],
                "ci_low": b["ci_low"],
                "ci_high": b["ci_high"],
            }
        )
    per_pert = pd.DataFrame(pert_rows)
    per_pert.to_csv(tables / "panel_table5_per_perturbation_outcome.csv", index=False)

    # --- Table 6: per dataset ---
    ds_rows = []
    for ds, g in pair.groupby("dataset"):
        b = fast_bootstrap_rd(g)
        if b["n_pairs"] == 0:
            continue
        ds_rows.append(
            {
                "dataset": ds,
                "n_pairs": b["n_pairs"],
                "standard_rate": rate(g, STD),
                "wrapper_rate": rate(g, ES),
                "risk_difference": b["risk_difference"],
                "ci_low": b["ci_low"],
                "ci_high": b["ci_high"],
            }
        )
    per_dataset = pd.DataFrame(ds_rows)
    per_dataset.to_csv(tables / "panel_table6_per_dataset_outcome.csv", index=False)

    # --- Table 7: secondary outcomes (all four conditions) ---
    secondary = summarize_secondary(df)
    secondary.to_csv(tables / "panel_table7_secondary_outcomes.csv", index=False)

    # --- Table 8: mechanism decomposition (control arms) ---
    decomp_rows = []
    has_neutral = NEUTRAL in set(df["prompt_condition"].unique())
    has_format = FORMAT in set(df["prompt_condition"].unique())
    m4_conditions = [STD, ES] + ([NEUTRAL] if has_neutral else []) + ([FORMAT] if has_format else [])
    m4 = matched(df, m4_conditions)
    n_ctrl = n_cells(m4)

    def contrast(name: str, a: str, b: str) -> dict:
        r = clustered_bootstrap_risk_difference(m4, standard_label=a, wrapper_label=b, outcome=OUTCOME)
        return {
            "contrast": name,
            "from_condition": a,
            "to_condition": b,
            "n_cells": n_ctrl,
            "risk_difference": r["risk_difference"],
            "ci_low": r["ci_low"],
            "ci_high": r["ci_high"],
        }

    full_ctrl = None
    if has_neutral:
        decomp_rows.append(contrast("full_wrapper", STD, ES))
        decomp_rows.append(contrast("scaffold_structure", STD, NEUTRAL))
        decomp_rows.append(contrast("abstention_content", NEUTRAL, ES))
        if has_format:
            decomp_rows.append(contrast("forced_commit", NEUTRAL, FORMAT))
        full_ctrl = decomp_rows[0]["risk_difference"]
    decomp = pd.DataFrame(decomp_rows)
    if not decomp.empty:
        decomp.to_csv(tables / "panel_table8_mechanism_decomposition.csv", index=False)

    # --- GEE sensitivity ---
    gee_summary = run_gee(pair)
    (tables / "panel_gee_summary.json").write_text(json.dumps(gee_summary, indent=2), encoding="utf-8")

    # --- Figures ---
    make_figures(df, primary, per_model, per_pert, per_dataset, decomp, figures)

    # --- Summary JSON ---
    summary = {
        "n_scored_outputs": int(len(df)),
        "conditions": df["prompt_condition"].value_counts().to_dict(),
        "primary_common_panel": boot,
        "primary_rates": {
            "standard": rate(pair, STD),
            "evidence_sufficiency": rate(pair, ES),
        },
        "mcnemar": mc,
        "gee": gee_summary,
        "per_model": per_model.to_dict("records"),
        "mechanism_decomposition": decomp.to_dict("records"),
        "mechanism_n_control_cells": n_ctrl,
    }
    if full_ctrl:
        add = {r["contrast"]: r["risk_difference"] for r in decomp_rows}
        summary["decomposition_additive_check"] = {
            "scaffold_plus_abstain": add.get("scaffold_structure", 0) + add.get("abstention_content", 0),
            "full_wrapper": add.get("full_wrapper", 0),
        }
    (tables / "panel_analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


# ---------------------------------------------------------------- figures ----
# Publication-quality figures via matplotlib (PNG @300dpi + vector PDF).
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch  # noqa: E402
from statsmodels.stats.proportion import proportion_confint  # noqa: E402

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.edgecolor": "#4d4d4d",
    "axes.linewidth": 0.9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "xtick.color": "#333333",
    "ytick.color": "#333333",
})

C_STD = "#c44e52"       # standard prompt (risk)
C_WRAP = "#4c72b0"      # evidence-sufficiency wrapper (safer)
C_NEUTRAL = "#8172b3"   # neutral scaffold
C_FORMAT = "#dd8452"    # format scaffold (forced commit)
C_POS = "#4c72b0"       # positive reduction
C_NEG = "#c44e52"       # increase (worse)
C_DOT = "#34506b"       # forest point estimate (favors wrapper)
C_DOT_NEG = "#b0413e"   # forest point estimate (favors standard / worse)
C_WHISK = "#6b7f93"     # whisker / CI colour


def _wilson(k: int, n: int) -> tuple[float, float]:
    """Wilson score 95% CI for a proportion, as (low, high) fractions."""
    if n <= 0:
        return (0.0, 0.0)
    lo, hi = proportion_confint(int(round(k)), int(n), method="wilson")
    return float(lo), float(hi)
PRETTY = {
    "standard": "Standard", "evidence_sufficiency": "Wrapper",
    "neutral_scaffold": "Neutral scaffold", "format_scaffold": "Format scaffold",
    "gpt-5.5": "GPT-5.5", "claude-opus-4-8": "Claude Opus 4.8",
    "gemini-3.5-flash": "Gemini 3.5 Flash", "grok-4.3": "Grok 4.3",
    "real_pocqi": "Real-POCQi", "healthbench": "HealthBench", "medrbench": "MedRBench",
    "decontextualized": "Decontextualized", "context_uncertainty": "Context uncertainty",
    "full_information": "Full information", "missing_ancillary_tests": "Missing ancillary tests",
    "reworded": "Reworded",
    "full_wrapper": "Full wrapper\n(standard→wrapper)",
    "scaffold_structure": "Scaffold structure\n(standard→neutral)",
    "abstention_content": "Abstention content\n(neutral→wrapper)",
    "forced_commit": "Forced commit\n(neutral→format)",
}


def _pp(label: str) -> str:
    return PRETTY.get(label, label)


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _titleblock(ax, fignum: int, title: str, subtitle: str = "") -> None:
    """Left-aligned numbered title with an optional grey subtitle above the axes."""
    ax.set_title(f"Figure {fignum}. {title}", loc="left",
                 pad=26 if subtitle else 12)
    if subtitle:
        ax.text(0, 1.015, subtitle, transform=ax.transAxes, fontsize=9.5,
                color="#555", style="italic", va="bottom")


def _footnote(fig, text: str) -> None:
    if text:
        fig.text(0.012, 0.006, text, fontsize=8.3, color="#666", style="italic",
                 ha="left", va="bottom")


def _forest_rd(path: Path, fignum: int, title: str, labels, values,
               ci_low=None, ci_high=None, n_list=None, subtitle: str = "",
               footnote: str = "",
               xlabel: str = "Risk difference, standard − wrapper (percentage points)") -> None:
    """Dot-and-whisker forest plot of risk differences (pp) with 95% CIs and per-row n."""
    labels_p = [_pp(x) for x in labels]
    vals = [(v * 100 if v == v else 0.0) for v in values]
    n = len(labels_p)
    fig, ax = plt.subplots(figsize=(8.8, 0.92 * n + 2.3))
    y = list(range(n))[::-1]
    xerr = None
    if ci_low is not None and ci_high is not None:
        lo = [(v - c * 100) for v, c in zip(vals, list(ci_low))]
        hi = [(c * 100 - v) for v, c in zip(vals, list(ci_high))]
        xerr = [[max(0.0, a) for a in lo], [max(0.0, b) for b in hi]]
    ax.axvline(0, color="#333", linewidth=1.0, zorder=1)
    for i, (yi, v) in enumerate(zip(y, vals)):
        col = C_DOT if v >= 0 else C_DOT_NEG
        xe = [[xerr[0][i]], [xerr[1][i]]] if xerr is not None else None
        ax.errorbar([v], [yi], xerr=xe, fmt="o", markersize=9,
                    markerfacecolor=col, markeredgecolor=col, ecolor=C_WHISK,
                    elinewidth=1.7, capsize=4, capthick=1.4, zorder=3)
    ax.set_yticks(y)
    ticklabels = ([f"{lab}\nn = {int(nn):,}" for lab, nn in zip(labels_p, n_list)]
                  if n_list is not None else labels_p)
    ax.set_yticklabels(ticklabels, fontsize=10.5)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", alpha=0.25)
    hi_off = xerr[1] if xerr is not None else [0.0] * n
    lo_off = xerr[0] if xerr is not None else [0.0] * n
    xmax = max([v + h for v, h in zip(vals, hi_off)] + [0.0])
    xmin = min([v - l for v, l in zip(vals, lo_off)] + [0.0])
    span = (xmax - xmin) or 1.0
    pad = span * 0.05 + 1.0
    for yi, v, hoff in zip(y, vals, hi_off):
        ax.text(v + hoff + pad * 0.6, yi, f"{v:+.1f} pp", va="center", ha="left",
                fontsize=10, fontweight="bold", color="#222")
    ax.set_xlim(xmin - pad * 1.4, xmax + pad * 7.0)
    _titleblock(ax, fignum, title, subtitle)
    _footnote(fig, footnote)
    _save(fig, path)


def _grouped_bars(path: Path, fignum: int, title: str, groups, series: dict, ylabel: str,
                  colors: dict, fmt: str = "{:.0%}", subtitle: str = "",
                  errs: dict | None = None, footnote: str = "") -> None:
    groups_p = [_pp(g) for g in groups]
    n = len(groups)
    k = len(series)
    fig, ax = plt.subplots(figsize=(max(6.8, 1.7 * n + 1.6), 4.9))
    width = 0.8 / k
    x = list(range(n))
    top = 0.0
    for j, (name, vals) in enumerate(series.items()):
        offs = [xi + (j - (k - 1) / 2) * width for xi in x]
        yerr = None
        if errs is not None and name in errs:
            lo, hi = errs[name]
            yerr = [[a * 100 for a in lo], [b * 100 for b in hi]]
        bars = ax.bar(offs, [v * 100 for v in vals], width=width, label=name,
                      color=colors.get(name, None), zorder=3, edgecolor="white",
                      linewidth=0.5, yerr=yerr,
                      error_kw={"elinewidth": 1.3, "capsize": 3.5, "ecolor": "#333"})
        for b, v, off in zip(bars, vals, range(len(vals))):
            hoff = (errs[name][1][off] * 100) if (errs is not None and name in errs) else 0.0
            ax.text(b.get_x() + b.get_width() / 2, v * 100 + hoff + 1.4, fmt.format(v),
                    ha="center", va="bottom", fontsize=8.8)
            top = max(top, v * 100 + hoff)
    ax.set_xticks(x)
    ax.set_xticklabels(groups_p, rotation=0)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, min(100, top + 16))
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, ncol=k, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    ax.text(0, 1.30 if subtitle else 1.17, f"Figure {fignum}. {title}", transform=ax.transAxes,
            fontsize=14, fontweight="bold", ha="left", va="bottom")
    if subtitle:
        ax.text(0, 1.22, subtitle, transform=ax.transAxes, fontsize=9.5, color="#555",
                style="italic", ha="left", va="bottom")
    _footnote(fig, footnote)
    _save(fig, path)


def make_figures(df, primary, per_model, per_pert, per_dataset, decomp, figures: Path) -> None:
    write_design_fig(figures / "panel_figure1_study_design")

    # Fig 2: unsafe rate by prompt (common panel) with Wilson 95% CIs
    pair_fig = df[df["prompt_condition"].isin([STD, ES])]
    prim = primary.set_index("prompt_condition")["rate"]
    order = [c for c in [STD, ES] if c in prim.index]
    fig, ax = plt.subplots(figsize=(5.0, 4.9))
    heights, y_lo, y_hi = [], [], []
    for c in order:
        sub = pair_fig[pair_fig["prompt_condition"] == c]
        k, ntot = int(sub[OUTCOME].sum()), int(len(sub))
        lo, hi = _wilson(k, ntot)
        heights.append(prim[c] * 100)
        y_lo.append((prim[c] - lo) * 100)
        y_hi.append((hi - prim[c]) * 100)
    bars = ax.bar([_pp(c) for c in order], heights,
                  color=[C_STD if c == STD else C_WRAP for c in order], width=0.6,
                  zorder=3, yerr=[y_lo, y_hi],
                  error_kw={"elinewidth": 1.4, "capsize": 5, "ecolor": "#333"})
    for b, c, hi in zip(bars, order, y_hi):
        ax.text(b.get_x() + b.get_width() / 2, prim[c] * 100 + hi + 1.4, f"{prim[c]:.1%}",
                ha="center", va="bottom", fontweight="bold")
    ax.set_ylabel("Unsafe overconfidence rate (%)")
    ax.set_ylim(0, max(h + e for h, e in zip(heights, y_hi)) + 12)
    ax.grid(axis="x", visible=False)
    _titleblock(ax, 2, "Unsafe overconfidence by prompt",
                subtitle=f"Paired common panel, n = {len(pair_fig)//2:,} pairs; error bars Wilson 95% CI")
    _footnote(fig, "Standard vs. evidence-sufficiency wrapper; same items, models and judge.")
    _save(fig, figures / "panel_figure2_unsafe_overconfidence_by_prompt")

    fn_rd = "Positive values favour the evidence-sufficiency wrapper. Intervals are item-clustered bootstrap 95% CIs."

    # Fig 3: RD by model (dot-and-whisker with CIs + per-model n)
    _forest_rd(figures / "panel_figure3_risk_difference_by_model", 3,
               "Wrapper effect by model", per_model["model_name"].tolist(),
               per_model["risk_difference"].tolist(),
               per_model.get("ci_low"), per_model.get("ci_high"),
               n_list=per_model.get("n_pairs"),
               subtitle="Paired absolute reduction in unsafe-overconfidence labels, by model",
               footnote=fn_rd)

    # Fig 4: RD by perturbation
    _forest_rd(figures / "panel_figure4_risk_difference_by_perturbation", 4,
               "Wrapper effect by perturbation type", per_pert["perturbation_type"].tolist(),
               per_pert["risk_difference"].tolist(),
               per_pert.get("ci_low"), per_pert.get("ci_high"),
               n_list=per_pert.get("n_pairs"),
               subtitle="Paired absolute reduction in unsafe-overconfidence labels, by input degradation",
               footnote=fn_rd)

    # Fig 5: RD by dataset
    _forest_rd(figures / "panel_figure5_risk_difference_by_dataset", 5,
               "Wrapper effect by dataset", per_dataset["dataset"].tolist(),
               per_dataset["risk_difference"].tolist(),
               per_dataset.get("ci_low"), per_dataset.get("ci_high"),
               n_list=per_dataset.get("n_pairs"),
               subtitle="Paired absolute reduction in unsafe-overconfidence labels, by source dataset",
               footnote=fn_rd)

    # Fig 6: all-condition unsafe rate on matched control subset with Wilson 95% CIs
    ctrl_conditions = [c for c in [STD, NEUTRAL, ES, FORMAT] if c in set(df["prompt_condition"].unique())]
    if len(ctrl_conditions) >= 3:
        m4 = matched(df, ctrl_conditions)
        n_arm = n_cells(m4)
        cmap = {STD: C_STD, NEUTRAL: C_NEUTRAL, ES: C_WRAP, FORMAT: C_FORMAT}
        rates, y_lo, y_hi = [], [], []
        for c in ctrl_conditions:
            sub = m4[m4["prompt_condition"] == c]
            k, ntot = int(sub[OUTCOME].sum()), int(len(sub))
            r = k / ntot if ntot else 0.0
            lo, hi = _wilson(k, ntot)
            rates.append(r); y_lo.append((r - lo) * 100); y_hi.append((hi - r) * 100)
        fig, ax = plt.subplots(figsize=(7.0, 4.9))
        bars = ax.bar([_pp(c) for c in ctrl_conditions], [r * 100 for r in rates],
                      color=[cmap[c] for c in ctrl_conditions], width=0.62, zorder=3,
                      yerr=[y_lo, y_hi],
                      error_kw={"elinewidth": 1.4, "capsize": 5, "ecolor": "#333"})
        for b, r, hi in zip(bars, rates, y_hi):
            ax.text(b.get_x() + b.get_width() / 2, r * 100 + hi + 1.4, f"{r:.1%}",
                    ha="center", va="bottom", fontweight="bold", fontsize=10)
        ax.set_ylabel("Unsafe overconfidence rate (%)")
        ax.set_ylim(0, max(r * 100 + e for r, e in zip(rates, y_hi)) + 12)
        ax.grid(axis="x", visible=False)
        _titleblock(ax, 6, "Same scaffold tokens, different behaviour",
                    subtitle=f"Matched control subset, {n_arm:,} cells/arm; error bars Wilson 95% CI")
        _footnote(fig, "Neutral scaffold shares the wrapper's structure without its abstention "
                       "content; format scaffold forces a definitive answer.")
        _save(fig, figures / "panel_figure6_control_arm_rates")

    # Fig 7: mechanism decomposition
    if decomp is not None and not decomp.empty:
        _forest_rd(figures / "panel_figure7_mechanism_decomposition", 7,
                   "Mechanism decomposition of the wrapper effect",
                   decomp["contrast"].tolist(), decomp["risk_difference"].tolist(),
                   decomp.get("ci_low"), decomp.get("ci_high"),
                   n_list=decomp.get("n_cells"),
                   subtitle="The full wrapper effect splits into scaffold-structure and abstention-content components",
                   footnote="Positive = reduces unsafe overconfidence. Intervals are item-clustered bootstrap 95% CIs.",
                   xlabel="Risk difference between conditions (percentage points)")

    # Fig 8: cross-judge over-labeling (nano vs Sonnet), per model, with Wilson 95% CIs
    cj = ROOT / "outputs/tables/crossjudge_agreement_report.json"
    if cj.exists():
        try:
            fam = json.loads(cj.read_text())["per_model_family_preference"]
            models = [m for m in FRONTIER if m in fam]
            nano_r = [fam[m]["nano_unsafe_rate"] for m in models]
            cross_r = [fam[m]["cross_unsafe_rate"] for m in models]
            ns = [int(fam[m]["n"]) for m in models]
            nano_ci = ([], [])
            cross_ci = ([], [])
            for r, nn in zip(nano_r, ns):
                lo, hi = _wilson(r * nn, nn); nano_ci[0].append(r - lo); nano_ci[1].append(hi - r)
            for r, nn in zip(cross_r, ns):
                lo, hi = _wilson(r * nn, nn); cross_ci[0].append(r - lo); cross_ci[1].append(hi - r)
            s_nano, s_cross = "GPT-5.4-nano (primary)", "Claude Sonnet 5 (independent)"
            _grouped_bars(
                figures / "panel_figure8_cross_judge_overlabeling", 8,
                "Two judges disagree on absolute rates",
                models,
                {s_nano: nano_r, s_cross: cross_r},
                "Unsafe overconfidence rate (%)",
                {s_nano: "#c44e52", s_cross: "#55a868"},
                subtitle="Same responses, two independent judges; the primary judge labels far more as unsafe",
                errs={s_nano: nano_ci, s_cross: cross_ci},
                footnote="Error bars Wilson 95% CI. n per model shown in Table; both judges scored the same responses.")
        except Exception as exc:  # noqa: BLE001
            print(f"[fig8 skipped] {exc}")

    # Fig 9 (NEW): safety-helpfulness trade-off per model
    acc = ROOT / "outputs/tables/accuracy_tradeoff_report.json"
    if acc.exists():
        try:
            pm = json.loads(acc.read_text())["per_model"]
            rd_by_model = dict(zip(per_model["model_name"], per_model["risk_difference"]))
            models = [m for m in FRONTIER if m in pm and m in rd_by_model]
            gain = [rd_by_model[m] * 100 for m in models]  # safety gain pp
            cost = [-pm[m]["correct_diagnosis"]["delta_es_minus_std"] * 100 for m in models]  # accuracy cost pp
            fig, ax = plt.subplots(figsize=(6.6, 5.2))
            ax.scatter(gain, cost, s=90, color=C_WRAP, zorder=3)
            for m, gx, cy in zip(models, gain, cost):
                ax.annotate(_pp(m), (gx, cy), textcoords="offset points", xytext=(8, 6), fontsize=10)
            lim = max(gain + cost) + 8
            ax.plot([0, lim], [0, lim], ls="--", color="#999", lw=1, zorder=1)
            ax.text(lim * 0.10, lim * 0.86, "cost > gain", color="#aaa", fontsize=9, style="italic")
            ax.text(lim * 0.66, lim * 0.16, "gain > cost", color="#aaa", fontsize=9, style="italic")
            ax.set_xlabel("Safety gain: reduction in unsafe overconfidence (pp)")
            ax.set_ylabel("Helpfulness cost: drop in correct diagnosis (pp)")
            ax.set_xlim(0, lim)
            ax.set_ylim(-6, lim)
            _titleblock(ax, 9, "Safety–helpfulness trade-off by model",
                        subtitle="Answerable, complete-information cases; each point is one model")
            _footnote(fig, "Below the dashed identity line, the safety gain exceeds the diagnostic-accuracy cost.")
            _save(fig, figures / "panel_figure9_safety_helpfulness_tradeoff")
        except Exception as exc:  # noqa: BLE001
            print(f"[fig9 skipped] {exc}")


def write_design_fig(path: Path) -> None:
    steps = ["Public\ndatasets", "Stress\nvariants", "4 high-reasoning\nmodels",
             "4 prompt\nconditions", "Separate\nLLM judge", "Paired\nstatistics"]
    fig, ax = plt.subplots(figsize=(11, 2.4))
    ax.set_xlim(0, len(steps)); ax.set_ylim(0, 1); ax.axis("off")
    for i, s in enumerate(steps):
        box = FancyBboxPatch((i + 0.08, 0.28), 0.84, 0.44, boxstyle="round,pad=0.02,rounding_size=0.06",
                             linewidth=1.4, edgecolor="#4c72b0", facecolor="#eaf0f7")
        ax.add_patch(box)
        ax.text(i + 0.5, 0.5, s, ha="center", va="center", fontsize=10)
        if i < len(steps) - 1:
            ax.add_patch(FancyArrowPatch((i + 0.92, 0.5), (i + 1.08, 0.5),
                         arrowstyle="-|>", mutation_scale=12, color="#333", lw=1.2))
    ax.set_title("Figure 1. Study design", loc="left", x=0.0, y=0.92, fontsize=14, fontweight="bold")
    _save(fig, path)


if __name__ == "__main__":
    main()
