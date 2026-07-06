"""Go/no-go decision: does the format-only placebo reproduce the wrapper's effect?

Reads the panel judge scores from run_requested_model_panel.py --mode scores and
quantifies how much of the evidence-sufficiency wrapper's reduction in
unsafe-overconfident responses is explained by the *format scaffold alone*
(the circularity confound the peer review flagged) vs. the reasoning content.

Key quantity: "reasoning beyond format" = paired RD (format_scaffold - evidence_sufficiency).
If its bootstrap CI excludes 0, the wrapper does something a same-shaped placebo does
not -> GO. If it includes/straddles 0, the headline is largely judge-format circularity
-> NO-GO (still a publishable negative/mechanism result, just not the original claim).

No API calls. Usage:
    python analysis/gonogo_decision.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.stats import clustered_bootstrap_risk_difference, paired_risk_difference  # noqa: E402

SCORES = ROOT / "outputs/scores/requested_panel_openai_judge_scores.jsonl"
FRONTIER = ["gpt-5.5", "claude-opus-4-8", "gemini-3.5-flash", "grok-4.3"]
OUTCOME = "unsafe_overconfident"
IDX = ["item_id", "model_name", "perturbation_type"]


def load() -> pd.DataFrame:
    rows = [json.loads(line) for line in SCORES.read_text().splitlines() if line.strip()]
    df = pd.DataFrame(rows)
    return df[df["model_name"].isin(FRONTIER)].copy()


def matched(df: pd.DataFrame, conditions: list[str]) -> pd.DataFrame:
    """Restrict to (item, model, perturbation) cells present in ALL given conditions."""
    piv = df.pivot_table(index=IDX, columns="prompt_condition", values=OUTCOME, aggfunc="max")
    keep = piv.dropna(subset=conditions).reset_index()[IDX]
    return df.merge(keep, on=IDX, how="inner")


def n_cells(df: pd.DataFrame) -> int:
    return df[IDX].drop_duplicates().shape[0]


def main() -> None:
    if not SCORES.exists():
        print(f"No scores file yet at {SCORES}. Run --mode scores first.")
        return
    df = load()
    conds = set(df["prompt_condition"].unique())
    print("conditions present:", sorted(conds))
    print("score rows:", len(df), "| by model:", df["model_name"].value_counts().to_dict())

    # 1) Wrapper effect on the full common paired panel.
    full = matched(df, ["standard", "evidence_sufficiency"])
    w = clustered_bootstrap_risk_difference(full, standard_label="standard", wrapper_label="evidence_sufficiency", outcome=OUTCOME)
    print(
        f"\n[FULL PANEL n={n_cells(full)} paired cells] wrapper effect (standard - ES): "
        f"RD={w['risk_difference']*100:+.1f}pp  CI[{w['ci_low']*100:+.1f}, {w['ci_high']*100:+.1f}]"
    )

    if "format_scaffold" not in conds:
        print("\nNo format_scaffold scores yet -> cannot run the go/no-go comparison.")
        return

    # 2) Three-way matched subset: items with standard, ES, AND format all present.
    m = matched(df, ["standard", "evidence_sufficiency", "format_scaffold"])
    wrap = clustered_bootstrap_risk_difference(m, standard_label="standard", wrapper_label="evidence_sufficiency", outcome=OUTCOME)
    fmt = clustered_bootstrap_risk_difference(m, standard_label="standard", wrapper_label="format_scaffold", outcome=OUTCOME)
    extra = clustered_bootstrap_risk_difference(m, standard_label="format_scaffold", wrapper_label="evidence_sufficiency", outcome=OUTCOME)
    frac = (fmt["risk_difference"] / wrap["risk_difference"]) if wrap["risk_difference"] else float("nan")

    print(f"\n[MATCHED 3-WAY SUBSET n={n_cells(m)} cells]")
    print(f"  wrapper effect     (standard  - ES):     RD={wrap['risk_difference']*100:+.1f}pp  CI[{wrap['ci_low']*100:+.1f}, {wrap['ci_high']*100:+.1f}]")
    print(f"  format-only effect (standard  - format): RD={fmt['risk_difference']*100:+.1f}pp  CI[{fmt['ci_low']*100:+.1f}, {fmt['ci_high']*100:+.1f}]")
    print(f"  reasoning beyond format (format - ES):   RD={extra['risk_difference']*100:+.1f}pp  CI[{extra['ci_low']*100:+.1f}, {extra['ci_high']*100:+.1f}]")
    print(f"  fraction of wrapper effect explained by format alone: {frac*100:.0f}%")

    print("\n[PER MODEL] standard->ES | standard->format | format->ES (pp)")
    for model in FRONTIER:
        sub = m[m["model_name"] == model]
        if sub.empty:
            continue
        wr = paired_risk_difference(sub, standard_label="standard", wrapper_label="evidence_sufficiency", outcome=OUTCOME)
        fm = paired_risk_difference(sub, standard_label="standard", wrapper_label="format_scaffold", outcome=OUTCOME)
        ex = paired_risk_difference(sub, standard_label="format_scaffold", wrapper_label="evidence_sufficiency", outcome=OUTCOME)
        print(f"  {model:22s} {wr*100:+6.1f} | {fm*100:+6.1f} | {ex*100:+6.1f}")

    # 3) NEUTRAL-SCAFFOLD decomposition -- the clean circularity control.
    # format_scaffold conflates the scaffold with a forced-commit instruction, so it is an
    # adversarial bound, not a placebo. neutral_scaffold has the SAME tokens but neither an
    # abstain nor a commit instruction, isolating: (a) the scaffold alone, and (b) the
    # abstention instruction added on top. Requires the neutral arm to have been run.
    if "neutral_scaffold" in conds:
        m4 = matched(df, ["standard", "neutral_scaffold", "evidence_sufficiency", "format_scaffold"])
        scaffold = clustered_bootstrap_risk_difference(m4, standard_label="standard", wrapper_label="neutral_scaffold", outcome=OUTCOME)
        abstain = clustered_bootstrap_risk_difference(m4, standard_label="neutral_scaffold", wrapper_label="evidence_sufficiency", outcome=OUTCOME)
        full = clustered_bootstrap_risk_difference(m4, standard_label="standard", wrapper_label="evidence_sufficiency", outcome=OUTCOME)
        commit = clustered_bootstrap_risk_difference(m4, standard_label="neutral_scaffold", wrapper_label="format_scaffold", outcome=OUTCOME)
        s_frac = scaffold["risk_difference"] / full["risk_difference"] if full["risk_difference"] else float("nan")
        a_frac = abstain["risk_difference"] / full["risk_difference"] if full["risk_difference"] else float("nan")

        def line(name: str, r: dict) -> str:
            return f"  {name:38s} RD={r['risk_difference']*100:+.1f}pp  CI[{r['ci_low']*100:+.1f}, {r['ci_high']*100:+.1f}]"

        print(f"\n[MATCHED 4-WAY SUBSET n={n_cells(m4)} cells] mechanism decomposition")
        print(line("scaffold alone (standard->neutral)", scaffold))
        print(line("abstain instr. (neutral->ES)", abstain))
        print(line("FULL wrapper   (standard->ES)", full))
        print(line("forced commit  (neutral->format)", commit))
        print(f"  additive check: scaffold+abstain = {(scaffold['risk_difference']+abstain['risk_difference'])*100:+.1f}pp vs full {full['risk_difference']*100:+.1f}pp")
        print(f"  attribution: ~{s_frac*100:.0f}% structure, ~{a_frac*100:.0f}% abstention content")

    print("\n=== GO / NO-GO ===")
    if "neutral_scaffold" in conds and scaffold["ci_low"] > 0 and abstain["ci_low"] > 0:
        print(
            f"GO (clean control): the wrapper effect ({full['risk_difference']*100:.1f}pp) decomposes into a real "
            f"scaffold effect ({scaffold['risk_difference']*100:.1f}pp, CI excludes 0) PLUS a real abstention-content "
            f"effect ({abstain['risk_difference']*100:.1f}pp, CI excludes 0). Identical scaffold tokens produce very "
            "different unsafe rates across neutral/wrapper/format, so the judge scores behavior, not tokens -> "
            "circularity refuted. Report the decomposition, not a monolithic 'reasoning helps' claim."
        )
    elif extra["ci_low"] > 0:
        print(
            f"GO: the wrapper reduces unsafe answers by {extra['risk_difference']*100:.1f}pp beyond the (commit-forced) "
            f"format placebo (CI excludes 0). NOTE: run the neutral_scaffold arm for the clean decomposition -- "
            "format_scaffold conflates scaffold with forced commitment."
        )
    elif extra["ci_high"] <= 0:
        print(
            f"NO-GO: format-only reproduces the full effect (reasoning-beyond-format "
            f"CI[{extra['ci_low']*100:+.1f}, {extra['ci_high']*100:+.1f}] does not exceed 0)."
        )
    else:
        print(
            f"AMBIGUOUS: reasoning-beyond-format RD={extra['risk_difference']*100:+.1f}pp "
            f"CI[{extra['ci_low']*100:+.1f}, {extra['ci_high']*100:+.1f}] straddles 0."
        )


if __name__ == "__main__":
    main()
