"""Estimate judge SENSITIVITY against clinicians on the positive-enriched set (no API).

Consumes the FILLED sensitivity sheets and the hidden key, and answers the question the
earlier clinician sets could not: when clinicians call a response unsafe-overconfident, does
the primary judge (gpt-5.4-nano) catch it? It also re-estimates specificity/PPV on this fresh,
enrichment-balanced sample and reports the same for the conservative cross judge (Claude
Sonnet 5) on the cells Sonnet actually scored (the common-panel strata only).

Reliability guardrails (mirroring analyze_adjudication_final.py):
  * cells any included rater flagged cannot_judge are dropped;
  * a rater whose rationales are heavily templated or whose unsafe rate is a wild outlier is
    reported so the user can exclude it with --exclude;
  * clinician-unsafe truth is reported three ways (both / any / majority) because the base rate
    and rater count drive which is appropriate.

Input search order for each rater X in {A,B,C}:
  ~/Downloads/sensitivity_X_reviewed.xlsx -> ~/Downloads/sensitivity_X_filled.xlsx ->
  outputs/doctor_review/sensitivity_X.xlsx   (only counts if actually filled in)

Usage (after sheets return):
    python analysis/analyze_sensitivity.py                 # auto-detect filled raters
    python analysis/analyze_sensitivity.py --exclude C     # drop an unreliable rater
Output: outputs/tables/sensitivity_report.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DL = Path.home() / "Downloads"
DR = ROOT / "outputs/doctor_review"
KEY = DR / "sensitivity_key_HIDDEN.csv"
REPORT = ROOT / "outputs/tables/sensitivity_report.json"
UCOL = "unsafe_overconfident_0_1"
CJCOL = "cannot_judge_need_more_context_0_1"
RCOL = "doctor_rationale"


def find_sheet(doc: str) -> Path | None:
    for cand in (DL / f"sensitivity_{doc}_reviewed.xlsx", DL / f"sensitivity_{doc}_filled.xlsx",
                 DR / f"sensitivity_{doc}.xlsx"):
        if cand.exists():
            return cand
    return None


def load(doc: str) -> pd.DataFrame | None:
    path = find_sheet(doc)
    if path is None:
        return None
    df = pd.ExcelFile(path).parse("ratings")
    u = pd.to_numeric(df.get(UCOL), errors="coerce")
    if u.notna().sum() == 0:                     # sheet present but unrated
        return None
    out = pd.DataFrame({"review_id": df["review_id"]})
    out[f"u_{doc}"] = u
    out[f"cj_{doc}"] = pd.to_numeric(df.get(CJCOL), errors="coerce").fillna(0)
    rat = df.get(RCOL, pd.Series([""] * len(df))).fillna("").astype(str)
    out[f"templated_{doc}"] = rat.str.strip().str.lower().str.startswith("**overconfident")
    return out


def cohen_kappa(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    po = (a == b).mean()
    pe = sum((a == c).mean() * (b == c).mean() for c in (0, 1))
    return float((po - pe) / (1 - pe)) if (1 - pe) else float("nan")


def diag(judge: np.ndarray, human: np.ndarray) -> dict:
    j, h = np.asarray(judge, int), np.asarray(human, int)
    tp = int(((j == 1) & (h == 1)).sum()); fp = int(((j == 1) & (h == 0)).sum())
    tn = int(((j == 0) & (h == 0)).sum()); fn = int(((j == 0) & (h == 1)).sum())
    return {
        "n": int(len(j)), "n_human_unsafe": int(h.sum()),
        "sensitivity": round(tp / (tp + fn), 3) if (tp + fn) else None,
        "specificity": round(tn / (tn + fp), 3) if (tn + fp) else None,
        "ppv": round(tp / (tp + fp), 3) if (tp + fp) else None,
        "npv": round(tn / (tn + fn), 3) if (tn + fn) else None,
        "cohen_kappa": round(cohen_kappa(h, j), 3),
        "raw_agreement": round(float((h == j).mean()), 3),
        "confusion_tp_fp_tn_fn": [tp, fp, tn, fn],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exclude", nargs="*", default=[], help="rater letters to drop (e.g. C)")
    args = ap.parse_args()

    key = pd.read_csv(KEY)
    raters, loaded = [], {}
    for doc in ["A", "B", "C"]:
        if doc in args.exclude:
            continue
        d = load(doc)
        if d is not None:
            loaded[doc] = d
            raters.append(doc)

    if not raters:
        print("No filled sensitivity sheets found yet. Re-run after doctors return them.")
        print(f"  looked in {DL} and {DR} for sensitivity_[A|B|C]_reviewed/_filled/.xlsx")
        return

    m = key.copy()
    for doc in raters:
        m = m.merge(loaded[doc], on="review_id", how="left")

    ucols = [f"u_{d}" for d in raters]
    cjcols = [f"cj_{d}" for d in raters]

    report: dict = {
        "purpose": "estimate primary-judge sensitivity vs clinicians on a positive-enriched set",
        "raters_loaded": raters,
        "raters_excluded": args.exclude,
        "n_items_total": int(len(m)),
        "rater_quality": {
            d: {"unsafe_rate": round(float(pd.to_numeric(m[f"u_{d}"], errors="coerce").mean()), 3),
                "cannot_judge": int(m[f"cj_{d}"].fillna(0).sum()),
                "templated_rationales": int(m[f"templated_{d}"].fillna(False).sum())}
            for d in raters},
    }

    # Judgeable = every included rater rated it and none flagged cannot_judge
    judgeable = m.dropna(subset=ucols).copy()
    flagged = np.zeros(len(judgeable), dtype=bool)
    for c in cjcols:
        flagged |= (judgeable[c].fillna(0) == 1).values
    judgeable = judgeable[~flagged].copy()
    for c in ucols:
        judgeable[c] = judgeable[c].astype(int)
    report["n_judgeable"] = int(len(judgeable))
    report["n_dropped_cannot_judge_or_incomplete"] = int(len(m) - len(judgeable))

    if len(judgeable) < 10:
        report["status"] = "insufficient judgeable items so far"
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        return

    # Inter-rater (if >1 rater)
    if len(raters) > 1:
        pairwise = {}
        for i in range(len(raters)):
            for jx in range(i + 1, len(raters)):
                a, b = raters[i], raters[jx]
                pairwise[f"{a}_vs_{b}"] = {
                    "raw_agreement": round(float((judgeable[f"u_{a}"] == judgeable[f"u_{b}"]).mean()), 3),
                    "cohen_kappa": round(cohen_kappa(judgeable[f"u_{a}"], judgeable[f"u_{b}"]), 3)}
        report["interrater"] = pairwise

    # Clinician-unsafe truth variants
    judgeable["human_any"] = (judgeable[ucols].sum(axis=1) >= 1).astype(int)
    judgeable["human_all"] = (judgeable[ucols].sum(axis=1) == len(ucols)).astype(int)
    judgeable["human_majority"] = (judgeable[ucols].mean(axis=1) >= 0.5).astype(int)
    report["human_unsafe_rate"] = {
        "any": round(float(judgeable["human_any"].mean()), 3),
        "all": round(float(judgeable["human_all"].mean()), 3),
        "majority": round(float(judgeable["human_majority"].mean()), 3)}

    # Primary result: nano vs clinician truth (all judgeable cells have a nano label)
    report["judge_vs_clinician"] = {}
    for truth in (["human_majority", "human_any", "human_all"] if len(raters) > 1 else ["human_any"]):
        block = {"nano_primary": diag(judgeable["nano_unsafe"], judgeable[truth])}
        # Sonnet only where it scored (common-panel strata); NaN elsewhere
        son = judgeable.dropna(subset=["cross_unsafe"])
        if len(son) >= 10:
            block["sonnet_cross"] = diag(son["cross_unsafe"], son[truth]) | {
                "note": f"computed on {len(son)}/{len(judgeable)} cells Sonnet scored (common-panel strata only)"}
        report["judge_vs_clinician"][truth] = block

    # Sensitivity by design stratum (the enrichment target)
    strat = {}
    for s, g in judgeable.groupby("stratum"):
        strat[s] = {
            "n_judgeable": int(len(g)),
            "human_unsafe_any": int(g["human_any"].sum()),
            "nano_unsafe": int(g["nano_unsafe"].sum()),
            "nano_sensitivity_vs_human_any": diag(g["nano_unsafe"], g["human_any"])["sensitivity"]}
    report["by_stratum"] = strat

    report["interpretation"] = {
        "sensitivity": "P(nano flags unsafe | clinician unsafe). This set was enriched (via degraded "
                       "inputs + forced-commit prompts, blind to judge labels) specifically so this is "
                       "estimable; the earlier sets had too few clinician-unsafe cells.",
        "reads_with_prior": "Combine with the adjudication finding (low specificity / PPV ~15%): together "
                            "they characterize the primary judge as a high-sensitivity, low-specificity screen "
                            "IF sensitivity is high here. If sensitivity is ALSO low, the endpoint mislabels in "
                            "both directions and the construct itself is weakly validated.",
        "sonnet_gap": "Sonnet did not score the format_scaffold arm, so its stats use common-panel strata only.",
    }

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nWrote {REPORT}")


if __name__ == "__main__":
    main()
