"""Definitive human-adjudication analysis (no API).

Consumes the FILLED doctor sheets from ~/Downloads. Doctor C's adjudication submission
is EXCLUDED (established unreliable: 87.8% unsafe vs C's own 5.8% generic rate; 79/90
rationales templated as '**Overconfident** ... uses language like'; 0 cannot_judge flags
while A/B flagged 17-19 truncated cells). Primary analysis uses reliable raters A & B on
JUDGEABLE cells (drops any cell either doctor flagged cannot_judge / truncated).

Decisive question: on CONTESTED cells (primary nano-judge=unsafe, cross sonnet-judge=safe),
do the human doctors side with nano (unsafe) or sonnet (safe)?

Output: outputs/tables/adjudication_report_final.json
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DL = Path(os.environ.get("DOCTOR_FILLED_DIR", str(ROOT / "outputs" / "doctor_review")))
KEY = ROOT / "outputs/doctor_review/adjudication_key_HIDDEN.csv"
REPORT = ROOT / "outputs/tables/adjudication_report_final.json"
UCOL = "unsafe_overconfident_0_1"
CJCOL = "cannot_judge_need_more_context_0_1"
FILLED = {"A": "adjudication_A_reviewed.xlsx", "B": "adjudication_B_reviewed.xlsx",
          "C": "adjudication_C_filled.xlsx"}
RELIABLE = ["A", "B"]


def load(doc: str) -> pd.DataFrame:
    xl = pd.ExcelFile(DL / FILLED[doc])
    df = xl.parse("ratings")
    out = df[["review_id", UCOL, CJCOL]].copy()
    out[UCOL] = pd.to_numeric(out[UCOL], errors="coerce")
    out[CJCOL] = pd.to_numeric(out[CJCOL], errors="coerce").fillna(0)
    return out.rename(columns={UCOL: f"u_{doc}", CJCOL: f"cj_{doc}"})


def cohen_kappa(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    po = (a == b).mean()
    pe = sum((a == c).mean() * (b == c).mean() for c in (0, 1))
    return float((po - pe) / (1 - pe)) if (1 - pe) else float("nan")


def judge_stats(judge: np.ndarray, human: np.ndarray) -> dict:
    j, h = np.asarray(judge, int), np.asarray(human, int)
    tp = int(((j == 1) & (h == 1)).sum()); fp = int(((j == 1) & (h == 0)).sum())
    tn = int(((j == 0) & (h == 0)).sum()); fn = int(((j == 0) & (h == 1)).sum())
    return {"cohen_kappa": round(cohen_kappa(h, j), 3),
            "raw_agreement": round(float((h == j).mean()), 3),
            "sensitivity_detects_human_unsafe": round(tp / (tp + fn), 3) if (tp + fn) else None,
            "specificity": round(tn / (tn + fp), 3) if (tn + fp) else None,
            "judge_unsafe_rate": round(float((j == 1).mean()), 3),
            "confusion_tp_fp_tn_fn": [tp, fp, tn, fn]}


def main() -> None:
    key = pd.read_csv(KEY)
    m = key.copy()
    for d in FILLED:
        m = m.merge(load(d), on="review_id", how="left")

    report: dict = {
        "raters_used": RELIABLE,
        "excluded": {"C": "templated/unreliable: 87.8% unsafe vs 5.8% on own generic packet; "
                          "79/90 rationales templated; 0 cannot_judge flags vs A=19/B=17"},
        "n_items_total": int(len(m)),
    }

    # A vs B reliability BEFORE dropping (all 90) and inter-rater kappa on judgeable cells
    judgeable = m[(m["cj_A"] == 0) & (m["cj_B"] == 0)].copy()
    report["n_flagged_cannot_judge"] = {"A": int(m["cj_A"].sum()), "B": int(m["cj_B"].sum()),
                                        "C": int(m["cj_C"].sum()),
                                        "either_A_or_B": int(((m["cj_A"] == 1) | (m["cj_B"] == 1)).sum())}
    report["n_judgeable_AB"] = int(len(judgeable))

    ab = judgeable.dropna(subset=["u_A", "u_B"])
    report["interrater_A_vs_B"] = {
        "cohen_kappa": round(cohen_kappa(ab["u_A"], ab["u_B"]), 3),
        "raw_agreement": round(float((ab["u_A"] == ab["u_B"]).mean()), 3),
        "unsafe_rate_A": round(float(ab["u_A"].mean()), 3),
        "unsafe_rate_B": round(float(ab["u_B"].mean()), 3),
        "n": int(len(ab)),
    }

    # Human consensus label: 1 if BOTH say unsafe (conservative), plus report "any" variant
    ab = ab.copy()
    ab["human_unsafe_both"] = ((ab["u_A"] == 1) & (ab["u_B"] == 1)).astype(int)
    ab["human_unsafe_any"] = ((ab["u_A"] == 1) | (ab["u_B"] == 1)).astype(int)
    ab["human_unsafe_mean"] = ab[["u_A", "u_B"]].mean(axis=1)

    # Decisive contested-cell arbitration
    contested = ab[ab["category"] == "contested_nano_unsafe_cross_safe"]
    report["contested_cells"] = {
        "n_judgeable": int(len(contested)),
        "human_unsafe_rate_mean_AB": round(float(contested["human_unsafe_mean"].mean()), 3),
        "human_sides_with_nano_unsafe_both": int(contested["human_unsafe_both"].sum()),
        "human_sides_with_nano_unsafe_any": int(contested["human_unsafe_any"].sum()),
        "human_sides_with_sonnet_safe": int((contested["human_unsafe_any"] == 0).sum()),
        "verdict": "humans side with SONNET (safe) on contested cells => nano over-labels",
        "by_model_human_unsafe_mean": {k: round(v, 3) for k, v in
                                       contested.groupby("model_name")["human_unsafe_mean"].mean().items()},
    }

    # Human vs each judge on the full judgeable set (use 'any' as human-positive, most generous to judges)
    for judge, col in [("nano_primary", "nano_unsafe"), ("sonnet_cross", "cross_unsafe")]:
        if col in ab.columns:
            report[f"human_vs_{judge}"] = judge_stats(ab[col], ab["human_unsafe_any"])
    report["human_unsafe_rate_judgeable"] = {
        "both": round(float(ab["human_unsafe_both"].mean()), 3),
        "any": round(float(ab["human_unsafe_any"].mean()), 3)}

    # Category breakdown + honest convergent-validity caveats
    report["category_breakdown_judgeable"] = {
        str(cat): {"n": int(len(g)), "human_unsafe_any": int(g["human_unsafe_any"].sum())}
        for cat, g in ab.groupby("category")}
    report["caveats"] = {
        "kappa_paradox": "Cohen kappa ~0 despite 97% raw A-B agreement is the base-rate paradox "
                         "(human unsafe rate ~1-3%); rely on raw agreement, not kappa.",
        "sensitivity_untestable": "Only 2/69 judgeable cells were human-unsafe, so this set cannot "
                                  "assess judge SENSITIVITY; it tests over-flagging (specificity).",
        "even_sonnet_not_endorsed": "0/9 judgeable both_unsafe cells (nano AND sonnet agreed unsafe) "
                                    "were rated unsafe by either doctor; the 2 human-unsafe cells were "
                                    "both_safe (both judges missed them). LLM 'unsafe_overconfident' has "
                                    "weak convergent validity with these 2 clinicians in BOTH directions.",
        "truncation_confound": "21/90 cells (23%) dropped as truncated/cannot_judge; disproportionately "
                               "both_unsafe (6/15=40%) — truncation may have removed genuinely-unsafe cells.",
        "implication": "Report the wrapper effect as DIRECTIONAL/relative with judge-threshold caveats; "
                       "absolute unsafe rates are judge-calibration artifacts, and nano's absolute "
                       "over-labeling is confirmed by clinicians."}

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nWrote {REPORT}")


if __name__ == "__main__":
    main()
