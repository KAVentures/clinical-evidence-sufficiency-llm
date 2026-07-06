"""Validate the gpt-5.4-nano judge against 3 doctors (no API calls).

Reads the completed blinded rating workbooks (outputs/doctor_review/doctor_review_[A|B|C].xlsx,
'ratings' tab) plus review_key_HIDDEN.csv, and reports:

  1. Inter-rater reliability among the 3 doctors (Fleiss' kappa, pairwise % agreement)
     for the PRIMARY outcome unsafe_overconfident.
  2. Judge vs human-majority agreement: Cohen's kappa, accuracy, sensitivity, specificity,
     PPV/NPV -- treating the doctor majority vote as ground truth.
  3. The headline wrapper effect recomputed under HUMAN majority labels vs the judge labels,
     so we can see whether standard->evidence_sufficiency still helps when the ground truth
     is expert clinicians instead of an LLM.

Run after doctors return their sheets:
    python analysis/judge_validation.py
Any unfilled rows are ignored with a warning, so you can run it on a partial return.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.stats import paired_risk_difference  # noqa: E402

DIR = ROOT / "outputs/doctor_review"
PRIMARY = "unsafe_overconfident_0_1"
DOCS = ["A", "B", "C"]


def load_doctor(doc: str) -> pd.DataFrame:
    xlsx = DIR / f"doctor_review_{doc}.xlsx"
    csv = DIR / f"doctor_review_{doc}.csv"
    if xlsx.exists():
        df = pd.read_excel(xlsx, sheet_name="ratings")
    elif csv.exists():
        df = pd.read_csv(csv)
    else:
        raise FileNotFoundError(f"No rating file for doctor {doc}")
    return df[["review_id", PRIMARY]].rename(columns={PRIMARY: f"doc_{doc}"})


def to_binary(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def fleiss_kappa(ratings: np.ndarray) -> float:
    """ratings: n_items x n_categories counts (here 2 categories, 3 raters)."""
    n_items, n_cat = ratings.shape
    n_raters = ratings.sum(axis=1)[0]
    p_j = ratings.sum(axis=0) / (n_items * n_raters)
    P_i = ((ratings**2).sum(axis=1) - n_raters) / (n_raters * (n_raters - 1))
    P_bar = P_i.mean()
    P_e = (p_j**2).sum()
    return float((P_bar - P_e) / (1 - P_e)) if (1 - P_e) else float("nan")


def cohen_kappa(a: pd.Series, b: pd.Series) -> float:
    tab = pd.crosstab(a, b).reindex(index=[0, 1], columns=[0, 1]).fillna(0).values
    n = tab.sum()
    po = np.trace(tab) / n
    pe = (tab.sum(0) * tab.sum(1)).sum() / n**2
    return float((po - pe) / (1 - pe)) if (1 - pe) else float("nan")


def main() -> None:
    key = pd.read_csv(DIR / "review_key_HIDDEN.csv")
    merged = key.copy()
    for doc in DOCS:
        merged = merged.merge(load_doctor(doc), on="review_id", how="left")
    for doc in DOCS:
        merged[f"doc_{doc}"] = to_binary(merged[f"doc_{doc}"])

    complete = merged.dropna(subset=[f"doc_{d}" for d in DOCS]).copy()
    n_done = len(complete)
    print(f"Fully-rated items: {n_done}/{len(merged)}")
    if n_done < 10:
        print("Not enough completed ratings yet -- returning. (Doctors still filling sheets.)")
        return

    for d in DOCS:
        complete[f"doc_{d}"] = complete[f"doc_{d}"].astype(int)

    # 1) Inter-rater reliability (primary outcome)
    counts = np.zeros((n_done, 2), dtype=int)
    for i, (_, row) in enumerate(complete.iterrows()):
        votes = [row[f"doc_{d}"] for d in DOCS]
        counts[i, 1] = sum(votes)
        counts[i, 0] = len(DOCS) - sum(votes)
    fk = fleiss_kappa(counts)
    print(f"\n[1] Inter-doctor reliability (unsafe_overconfident):")
    print(f"    Fleiss' kappa (3 raters) = {fk:.3f}")
    for a, b in [("A", "B"), ("A", "C"), ("B", "C")]:
        agree = (complete[f"doc_{a}"] == complete[f"doc_{b}"]).mean()
        print(f"    pairwise {a}-{b}: %agreement={agree*100:.1f}  Cohen kappa={cohen_kappa(complete[f'doc_{a}'], complete[f'doc_{b}']):.3f}")

    # Human majority vote
    complete["human_majority"] = (complete[[f"doc_{d}" for d in DOCS]].sum(axis=1) >= 2).astype(int)

    # 2) Judge vs human majority (human = truth)
    j = complete["unsafe_overconfident"].astype(int)
    h = complete["human_majority"]
    tp = int(((j == 1) & (h == 1)).sum()); fp = int(((j == 1) & (h == 0)).sum())
    tn = int(((j == 0) & (h == 0)).sum()); fn = int(((j == 0) & (h == 1)).sum())
    acc = (tp + tn) / n_done
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) else float("nan")
    npv = tn / (tn + fn) if (tn + fn) else float("nan")
    print(f"\n[2] Judge (gpt-5.4-nano) vs human majority [truth]:")
    print(f"    Cohen kappa={cohen_kappa(j, h):.3f}  accuracy={acc*100:.1f}%")
    print(f"    sensitivity={sens*100:.1f}%  specificity={spec*100:.1f}%  PPV={ppv*100:.1f}%  NPV={npv*100:.1f}%")
    print(f"    confusion: TP={tp} FP={fp} TN={tn} FN={fn}")

    # 3) Wrapper effect under human labels vs judge labels (on this subset)
    sub = complete.rename(columns={"model_name": "model_name"}).copy()
    sub["item_id"] = sub["perturbation_id"]  # cluster/pair key within this subset
    print(f"\n[3] standard -> evidence_sufficiency risk difference on the {n_done}-item review subset:")
    for label, col in [("JUDGE labels", "unsafe_overconfident"), ("HUMAN majority", "human_majority")]:
        tmp = sub.copy()
        tmp[col] = tmp[col].astype(int)
        try:
            rd = paired_risk_difference(tmp, item_cols=["perturbation_id", "model_name", "perturbation_type"], outcome=col)
            print(f"    {label:16s}: RD={rd*100:+.1f}pp  (paired items available in subset)")
        except Exception as exc:  # noqa: BLE001
            print(f"    {label:16s}: n/a ({exc})")

    print("\nInterpretation guide:")
    print("  Fleiss kappa >0.6 = good doctor agreement (the primary outcome is reliably ratable).")
    print("  Judge-vs-human kappa >0.6 and specificity high = the LLM judge is a defensible instrument.")
    print("  If the wrapper RD stays clearly positive under HUMAN labels, the headline survives human ground truth.")


if __name__ == "__main__":
    main()
