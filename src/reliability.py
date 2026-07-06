from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score


def percent_agreement(a: pd.Series, b: pd.Series) -> float:
    valid = a.notna() & b.notna()
    if valid.sum() == 0:
        return float("nan")
    return float((a[valid] == b[valid]).mean())


def cohen_kappa(a: pd.Series, b: pd.Series) -> float:
    valid = a.notna() & b.notna()
    if valid.sum() == 0:
        return float("nan")
    return float(cohen_kappa_score(a[valid], b[valid]))


def krippendorff_alpha_nominal(ratings: pd.DataFrame) -> float:
    """Compute nominal Krippendorff alpha with rows as items and columns as raters."""
    values = ratings.to_numpy()
    observed_disagreement = 0.0
    observed_pairs = 0
    all_values = []
    for row in values:
        row = [x for x in row if pd.notna(x)]
        all_values.extend(row)
        for a, b in itertools.combinations(row, 2):
            observed_disagreement += float(a != b)
            observed_pairs += 1
    if observed_pairs == 0:
        return float("nan")
    observed = observed_disagreement / observed_pairs
    expected_pairs = list(itertools.combinations(all_values, 2))
    if not expected_pairs:
        return float("nan")
    expected = np.mean([a != b for a, b in expected_pairs])
    if expected == 0:
        return 1.0
    return float(1 - observed / expected)


def adjudicate_majority(ratings: pd.DataFrame) -> pd.Series:
    return ratings.mode(axis=1, dropna=True).iloc[:, 0]

