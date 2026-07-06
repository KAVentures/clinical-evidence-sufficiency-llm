from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def exact_two_sided_binom_p(k: int, n: int, p: float = 0.5) -> float:
    """Exact two-sided binomial test for p=0.5 without requiring SciPy."""
    if n == 0:
        return 1.0
    probs = np.array([_binom_pmf(i, n, p) for i in range(n + 1)])
    observed = _binom_pmf(k, n, p)
    return float(min(1.0, probs[probs <= observed + 1e-15].sum()))


def _binom_pmf(k: int, n: int, p: float) -> float:
    from math import comb

    return comb(n, k) * (p**k) * ((1 - p) ** (n - k))


def simulate_power(
    n_items: int = 200,
    n_models: int = 4,
    n_perturbations: int = 4,
    baseline_rate: float = 0.40,
    absolute_reduction: float = 0.10,
    within_pair_correlation: float = 0.30,
    n_sims: int = 1000,
    alpha: float = 0.05,
    seed: int = 20260704,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_pairs = n_items * n_models * n_perturbations
    rows = []
    wrapper_rate = max(0.0, baseline_rate - absolute_reduction)
    for sim in range(n_sims):
        latent = rng.normal(size=n_pairs)
        noise_a = rng.normal(size=n_pairs)
        noise_b = rng.normal(size=n_pairs)
        standard_score = within_pair_correlation * latent + np.sqrt(1 - within_pair_correlation**2) * noise_a
        wrapper_score = within_pair_correlation * latent + np.sqrt(1 - within_pair_correlation**2) * noise_b
        standard = standard_score < np.quantile(standard_score, baseline_rate)
        wrapper = wrapper_score < np.quantile(wrapper_score, wrapper_rate)
        b = int((standard & ~wrapper).sum())
        c = int((~standard & wrapper).sum())
        discordant = b + c
        p_value = 1.0 if discordant == 0 else exact_two_sided_binom_p(min(b, c), discordant)
        rows.append({"sim": sim, "risk_difference": standard.mean() - wrapper.mean(), "b": b, "c": c, "p_value": p_value, "significant": p_value < alpha})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-items", type=int, default=200)
    parser.add_argument("--n-models", type=int, default=4)
    parser.add_argument("--n-perturbations", type=int, default=4)
    parser.add_argument("--baseline-rate", type=float, default=0.40)
    parser.add_argument("--absolute-reduction", type=float, default=0.10)
    parser.add_argument("--n-sims", type=int, default=1000)
    parser.add_argument("--out", type=Path, default=Path("outputs/tables/power_simulation.csv"))
    args = parser.parse_args()
    result = simulate_power(
        n_items=args.n_items,
        n_models=args.n_models,
        n_perturbations=args.n_perturbations,
        baseline_rate=args.baseline_rate,
        absolute_reduction=args.absolute_reduction,
        n_sims=args.n_sims,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)
    print(f"Estimated power: {result['significant'].mean():.3f}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
