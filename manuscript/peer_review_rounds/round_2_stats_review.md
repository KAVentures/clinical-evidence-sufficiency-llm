# Round 2: Statistical Review

## Major concerns

- The unit of analysis is repeated by model, item, perturbation, prompt condition, and replicate.
- Model should be reported both as a fixed descriptive factor and in sensitivity analyses; item-level clustering is required.
- Absolute risk difference should be the main clinical estimate.

## Minor concerns

- Logistic model odds ratios should not be overemphasized when risk differences are more interpretable.

## Required fixes applied

- Implemented paired risk difference, clustered bootstrap confidence intervals, McNemar tests, GEE logistic sensitivity model, and BH/FDR correction helper.
- Power simulation stores assumptions in executable code.
- Final analysis reports paired absolute risk difference with 10,000 bootstrap resamples, McNemar discordance counts, per-model estimates, per-dataset estimates, and GEE sensitivity.
