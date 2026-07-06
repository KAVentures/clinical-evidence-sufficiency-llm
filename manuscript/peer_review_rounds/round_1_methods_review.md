# Round 1: Methods Review

## Major concerns

- The primary hypothesis and primary endpoint are prespecified.
- MIMIC-CDM must be excluded from external API runs because MIMIC-derived data are currently prohibited from use with external APIs.
- Synthetic contradictions must remain visibly tagged in manifests and methods.

## Minor concerns

- Real-POCQi perturbations are text-query perturbations and should be complemented by public diagnostic cases from MedRBench.

## Required fixes applied

- Added explicit MIMIC external-API exclusion to README, protocol, and manuscript.
- Added perturbation manifest fields for synthetic text and expected missing evidence.
- Added cost-controlled final analysis description: 80 paired item-perturbations per requested high-reasoning model.
