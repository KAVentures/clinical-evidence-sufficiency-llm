# Round 5: Citation and Factuality Audit

## Major concerns

- No empirical result should appear unless it comes from generated output tables.
- Model names and dates must be recorded at run time, because provider endpoints change.
- MIMIC-CDM should be cited only as excluded background, not as an executed API dataset.

## Minor concerns

- Bibliographic metadata should be checked again before journal submission.

## Required fixes applied

- Results section reports computed model-performance results from the cost-controlled four-model subset.
- References file includes source pointers but should be audited before submission.
- Results section was replaced with computed values from `outputs/tables/panel_*` tables. Claims remain limited to public-data computational benchmark findings.
