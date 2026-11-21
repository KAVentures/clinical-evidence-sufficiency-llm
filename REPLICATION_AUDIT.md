# Independent reproducibility and reviewer-provenance audit

## Critical correction: purported human review was AI-generated or unauthenticated

Earlier versions of this research manuscript described three blinded physician
raters, reported sensitivity 1.00, specificity 0.55, and drew clinical
conclusions from reviewer adjudication worksheets.

**Those claims must not be treated as physician validation.** The user's
research file library contains several completed workbooks with
`model_generated=true` provenance. Prior July 2026 research conversations
explicitly instructed ChatGPT to complete the 120-item review, and to
complete the 72-item sensitivity review in different risk-averse/aggressive
reviewer styles. Examples include `doctor_review_A_completed.xlsx`,
`sensitivity_A_filled.xlsx` and `sensitivity_B_filled_aggressive.xlsx`.
Some adjudication sheets are also model-generated, and independent human
authorship of the remaining ratings cannot be verified.

The GitHub `outputs/doctor_review/{doctor_review,adjudication,sensitivity}_{A,B,C}.csv`
files contain **no completed rating fields**; they are blank reviewer
templates. Historical aggregate reports derived from scoring sheets cannot
be reverse-engineered into authentic physician ratings.

**Decision:** remove all purported clinician-majority outcomes, physician
sensitivity/specificity/PPV, and reviewer consensus/exclusion claims from
the scientific findings. Retain old output tables only as clearly labeled
**DEPRECATED — model-generated / unauthenticated reviewer provenance** for
the historical audit trail. No valid physician-equivalence or clinician
calibration inference can be derived from them. Do not fabricate missing
reference labels.

## Valid computational evidence cross-checked against archived JSON

- 1,200 paired model-item comparisons; GPT-5.4-nano-assigned unsafe
  overconfidence 49.33% standard versus 24.67% wrapper; reduction
  24.67 percentage points (95% item-bootstrap CI 21.75-27.67).
- Paired label changes: 348 unsafe-to-safe and 52 safe-to-unsafe.
- Claude Sonnet 5 cross-judge: 3,534 doubly rated cells; 1,147 primary
  unsafe/Sonnet safe versus 15 reverse. The paired reduction was 13.08
  points under Sonnet versus 24.67 under the primary judge.
- The matched 480-cell control subset had 9.0-point standard-to-neutral
  structure-associated and 14.8-point neutral-to-wrapper
  abstention-instruction-associated contrasts; the sum equals the 23.75
  point full-wrapper contrast on those same items. These are under the
  primary scoring judge, and condition-dependent grader bias remains possible.
- Single-judge correctness analysis: 330 paired model-item observations
  from 200 answerable source items, not 330 independent clinical cases.
  Allocation was uneven: GPT 48; Claude 44; Gemini 148; Grok 90.
  GPT-5.4-mini-assigned correct diagnoses 80.3% standard versus 50.3%
  wrapper. These are **not clinician-verified diagnostic accuracies**.
- Provider metadata, archived score files, model settings, and source
  manifest need to accompany every reproducible claim.

## Scope and next independent requirements

The paired within-judge result and cross-judge disagreement are legitimate,
informative findings about model response behavior and evaluation
procedures. They **do not** show that a given diagnosis or medical plan
was objectively safe or harmful. Genuine human clinical validation would
require identifying independently recruited, consented clinicians;
locking their rubric before viewing responses; preserving original
responses and timestamps; ensuring rater/model/condition blinding;
and maintaining unmodified deidentified scoring records with verifiable
provenance. That validation is not represented as completed here.

Historical clinician-review code under `analysis/` is retained for research
history and must not be used to relabel AI-produced sheets as doctors.
The archived `sensitivity_report*.json` and
`adjudication_report_final.json` are not clinical validation.

## Reproduction checks and document artifacts

Run `python scripts/check_preprint_integrity.py` to verify the saved
computational panel, cross-judge, and correctness summaries.
The human-source-data gate remains intentionally unavailable, because
there is no authenticated human reference dataset.

The authoritative manuscript source is `manuscript/manuscript.md`.
Regenerate and visually inspect `manuscript/manuscript.pdf` after source
changes before distribution; the old committed PDF may contain withdrawn
physician-validation claims until rebuilt. No existing DOI/archive is
claimed for a new fixed release.
