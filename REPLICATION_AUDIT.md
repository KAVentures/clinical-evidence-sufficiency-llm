# Independent reproducibility and claim audit

This audit distinguishes **archived statistical outputs** from analyses whose input
rating sheets can actually be independently re-run. It is not a new experiment
or an independent re-adjudication of clinical cases.

## Computational results cross-checked against saved JSON

- Primary paired panel: 1,200 matched cells; standard 49.33% versus
  evidence-sufficiency 24.67%; paired risk difference 24.67 percentage points;
  348 pairs switched unsafe-to-safe and 52 safe-to-unsafe.
- Cross-judge: 3,534 paired ratings; GPT-5.4-nano unsafe/Sonnet safe 1,147
  versus 15 reverse; common-panel reduction 24.67 versus 13.08 percentage points.
- Correctness analysis: 330 paired model-item observations from 200 answerable
  items (not 330 independently sampled clinical cases), uneven by model:
  GPT-5.5 48; Claude 44; Gemini 148; Grok 90. Correctness was adjudicated
  by **one** AI correctness judge, not verified by clinicians.
- Positive-enriched clinician-review *saved aggregate*: 66 judgeable from 72
  reviewed, 6/6 majority-unsafe detected by the primary judge, specificity
  33/60 (55%). These values are documented reports, **not independently
  audited labels** in this public repository.
- Adjudication *saved aggregate*: among 69 judgeable cases, the two retained
  clinicians marked 2 responses unsafe by either rater, and **both AI judges
  marked those two safe**. This conflicts with a general claim that the AI
  judges reliably detect all clinician-unsafe responses. The other 46
  judge-discordant cases were rated safe by the retained clinicians.

## Critical missing provenance

The nine public `outputs/doctor_review/{doctor_review,adjudication,sensitivity}_{A,B,C}.csv`
files contain case text and **blank clinician rating columns** (120, 90 and
72 cases per sheet, respectively). They are reviewer templates, not completed
ratings. `analysis/analyze_adjudication_final.py` reads separate
`*_reviewed.xlsx` / `*_filled.xlsx` from `DOCTOR_FILLED_DIR`, which are
not present at those paths in the public repository. The current public
archive therefore cannot independently reproduce the blinded clinician
outcomes or the rationale for excluding one adjudication sheet. The
manuscript must not imply otherwise.

To resolve: obtain the **authentic** finished reviewer workbooks, verify
reviewer provenance/authorization, deidentify, record immutable checksums,
and release them if permitted (or arrange an auditable controlled-access
deposit with a documented re-execution method). **Never populate blank
sheets with synthetic physician scores.** Rerun all physician calibration,
sensitivity, missingness, exclusion-inclusion sensitivity, and adjudication
analyses from those source records; then compare with archived reports.

## Remaining research limitations, not clerical errors

- 6 clinician-majority positive cases yield a very wide sensitivity interval.
  Enrichment was judge-label-blind but does not remove spectrum/selection bias.
- In the adjudication subset, 21/90 responses were excluded as unjudgeable,
  and exclusions were not equally distributed across judge categories.
- One adjudication rater was removed after inspecting an unusual pattern.
  Whether this was prespecified and the sensitivity to reinstating the rater
  must be made independently auditable.
- The correctness and model-specific abstention cost require separate
  blinded validation for *clinical* accuracy claims. The large Gemini signal
  remains informative as an AI-judge-assigned estimate.
- The cross-judge direction and matched scaffold controls strongly argue
  against a trivial formatting-only explanation. They do not establish a
  fully calibrated clinical safety effect.

## Automated sanity check

Run `python scripts/check_preprint_integrity.py` to confirm the numerical
results and report current reviewer-sheet availability. Run it with
`--require-filled-reviews` for a strict source-data gate; this is expected
to fail until authentic scored workbooks are supplied.

The checked-in PDFs were generated before subsequent Markdown source edits.
**Rebuild and visually check the PDF from the final Markdown source before
distributing it**; a stale PDF should not be represented as containing
these corrections.
