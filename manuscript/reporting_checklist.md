# Supplementary File S1 — Reporting checklist

**Manuscript:** Judge-dependent safety gains and model-specific helpfulness costs of evidence-sufficiency prompting in clinical LLMs.

No single reporting guideline governs non-interventional computational benchmarks of clinical LLM behavior. This study is not a clinical trial (no prospective patient care, no intervention delivered to patients) and not a conventional diagnostic-accuracy study of a device against a clinical reference standard. We therefore report following the *applicable principles* of DECIDE-AI [9], CONSORT-AI / SPIRIT-AI [10,12], and STARD-AI [11], adapted to a computational study in which the primary judge is an LLM and a second-family LLM offers a robustness check, **not an independent clinical reference standard**. Items that apply only to prospective, interventional, or in-clinic evaluations are marked Not applicable with the reason.

## Study identification and framing

| Item | Addressed | Location |
|---|---|---|
| Title identifies the study as an AI/LLM evaluation | Yes | Title |
| Abstract: structured background, methods, results, conclusions | Yes | Abstract |
| Intended use / clinical task stated | Yes — clinical decision-support text generation under incomplete information; not a deployed device | Introduction; Ethics |
| Deployment/readiness claims explicitly bounded | Yes — no deployment or patient-outcome claim | Abstract Conclusions; Discussion; Ethics |

## Data and participants (cases)

| Item | Addressed | Location |
|---|---|---|
| Data sources and eligibility | Yes — Real-POCQi, HealthBench, MedRBench; public, non-MIMIC | Methods: Datasets |
| Sampling frame and sample size | Yes — 300 paired item-perturbations/model (1,200 pairs; 2,400 outputs) + 480/arm control subset | Methods: Datasets; Results: Composition |
| Data-use / governance restrictions | Yes — MIMIC-CDM excluded from external API use | Methods; Ethics; Funding and Conflicts |
| Perturbation / input-degradation procedure | Yes | Methods: Perturbations |
| Handling of unusable (truncated/malformed) outputs | Yes — flagged and reported; a QA limitation | Methods: Cross-Judge Robustness; Limitations |

## Model(s) under test

| Item | Addressed | Location |
|---|---|---|
| Models, versions, and configuration (reasoning effort, decoding) | Yes — GPT-5.5, Claude Opus 4.8, Gemini 3.5 Flash, Grok 4.3 | Methods: Models; Table 2 |
| Prompts / intervention fully specified | Yes — standard, wrapper, neutral scaffold, format scaffold | Methods: Prompts |
| Run period / version drift caveat | Yes | Limitations (point 8) |

## Index test (LLM judge) and reference standard

| Item | Addressed | Location |
|---|---|---|
| Index test defined (primary judge, rubric, outputs) | Yes — GPT-5.4-nano structured rubric | Methods: Scoring |
| Index test treated as label, not ground truth | Yes | Methods: Scoring |
| Independent index test (different family) | Yes — Claude Sonnet 5 re-scored all double-judged cells | Methods: Judge Validation; Results: Cross-Judge |
| Clinician reference analysis | **Not reported as evidence** — AI-generated reviewer simulations were previously mislabeled as clinician validation and have been withdrawn | REPLICATION_AUDIT.md |
| Clinical-rater blinding | Not applicable — no authenticated human reference raters in this computational study | REPLICATION_AUDIT.md |
| Physician rater reliability | Not applicable — historical reviewer metrics excluded from scientific analysis | REPLICATION_AUDIT.md |
| Clinician-calibrated sensitivity/specificity | Not estimated; the nominal positive-enriched reviewer data were AI-generated and excluded | REPLICATION_AUDIT.md |

## Analysis

| Item | Addressed | Location |
|---|---|---|
| Primary estimand pre-specified | Yes — paired absolute risk difference, standard − wrapper | Methods: Statistical Analysis |
| Confirmatory vs. secondary/exploratory analyses labeled | Yes | Methods: Statistical Analysis |
| Uncertainty quantification | Yes — 10,000-resample paired bootstrap CIs; Wilson CIs for rates; McNemar; GEE | Methods; all figures |
| Multiplicity control | Yes — Holm and Benjamini-Hochberg across 13 subgroup tests | Methods; Results: Robustness |
| Effect heterogeneity tested formally | Yes — GEE model×wrapper interaction, joint Wald | Results: Robustness |
| Power / minimum detectable effect | Yes | Results: Robustness |
| Circularity / construct-validity control | Yes — matched neutral- and format-scaffold control arms; additive decomposition | Methods: Prompts; Results: Mechanism and Circularity Control |

## Results, harms, and interpretation

| Item | Addressed | Location |
|---|---|---|
| Flow of cases / composition reported | Yes | Results: Dataset and Run Composition |
| Primary result with CI and test | Yes | Results: Primary Outcome; Figure 2 |
| Per-subgroup results (model, dataset, perturbation) | Yes | Results; Figures 3–5 |
| Potential harms / safety-helpfulness trade-off | Yes — diagnostic-accuracy cost per model | Results: Helpfulness and Accuracy Trade-off; Figure 9 |
| Limitations | Yes — nine enumerated | Limitations |
| Generalizability bounded | Yes | Discussion; Limitations |

## Transparency

| Item | Addressed | Location |
|---|---|---|
| Data and code availability | Computational code, scores, tables and figures are public; raw model outputs are request-only subject to licences. Historical AI-generated reviewer aggregates are labeled deprecated, not physician data; no unconfirmed archive DOI is asserted | Data and Code Availability; REPLICATION_AUDIT.md |
| Funding and competing interests | Yes | Funding and Conflicts |
| Ethics statement | Yes — retrospective public non-MIMIC data; no prospective patient care | Ethics |
| Author contributions | Yes | Author Contributions |

## Items marked Not applicable

- Prospective enrolment, consent, randomization, allocation concealment, trial registration (CONSORT-AI/SPIRIT-AI): Not applicable — no prospective human participants and no delivered intervention.
- In-clinic human-AI interaction, workflow integration, and clinical safety monitoring (DECIDE-AI live-evaluation items): Not applicable — computational benchmark only; the manuscript explicitly states a prospective clinician-in-the-loop evaluation would be required before any deployment claim.
- Patient-outcome endpoints: Not applicable — the endpoint is a response-safety label, not a patient outcome.
