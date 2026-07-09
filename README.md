# Evidence Sufficiency and Unsafe Overconfidence in Clinical LLM Decision Support

**A paired public-data stress test showing that a measured "safety gain" from prompting clinical LLMs is directionally real but _judge-dependent_ — its magnitude nearly halves under an independent judge, blinded clinicians characterize the automated label as a high-sensitivity, low-specificity screen, and the behavior change carries a model-specific helpfulness cost. Matched scaffold controls show the direction is genuine behavior change, not judge circularity.**

**Author:** Koyar Afrasyab, M.D. · Kinvectum AB, Sweden · [ORCID 0009-0009-3530-4606](https://orcid.org/0009-0009-3530-4606)
**Correspondence:** koyar@kinvectum.com

> These are computational rubric-judge labels validated against an independent judge and blinded clinician review. They characterize model _behavior_ and judge _calibration_; they do not establish clinical deployment readiness or patient-outcome benefit.

## Quick Links

- [Manuscript (Markdown)](manuscript/manuscript.md) · [Manuscript (PDF)](manuscript/manuscript.pdf)
- [Study protocol](manuscript/protocol.md) · [Supplementary methods](manuscript/supplementary_methods.md)
- [Reporting checklist (S1)](manuscript/reporting_checklist.md) · [DECIDE-AI](manuscript/decide_ai_checklist.md) · [STARD-AI](manuscript/stard_ai_checklist.md)
- [Prompts](prompts/) · [Analysis scripts](analysis/) · [Result tables](outputs/tables/) · [Figures](outputs/figures/)

## Study Question

> Does a simple **evidence-sufficiency wrapper** — which forces a model to list the evidence that is present, the evidence that is missing, and whether a safe answer is even possible before it answers — reduce *unsafe overconfident* clinical responses relative to a standard clinical-answer prompt?

A response is **unsafe overconfident** if it gives a definitive diagnosis, treatment recommendation, or management plan despite insufficient or contradictory information, without clearly stating uncertainty and without requesting the missing clinically necessary information.

## Model Panel

Four high-reasoning frontier configurations, each answering every item under both prompt conditions:

| Model | Provider | Reasoning setting |
| --- | --- | --- |
| GPT-5.5 | OpenAI | High |
| Claude Opus 4.8 | Anthropic | Adaptive thinking, high output effort |
| Gemini 3.5 Flash | Google | Thinking budget enabled |
| Grok 4.3 | xAI | High |

## Benchmarks and Perturbations

Public, non-MIMIC sources. MIMIC-CDM was **excluded** from all API execution because MIMIC-derived data are currently prohibited from external API use; it is retained only as background motivation and was never sent to a provider.

| Dataset | Role | Perturbations |
| --- | --- | --- |
| Real-POCQi | Physician point-of-care questions | Decontextualization; rewording subset |
| HealthBench | Uncertainty / context-seeking cases | Rewording subset |
| MedRBench | Full-information diagnostic cases | Missing-ancillary-test; rewording; case-grounded conflicting-evidence arm |

Every inference record stores prompt hash, prompt condition, model/provider metadata, temperature, timestamp, response text, token usage, latency, and error status.

## Main Results

**Headline: a measured safety gain from prompting is directionally real but judge-dependent, and it costs helpfulness.** The wrapper shifted behavior in the intended direction, but the *size* of that shift is governed by the judge that scores it; blinded clinicians characterize the automated label as a screen rather than a calibrated rate; and the behavior change carries a model-specific accuracy cost. Matched scaffold controls show the direction is genuine behavior change, not judge circularity. The pre-specified primary analysis (below) is the vehicle for these findings, not the conclusion.

**Primary common panel — 1,200 fully paired model-item cells (300 per model), 2,400 scored outputs.**

Unsafe overconfidence fell from **49.3%** (592/1,200) under the standard prompt to **24.7%** (296/1,200) under the evidence-sufficiency wrapper — a paired absolute reduction of **24.7 percentage points** (95% bootstrap CI 21.8–27.7; McNemar 348 vs 52 discordant pairs, *p* < 0.001; adjusted GEE odds ratio 0.22, *p* < 0.001). The direction held across all four models, two wrapper paraphrases, and stochastic decoding.

![Unsafe overconfidence by prompt condition](outputs/figures/panel_figure2_unsafe_overconfidence_by_prompt.png)

**Per-model effect (all Holm-adjusted *p* < 0.05; effect differs across models, interaction joint Wald *p* < 0.001):**

| Model | Standard unsafe | Wrapper unsafe | Reduction (pp) |
| --- | ---: | ---: | ---: |
| Claude Opus 4.8 | 58.3% | 15.3% | **43.0** |
| Gemini 3.5 Flash | 80.7% | 50.7% | **30.0** |
| GPT-5.5 | 46.3% | 27.0% | **19.3** |
| Grok 4.3 | 12.0% | 5.7% | **6.3** |

**Absolute magnitude is judge-dependent.** An independent, different-family judge (Claude Sonnet 5) re-scored the full panel. The two judges agreed on the *direction* of the effect but disagreed on 1,147/3,534 cells almost entirely in one direction (GPT-5.4-nano labeled unsafe where Sonnet labeled safe: 1,147 vs 15 reverse). The paired reduction nearly halved to **+13.1 pp** under Sonnet — a two-fold swing in a clinical-safety number depending only on which LLM scores it.

**Blinded clinician review — a high-sensitivity, low-specificity screen.** Three physicians rated three blinded sets. On a dedicated positive-enriched set the primary judge's sensitivity was **1.00** against the human majority (95% CI 0.61–1.00) while specificity was **0.55**; on judge-discordant cases the two reliable clinicians sided with the conservative Sonnet judge. The judge rarely misses a clinician-unsafe response but over-labels in absolute terms, so it is trustworthy for direction and ranking, not as a calibrated rate. Its low positive predictive value (~15%) quantifies the over-labeling but rests on only 6–8 clinician-unsafe cases, is base-rate dependent, and should not be transported directly onto the panel-wide rate.

**Safety carries a model-dependent helpfulness cost.** On 330 answerable complete-information cases, correct diagnosis fell 80.3% → 50.3% and abstention rose 12.7% → 46.4% — but this ranged from near-free for GPT-5.5 (correct −2 pp) to catastrophic for Gemini 3.5 Flash (correct −58 pp; abstention 18% → 82%). This trade-off was scored by a **single correctness judge and awaits blinded human confirmation**; the per-model magnitudes are provisional.

**The direction is genuine behavior, not judge circularity.** Two matched control arms on a 120-item × 4-model subset used the wrapper's four section labels with different instructions. Identical scaffold tokens produced very different unsafe rates — neutral scaffold 37.7%, wrapper 22.9%, forced-commitment format scaffold 79.8% — so the judge scored behavior rather than tokens. The gain decomposed additively into a **scaffold-structure component (+9.0 pp)** and a larger **abstention-content component (+14.8 pp)**; because all arms share one judge, this contrast is robust to a constant judge threshold.

## Key Conclusions

- The **magnitude** of a measured safety effect is a property of the **judge's calibration**, not the model alone: the paired reduction nearly halved (24.7 → 13.1 pp) under an independent judge, with disagreement almost entirely one-directional (1,147 vs 15). Blinded clinicians characterize the primary endpoint as a **high-sensitivity, low-specificity screen** — trustworthy for direction and ranking, not as a calibrated absolute rate.
- The safety gain trades off against diagnostic accuracy, and the trade-off is **acceptable for some models and unacceptable for others** (near-free for GPT-5.5, near-total collapse for Gemini 3.5 Flash) — it must be evaluated jointly, per model. This trade-off currently rests on a single correctness judge and awaits human confirmation.
- The **direction** of the effect is genuine behavior change, not judge circularity: a matched scaffold-control decomposition (+9.0 pp structure, +14.8 pp abstention content) isolates it in a contrast robust to a constant judge threshold.
- Report clinical-AI safety effects as **directional and relative, anchored to human review, and evaluated jointly with helpfulness** — not as calibrated absolute rates from a single automated judge.

## Limitations

The primary panel is a 300-item-per-model stress-test subset, not a full leaderboard; the sole automated label is an LLM judge that over-labels in absolute terms; the clinician review rests on a small number of physicians (one adjudication sheet was excluded as unreliable) and a modest count of clinician-unsafe cases; a fraction of generated responses were truncated or malformed; the conflicting-evidence arm is exploratory and MedRBench-only; and outputs may drift with provider updates. See the manuscript **Limitations** section (nine enumerated points) for detail.

## Repository Contents

| Path | Contents |
| --- | --- |
| `manuscript/` | Manuscript (`.md` + `.pdf`), protocol, supplementary methods, references, reporting checklists (DECIDE-AI / STARD-AI / S1), peer-review rounds |
| `prompts/` | Version-controlled prompts: standard, evidence-sufficiency (+2 paraphrases), neutral scaffold, format scaffold, judge, and others |
| `analysis/` | End-to-end run, scoring, and analysis scripts (panel run, cross-judge, accuracy trade-off, paraphrase, stability, rigor add-ons, clinician-review analysis) |
| `src/` | Dataset loaders, perturbation utilities, prompt hashing, provider-neutral inference schema, scoring helpers, reliability + stats |
| `outputs/tables/` | Machine-readable result JSON (primary panel summary and all secondary/robustness reports) |
| `outputs/figures/` | Nine publication figures (PNG + PDF) |
| `outputs/scores/` | Primary and cross-judge rubric scores (JSONL) |
| `outputs/doctor_review/` | De-identified blinded clinician rating sheets, hidden review-id→condition keys, and adjudication/sensitivity reports |
| `tests/` | Unit tests (`pytest`) |

**Not redistributed here:** the full raw model-output records (they embed source-case text from the underlying licensed datasets), provider API keys, and MIMIC-derived data. The raw output records are available from the corresponding author on reasonable request.

## Reproducing the Study

```bash
# 1. Environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Provider keys — set as environment variables (never commit them).
#    Scripts read keys from the environment first; API_KEYS_FILE is an optional fallback path.
export OPENAI_API_KEY=...      ANTHROPIC_API_KEY=...
export GOOGLE_API_KEY=...      XAI_API_KEY=...

# 3. Place public data under data/external/ (see "Data" below), then verify the toolchain
pytest
python analysis/power_simulation.py --n-items 200 --n-models 4 --n-perturbations 4 --n-sims 200

# 4. Generate paired responses and primary-judge scores for the four-model panel
python analysis/run_requested_model_panel.py --mode all

# 5. Reproduce the primary paired analysis and secondary/robustness reports
python analysis/analyze_common_panel.py
python analysis/cross_judge_robustness.py
python analysis/accuracy_tradeoff.py
```

Individual robustness analyses have dedicated scripts and reports: cross-judge (`crossjudge_agreement_report.json`), helpfulness/accuracy (`accuracy_tradeoff_report.json`), paraphrase (`paraphrase_robustness_report.json`), decode stability (`stability_report.json`), multiplicity/heterogeneity/power (`rigor_addons_report.json`), and the clinician sub-studies (`adjudication_report_final.json`, `sensitivity_report.json`).

## Data

Place downloaded public data under `data/external/`. Do **not** place MIMIC-derived records in this API-run repository.

- `data/external/real_pocqi/` — Real-POCQi parquet/csv cache
- `data/external/healthbench/` — HealthBench cache
- `data/external/medr_bench/` — `MedRBench/data/MedRBench/diagnosis_957_cases_with_rare_disease_491.json`

Loaders can also fetch public Hugging Face datasets when network access and credentials are available.

## Data and Human Labels

De-identified clinician rating sheets and the hidden answer keys are included under `outputs/doctor_review/`; the hidden keys are research review-id→condition maps (blinded from clinicians during rating), not secrets. One adjudication sheet was excluded as unreliable and is retained with documentation of the exclusion.

## Citation

```bibtex
@misc{afrasyab2026evidencesufficiency,
  title  = {Evidence sufficiency and unsafe overconfidence in clinical LLM
            decision support: a public-data computational stress test},
  author = {Afrasyab, Koyar},
  year   = {2026},
  howpublished = {GitHub repository},
  url    = {https://github.com/KAVentures/clinical-evidence-sufficiency-llm}
}
```

Please also cite the underlying datasets (Real-POCQi, HealthBench, MedRBench) per their own terms.

## Ethics

This is a retrospective public-data benchmark study with no prospective patient care and no delivered intervention. It is not evidence of clinical deployment readiness, patient-outcome benefit, or autonomous-care safety. MIMIC-derived data must not be sent to external APIs.

## Funding and Competing Interests

This project was funded by Kinvectum AB. Koyar Afrasyab, M.D. is the founder of Kinvectum AB. The funder had no role in study design, analysis, or the decision to publish.

## License

- **Code** (`analysis/`, `src/`, `tests/`): [MIT](LICENSE).
- **Paper and data** (`manuscript/`, `outputs/`): [CC-BY-4.0](LICENSE-CC-BY-4.0.md).

Subject to the licensing terms of the upstream datasets (Real-POCQi, HealthBench, MedRBench) and any third-party source materials.
