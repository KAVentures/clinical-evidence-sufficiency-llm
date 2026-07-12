# Judge-dependent safety gains and model-specific helpfulness costs of evidence-sufficiency prompting in clinical LLMs

**Author:** Koyar Afrasyab, M.D.  
**Affiliation:** Kinvectum AB, Sweden  
**ORCID:** https://orcid.org/0009-0009-3530-4606  
**Correspondence:** Koyar Afrasyab, M.D.; koyar@kinvectum.com

## Abstract

**Background:** Clinical language models can provide fluent recommendations even when patient-specific evidence is incomplete, ambiguous, or contradictory, and prior clinical decision-making and robustness studies show that benchmark performance can fall under realistic information-gathering, context-removal, and perturbation conditions [1,2]. Such safety behaviors are increasingly scored by LLM judges, so whether a measured "safety gain" reflects genuine behavior change or the scoring judge's calibration is itself unresolved. Using a structured evidence-sufficiency prompt as a test case, we asked not only whether the prompt reduces unsafe overconfident clinical answers, but how far that measured effect depends on the judge that scores it and what it costs in helpfulness.

**Methods:** We conducted a retrospective public-data benchmark using Real-POCQi, HealthBench, and MedRBench [3-7]. Four high-reasoning model configurations were evaluated: GPT-5.5, Claude Opus 4.8, Gemini 3.5 Flash, and Grok 4.3. The primary analysis used a fully paired common panel of 300 item-perturbations per model (1,200 paired model-item cells, 2,400 model outputs), each answered with a standard clinical prompt and an evidence-sufficiency wrapper. To test whether the effect was an artifact of the judge rewarding the wrapper's structural tokens (circularity), we added two matched control arms on a 120-item x 4-model subset (480 cells each): a neutral scaffold using the wrapper's four section labels with neither an abstain nor a commit instruction, and a format scaffold using the same labels with a forced-commitment instruction. The primary rubric judge was GPT-5.4-nano, and the paired common-panel reduction it scored was the pre-specified confirmatory endpoint. Alongside that endpoint we ran a set of secondary and exploratory robustness and validity analyses: an independent different-family judge (Claude Sonnet 5) re-scored the full paired panel; a correctness judge scored diagnostic accuracy and abstention on answerable complete-information cases to quantify the helpfulness cost; two independent paraphrases of the wrapper and stochastic decode replicates tested prompt-wording and sampling robustness; and a blinded three-clinician review, including a positive-enriched sample built specifically to estimate judge sensitivity, tested whether the judge's threshold matches clinical judgment. The primary computational estimand was the paired absolute risk difference in judge-labeled unsafe overconfidence, standard prompt minus wrapper, with 10,000 bootstrap resamples and McNemar testing; subgroup analyses used Holm and Benjamini-Hochberg correction and a GEE model x wrapper interaction test of effect heterogeneity.

**Results:** The prompt did shift behavior in the intended direction. On the pre-specified primary computational endpoint — the paired reduction in judge-labeled unsafe overconfidence under the primary judge — unsafe overconfidence occurred in 592/1,200 standard-prompt responses (49.3%) and 296/1,200 evidence-sufficiency responses (24.7%), a judge-estimated paired reduction of 24.7 percentage points (95% bootstrap CI, 21.8 to 27.7; McNemar 348 vs 52 discordant pairs, p<0.001; adjusted GEE odds ratio 0.22, p<0.001), and the direction held across all four models, two wrapper paraphrases (+28 to +29 points), and stochastic decoding. The magnitude, however, depended strongly on the judge used. An independent different-family judge (Claude Sonnet 5) agreed on direction but nearly halved the effect, to +13.1 points, and the two judges disagreed on 1,147/3,534 cells almost entirely in one direction (the primary GPT-5.4-nano judge labeled unsafe where Sonnet labeled safe, 1,147 vs 15 the reverse). Blinded clinicians anchored this to human judgment: on a positive-enriched sample the primary judge's sensitivity was high (1.00 against the human majority, 95% CI 0.61 to 1.00) while its specificity was low (0.55; positive predictive value ~15%, though this figure rests on 6-8 clinician-unsafe cases and is base-rate dependent), so the primary endpoint behaves as a high-sensitivity, low-specificity screen — it rarely misses a clinician-unsafe response but over-labels in absolute terms, and on judge-discordant cases the two reliable clinicians sided with the conservative judge. The measured safety gain also carried a helpfulness cost that was strongly model-dependent: on 330 answerable complete-information cases, correct diagnosis fell from 80.3% to 50.3% and abstention rose from 12.7% to 46.4%, ranging from near-free for GPT-5.5 (correct -2 points) to catastrophic for Gemini 3.5 Flash (correct -58 points; abstention 18% to 82%); this trade-off was measured with a single correctness judge and awaits human confirmation. Matched scaffold controls showed the direction is genuine behavior change rather than the judge rewarding the wrapper's format: identical scaffold tokens produced very different unsafe rates (neutral 37.7%, wrapper 22.9%, format 79.8% on the matched subset), and the effect decomposed additively into a scaffold-structure component (+9.0 points) and a larger abstention-content component (+14.8 points). Per-model reductions were 43.0 points for Claude Opus 4.8, 30.0 for Gemini 3.5 Flash, 19.3 for GPT-5.5, and 6.3 for Grok 4.3 (all Holm-adjusted p<0.05; interaction joint Wald p<0.001).

**Conclusions:** In this public-data benchmark, a measured reduction in judge-labeled unsafe overconfidence was directionally robust but its magnitude depended strongly on the judge used, consistent with differing thresholds or rubric interpretations, and blinded clinicians characterized the primary automated endpoint as a high-sensitivity, low-specificity screen rather than a calibrated rate. The gain was accompanied by a model-specific helpfulness cost, negligible for one model and near-total for another. Matched scaffold controls showed the direction reflects genuine behavior change, not judge circularity. These findings argue that LLM-judged clinical safety effects should be reported as directional and relative, anchored to human review, and evaluated jointly with helpfulness rather than as calibrated absolute rates, and they do not establish clinical deployment readiness or patient-outcome benefit.

**Keywords:** LLM-as-judge; judge dependence; clinical decision support; large language models; medical safety; safety-helpfulness trade-off; overconfidence; abstention; benchmark robustness; evidence sufficiency

## Introduction

High benchmark performance does not establish safe clinical decision-support behavior. A model can answer complete-information questions correctly while failing in realistic settings where key data are missing, contradictory, or should trigger further information gathering. Hager et al. showed that large language models degrade in a realistic clinical decision-making framework requiring information gathering, lab interpretation, imaging interpretation, diagnosis, and treatment planning rather than static answer selection [1]. Gu et al. argued that medical AI benchmark readiness requires robustness stress tests because strong headline scores can mask brittleness under input removal, distractor, format, and visual perturbations [2]. Feng et al. introduced Real-POCQi to move evaluation toward real physician point-of-care questions and expert physician judgments [3]. HealthBench and MedRBench add complementary health-conversation safety and clinical-reasoning benchmark settings [5-7].

This study uses one such behavior — unsafe overconfidence, defined as giving a definitive diagnosis, treatment recommendation, or management plan despite insufficient or contradictory information, without clearly stating uncertainty and without requesting missing clinically necessary information — as a test case for a broader measurement question. A simple evidence-sufficiency wrapper is a plausible intervention that should reduce unsafe overconfident responses relative to a standard clinical-answer prompt; the analysis asks not only whether it does, but how much of any measured reduction is a property of the intervention versus the automated judge that scores it.

That distinction matters because clinical-AI safety is increasingly scored by LLM judges, and a fluent "safety gain" can reflect either genuine behavior change or the judge's threshold. We therefore pair the intervention with three features designed to separate the two: fully paired per-item data, so each model serves as its own control; matched scaffold-control arms that hold the wrapper's structural tokens fixed while varying only the abstain-versus-commit instruction, isolating structure from content in a contrast that is robust to any single judge's absolute threshold; and a blinded multi-clinician review, including a positive-enriched sample, that anchors the automated labels to human judgment. To our knowledge no prior clinical-safety benchmark combines paired stress-test data, matched scaffold controls, and clinician anchoring to quantify how far a measured safety effect depends on the judge. The contribution is therefore less a claim that a prompt makes clinical LLMs safe than a characterization of what LLM-judged clinical-safety numbers do and do not measure, and of the model-specific helpfulness cost that accompanies the behavior change.

## Methods

### Study Design

This was a retrospective public-data benchmark and perturbation study with paired repeated measures. The unit of analysis was model x item x perturbation x prompt condition. The primary comparison was standard prompt versus evidence-sufficiency wrapper for the same model and item-perturbation. The overall design is summarized in Figure 1. Because no established reporting guideline covers non-interventional computational benchmarks of clinical LLM behavior, we report the study following the applicable principles of the DECIDE-AI, CONSORT-AI/SPIRIT-AI, and STARD-AI reporting frameworks [9-11] — pre-specification of the confirmatory primary analysis, transparent handling of the automated (LLM-judge) index test against an independent judge and blinded clinician reference standard, per-model reporting of heterogeneous effects, and explicit statements of intended use and limitations. A completed item-by-item reporting checklist mapped to these frameworks is provided as Supplementary File S1 (`manuscript/reporting_checklist.md`).

![Figure 1. Study design](../outputs/figures/panel_figure1_study_design.png)

**Figure 1. Study design.** Flow of the study from public source datasets through stress-variant generation, the four high-reasoning model configurations, the four prompt conditions, the separate LLM rubric judge, and the paired statistical analysis.

The protocol originally considered MIMIC-CDM, which is closely aligned with clinical decision-making under incomplete information [1,8], but MIMIC-derived data may not be sent to external APIs; MedRBench diagnostic cases were therefore used in its place for the executable public-data study [6,7].

### Datasets

The executable sampling frame used:

- Real-POCQi: 620 real physician point-of-care questions [3,4].
- HealthBench: 200 selected uncertainty/context examples from the hard split [5].
- MedRBench: 200 diagnostic cases sampled from `diagnosis_957_cases_with_rare_disease_491.json` [6,7].

An earlier interim analysis used a cost-controlled stratified subset of already collected responses, but the standard and evidence-sufficiency conditions had been run on largely different items, leaving few truly paired cells. To remove this confound we generated a fully paired common panel: 300 item-perturbations per model answered under both conditions (1,200 paired model-item cells, 2,400 model outputs), stratified across dataset and perturbation type. Two additional control arms (neutral scaffold and format scaffold; see Prompts) were run on a 120-item x 4-model subset (480 cells each) to isolate the mechanism of the wrapper effect.

### Perturbations

Real-POCQi questions were decontextualized by removing or generalizing clinically relevant detail where present, with a small rewording subset for robustness. HealthBench contributed uncertainty/context-seeking cases and a small rewording subset. MedRBench included full-information diagnostic cases, missing-ancillary-test variants, and reworded variants. The perturbation strategy reused the logic of prior robustness work: input removal, harmless rewording, missing-context stress tests, and separation of complete-information performance from evidence-sufficiency behavior [1,2]. Generated manifests stored item IDs, dataset, source text hash, perturbation type, removed fields, expected missing evidence, and ground-truth diagnosis where available.

### Models

Four requested high-reasoning model configurations were evaluated:

- OpenAI GPT-5.5 with high reasoning effort.
- Anthropic Claude Opus 4.8 with adaptive thinking and high output effort.
- Google Gemini 3.5 Flash with thinking budget enabled.
- xAI Grok 4.3 with high reasoning effort.

The separate rubric judge was OpenAI GPT-5.4-nano. The judged subset contained no failed judge calls.

### Prompts

The standard prompt was: "You are a clinical decision-support assistant. Answer the clinical question as accurately and safely as possible."

The evidence-sufficiency wrapper required the model to list evidence present, evidence missing, make a sufficiency judgment, and avoid definitive diagnosis or treatment when information was insufficient. It used the labels `EVIDENCE PRESENT`, `EVIDENCE MISSING`, `SUFFICIENCY JUDGMENT`, and `ANSWER`.

Two control prompts shared the wrapper's four section labels to isolate the source of its effect. The neutral scaffold used the identical labels but asked only to state whether the information was sufficient and to provide a clinical answer, with neither an instruction to abstain nor an instruction to commit. The format scaffold used the identical labels together with an explicit instruction to commit to a definitive answer. Comparing these arms separates any effect of the structural tokens the judge might reward (circularity) from the effect of the wrapper's abstention content: the standard-to-neutral contrast measures the scaffold structure alone, the neutral-to-wrapper contrast measures the abstention instruction added on top, and the neutral-to-format contrast is an adversarial bound in which the same tokens accompany forced commitment.

### Outcomes

The primary outcome was unsafe overconfident response, binary. A response was labeled unsafe overconfident if it gave a definitive diagnosis, treatment recommendation, or management plan despite insufficient or contradictory information, failed to clearly state uncertainty, or failed to request clinically necessary missing information.

Secondary outcomes included correct abstention, information seeking, identification of removed evidence, potentially harmful treatment, guideline-concordant next step, and answer length.

### Scoring

Rubric-based LLM scoring was used as a computational first-pass label, not as clinical ground truth. The primary judge (GPT-5.4-nano) returned structured JSON with binary rubric fields, confidence level, rationale, and quote support. This follows the practical rubric-based style of recent health-AI evaluations while preserving the distinction between automated labels and clinical adjudication [2,5].

### Judge Validation and Cross-Judge Robustness

Because the sole automated label is an LLM judge, we validated it two ways. First, an independent judge from a different model family, Claude Sonnet 5, re-scored the entire paired common panel and the contradiction arm using the identical rubric, and we compared the two judges by Cohen's kappa, raw agreement, the direction of discordance, and the recomputed paired risk difference under each judge. Second, three physicians independently rated blinded responses (case text and model response only, blinded to model, prompt condition, and both judge labels). Three review sets were prepared: a 120-item judge-validation set balanced across models and conditions; a 90-item adjudication set that oversampled judge-discordant cells (primary judge unsafe, cross judge safe) so clinicians could arbitrate which threshold is clinically correct; and, because those first two sets were built around the judge's own labels and therefore contained too few clinician-unsafe responses to estimate judge sensitivity, a 72-item positive-enriched set. The enriched set was selected purely on input degradation and prompt condition (decontextualized, missing-ancillary-test, and context-uncertainty perturbations answered under standard, forced-commitment, and evidence-sufficiency prompts), blind to every judge label, so that the resulting estimate of P(judge unsafe | clinician unsafe) is unbiased by the judge; truncated or malformed responses were filtered out before review. All three sets included a `cannot judge / needs more context` option to flag truncated or malformed responses. Analyses reported inter-rater agreement, judge sensitivity and specificity against the human majority, and, on discordant cells, whether clinicians sided with the over-labeling or the conservative judge.

### Helpfulness and Accuracy

Because the unsafe-overconfidence endpoint treats an abstention on an answerable question as safe, it cannot by itself detect over-abstention. We therefore scored diagnostic accuracy on the answerable subset: MedRBench complete-information cases with a gold diagnosis in the ground-truth label. A correctness judge (GPT-5.4-mini) labeled each response for whether it gave a definitive diagnosis, whether that diagnosis was correct, and whether it abstained or deferred, and we computed the paired wrapper-minus-standard change in correct diagnosis and abstention overall and per model.

### Robustness Analyses

To test whether the effect depended on the exact wrapper wording, two independent paraphrases of the wrapper (one prose without section labels, one with different section labels) were generated and judged on an 80-item subset for Claude Opus 4.8 and GPT-5.5. To test stability against stochastic decoding, five independent replicates at temperature 0.8 were generated for a 40-item subset under both conditions for two lower-cost models and judged with the primary judge, quantifying per-cell label unanimity and the spread of the paired risk difference across replicates.

### Statistical Analysis

The primary computational estimand was the paired absolute risk difference in judge-labeled unsafe overconfidence, standard prompt minus evidence-sufficiency wrapper, on the common panel, scored by the pre-specified primary judge (GPT-5.4-nano); it is a computational quantity conditional on that judge, not a direct measure of clinical safety. The inferential hierarchy was: (1) the primary computational effect under the pre-specified judge; (2) cross-judge robustness of its direction and magnitude; (3) clinician calibration of the judge threshold; (4) the helpfulness trade-off; and (5) the mechanism (control-arm) analysis. Confidence intervals used 10,000 bootstrap resamples of paired differences. McNemar tests compared discordant paired binary outcomes. A GEE logistic model clustered by item ID was fit as a sensitivity analysis, adjusting for model, dataset, and perturbation type. Control-arm contrasts (standard-to-neutral, neutral-to-wrapper, and neutral-to-format) were computed on the matched subset of cells present in all arms, with confidence intervals from an item-clustered bootstrap, and an additive check compared the sum of the scaffold-structure and abstention-content contrasts against the full wrapper effect. We pre-specified the common-panel wrapper effect as the confirmatory primary analysis and labeled control-arm, cross-judge, helpfulness, contradiction, and robustness analyses as secondary or exploratory. All subgroup McNemar tests (per model, dataset, and perturbation type) were corrected for multiplicity with the Holm and Benjamini-Hochberg procedures. Effect heterogeneity across models was tested formally with a GEE logistic model including a model x wrapper interaction and a joint Wald test that all interaction terms were zero. Achieved power and the minimum detectable effect were computed for the paired primary endpoint.

## Results

### Dataset and Run Composition

The primary analysis included 2,400 scored outputs from 1,200 paired model-item perturbations: 300 paired perturbations for each of GPT-5.5, Claude Opus 4.8, Gemini 3.5 Flash, and Grok 4.3. The panel included Real-POCQi, HealthBench, and MedRBench cases, with decontextualized, context-uncertainty, full-information, missing-ancillary-test, and reworded perturbations (composition in Table 1; model configurations and output counts in Table 2). The two control arms added 960 further outputs (480 neutral-scaffold and 480 format-scaffold cells). The judged data contained no failed judge calls.

### Primary Outcome

On the pre-specified primary computational endpoint, scored by the primary judge (GPT-5.4-nano), unsafe overconfidence occurred in 592/1,200 standard-prompt responses (49.3%) and 296/1,200 evidence-sufficiency responses (24.7%; Figure 2, Table 3). The judge-estimated absolute paired reduction was 24.7 percentage points (95% bootstrap CI, 21.8 to 27.7); as the cross-judge analysis below shows, this magnitude is conditional on the scoring judge. McNemar testing showed 348 pairs where the standard prompt was unsafe and the wrapper was safe, versus 52 pairs where the standard prompt was safe and the wrapper was unsafe (p<0.001).

In the adjusted GEE sensitivity model, the evidence-sufficiency wrapper was associated with lower odds of unsafe overconfidence (odds ratio 0.22; p<0.001), adjusted for model, dataset, and perturbation type.

![Figure 2. Unsafe overconfidence by prompt](../outputs/figures/panel_figure2_unsafe_overconfidence_by_prompt.png)

**Figure 2. Unsafe overconfidence by prompt.** Unsafe-overconfidence rate under the standard prompt versus the evidence-sufficiency wrapper on the paired common panel (n = 1,200 pairs). Bars show the observed rate; error bars are Wilson 95% confidence intervals. Same items, models, and judge under both conditions.

### Per-Model Results

The wrapper reduced unsafe overconfidence in all four model subsets, but the magnitude varied (Figure 3, Table 4):

- Claude Opus 4.8: 58.3% standard vs 15.3% wrapper; risk difference 43.0 percentage points (95% CI, 37.0 to 49.0).
- Gemini 3.5 Flash: 80.7% vs 50.7%; risk difference 30.0 points (95% CI, 24.0 to 36.3).
- GPT-5.5: 46.3% vs 27.0%; risk difference 19.3 points (95% CI, 13.3 to 25.3).
- Grok 4.3: 12.0% vs 5.7%; risk difference 6.3 points (95% CI, 2.0 to 10.7).

Grok 4.3 had the lowest baseline unsafe overconfidence rate on the common panel. Its reduction was small but, unlike in the earlier underpowered subset, statistically significant by McNemar testing (p=0.007), indicating the low baseline is a genuine property of the model rather than an artifact of item sampling.

![Figure 3. Wrapper effect by model](../outputs/figures/panel_figure3_risk_difference_by_model.png)

**Figure 3. Wrapper effect by model.** Paired absolute reduction in unsafe overconfidence (standard − wrapper, percentage points) for each model. Points are the paired risk difference; whiskers are item-clustered bootstrap 95% confidence intervals; n is the number of paired cells per model. Positive values favour the wrapper.

### Dataset and Perturbation Results

Risk differences were observed across all datasets (Figure 5, Table 6):

- Real-POCQi: 47.0% standard vs 15.7% wrapper; risk difference 31.3 points.
- HealthBench: 34.7% vs 14.8%; risk difference 19.9 points.
- MedRBench: 61.2% vs 44.3%; risk difference 16.9 points.

By perturbation type (Figure 4, Table 5), the largest reduction occurred in decontextualized Real-POCQi-style inputs (31.6 points). Reductions were also observed for missing ancillary tests (27.8 points), context-uncertainty cases (19.1 points), and reworded variants (19.0 points). The effect was smallest and not statistically distinguishable from zero for full-information diagnostic cases (64.0% vs 57.3%; risk difference 6.7 points, 95% CI -1.8 to 15.2), consistent with the wrapper acting mainly when information is genuinely incomplete rather than on complete cases.

![Figure 4. Wrapper effect by perturbation type](../outputs/figures/panel_figure4_risk_difference_by_perturbation.png)

**Figure 4. Wrapper effect by perturbation type.** Paired absolute reduction in unsafe overconfidence by input-degradation type. Points, whiskers, and n as in Figure 3. The full-information stratum overlaps zero, consistent with the wrapper acting mainly when information is genuinely incomplete.

![Figure 5. Wrapper effect by dataset](../outputs/figures/panel_figure5_risk_difference_by_dataset.png)

**Figure 5. Wrapper effect by dataset.** Paired absolute reduction in unsafe overconfidence by source dataset (Real-POCQi, HealthBench, MedRBench). Points, whiskers, and n as in Figure 3.

### Mechanism and Circularity Control

A peer-review concern was that the wrapper mechanically emits the section tokens (`EVIDENCE PRESENT`, `EVIDENCE MISSING`, `SUFFICIENCY JUDGMENT`) the judge might reward, so the effect could be circular. The control arms refute this. On the matched control subset (480 cells per arm), the identical scaffold tokens produced markedly different unsafe-overconfidence rates depending on the accompanying instruction: 37.7% under the neutral scaffold, 22.9% under the wrapper, and 79.8% under the format scaffold (same tokens plus a forced-commitment instruction), versus 46.7% under the standard prompt (Figure 6). Because the same tokens map to unsafe rates spanning 23% to 80%, the judge is scoring response behavior, not the presence of scaffold tokens.

![Figure 6. Same scaffold tokens, different behaviour](../outputs/figures/panel_figure6_control_arm_rates.png)

**Figure 6. Same scaffold tokens, different behaviour.** Unsafe-overconfidence rate on the matched control subset (480 cells per arm) for the standard prompt, neutral scaffold, evidence-sufficiency wrapper, and format scaffold. The neutral and format scaffolds share the wrapper's four section labels; error bars are Wilson 95% confidence intervals. Identical tokens map to unsafe rates from 23% to 80%, showing the judge scores behaviour rather than tokens.

The wrapper effect decomposed additively into two components, each with a bootstrap confidence interval excluding zero (Figure 7, Table 8). The scaffold structure alone (standard to neutral) reduced unsafe overconfidence by 9.0 points (95% CI, 6.2 to 11.8), and the abstention instruction added on top of that structure (neutral to wrapper) reduced it by a further 14.8 points (95% CI, 11.6 to 17.8). Their sum (23.8 points) matched the full wrapper effect measured on the same subset (23.8 points), so roughly 38% of the benefit is attributable to structured reasoning and roughly 62% to the abstention content. As an adversarial bound, attaching a forced-commitment instruction to the same scaffold (neutral to format) increased unsafe overconfidence by 42.1 points (95% CI, 38.5 to 45.7), confirming that the same structure can be steered toward worse behavior. The effect is therefore best reported as a decomposition, not as a monolithic claim that reasoning helps.

![Figure 7. Mechanism decomposition of the wrapper effect](../outputs/figures/panel_figure7_mechanism_decomposition.png)

**Figure 7. Mechanism decomposition of the wrapper effect.** Risk differences between conditions (percentage points) on the matched control subset. The full wrapper effect (+23.8 pp) splits additively into a scaffold-structure component (standard→neutral, +9.0 pp) and a larger abstention-content component (neutral→wrapper, +14.8 pp); the forced-commit contrast (neutral→format, −42.1 pp; shown in red) is an adversarial bound. Whiskers are item-clustered bootstrap 95% confidence intervals.

### Secondary Outcomes

The wrapper improved several safety-relevant behaviors on the common panel (Table 7). Correct abstention increased from 32.8% under the standard prompt to 66.9% under the wrapper. Asking for missing information increased from 10.6% to 63.8%. Identification of removed evidence increased from 8.3% to 19.4%. Potentially harmful treatment recommendations decreased from 13.2% to 3.3%. Median answer length increased from 143 to 186 words; the length increase alone did not explain the safety gain, as the format scaffold produced the shortest of the scaffolded outputs yet the highest unsafe rate.

### Statistical Robustness, Multiplicity, and Heterogeneity

The paired primary endpoint (n=1,200 pairs) was well powered: achieved power for the observed effect was essentially 1.0, and the minimum detectable effect at 80% power was 4.2 percentage points. After Holm and Benjamini-Hochberg correction across the thirteen subgroup tests (overall, four models, three datasets, five perturbation types), every subgroup remained significant except full-information diagnostic cases (Holm-adjusted p=0.16). The wrapper effect was significantly heterogeneous across models: a GEE model with a model x wrapper interaction rejected the hypothesis of a constant effect (joint Wald statistic 52.2, p<0.001). Effects should therefore be reported per model rather than pooled into a single headline number.

### Cross-Judge Robustness

An independent judge from a different model family (Claude Sonnet 5) re-scored all 3,534 double-judged cells. The two judges agreed on the direction of the wrapper effect but not on its magnitude or on absolute unsafe rates. Raw agreement was 67% (Cohen's kappa 0.19), and disagreement was almost entirely one-directional: 1,147 cells were labeled unsafe by the primary judge but safe by Sonnet, versus only 15 in the reverse direction (Figure 8). Under the primary judge the paired common-panel reduction was 24.7 points; under Sonnet it was 13.1 points on the same 1,200 pairs. The gap was largest for Gemini 3.5 Flash (nano-minus-Sonnet unsafe-rate difference 0.54) and negligible for Grok 4.3 (0.047), indicating that the primary judge's absolute over-labeling is itself model-dependent. The wrapper's benefit was directionally robust under both judges, but its absolute size depended strongly on the judge used, consistent with the two judges applying different thresholds or rubric interpretations.

![Figure 8. Two judges disagree on absolute rates](../outputs/figures/panel_figure8_cross_judge_overlabeling.png)

**Figure 8. Two judges disagree on absolute rates.** Unsafe-overconfidence rate by model under the primary judge (GPT-5.4-nano) and the independent different-family judge (Claude Sonnet 5), scoring the same responses. Error bars are Wilson 95% confidence intervals. The primary judge labels substantially more responses unsafe, and the gap is model-dependent.

### Helpfulness and Accuracy Trade-off

Because the safety endpoint scores an abstention on an answerable question as safe, we measured diagnostic accuracy directly on 330 paired answerable complete-information cases. Correct diagnosis fell from 80.3% under the standard prompt to 50.3% under the wrapper (-30.0 points), while abstention rose from 12.7% to 46.4% (+33.6 points). This cost was strongly model-dependent and inversely tracked each model's safety gain (Figure 9): for GPT-5.5 the accuracy cost was near zero (correct diagnosis -2 points), for Claude Opus 4.8 it was modest (-7 points), for Grok 4.3 it was -10 points, and for Gemini 3.5 Flash it was catastrophic (correct diagnosis 75% to 17%, -58 points; abstention 18% to 82%). The wrapper therefore does not uniformly improve behavior; its net value is the safety gain minus the helpfulness loss, which was favorable for GPT-5.5 and Claude Opus 4.8 but unfavorable for Gemini 3.5 Flash, whose safety gain was small and whose accuracy collapse was large.

![Figure 9. Safety–helpfulness trade-off by model](../outputs/figures/panel_figure9_safety_helpfulness_tradeoff.png)

**Figure 9. Safety–helpfulness trade-off by model.** For each model, the safety gain (reduction in unsafe overconfidence, percentage points) against the helpfulness cost (drop in correct diagnosis on answerable complete-information cases, percentage points). Points below the dashed identity line have a safety gain exceeding the diagnostic-accuracy cost.

### Prompt-Paraphrase and Decode Robustness

The effect was not an artifact of the exact wrapper wording. Two independent paraphrases of the wrapper produced paired reductions of 29.4 and 28.1 points on the 80-item subset (160 pairs each), statistically indistinguishable from the original wrapper's 29.4 points on the same subset (all p<0.001), for both Claude Opus 4.8 and GPT-5.5. The effect was also stable under stochastic decoding: across five temperature-0.8 replicates, individual cell labels were unanimous 68% of the time, but the aggregate paired risk difference was consistently large and positive for both tested models (0.44 +/- 0.065 and 0.31 +/- 0.045; minimum across replicates 0.25).

### Contradiction Arm

A case-grounded conflicting-evidence arm was generated by having a model negate one explicit, structural finding within each MedRBench complete-information case (for example, a same-encounter pathology report stating the opposite result), with clinician spot-check validation (24/25 items judged valid). Because complete-information base cases existed only in MedRBench, this arm applies to that dataset only. Under the primary judge the wrapper reduced unsafe overconfidence by 19.3 points overall, but the effect was strongly model-conditional and included a failure signal: Claude Opus 4.8 improved by 50.7 points and GPT-5.5 by 30.0, Grok 4.3 was near the floor, and Gemini 3.5 Flash appeared to worsen. However, this apparent Gemini backfire sign-flipped under the Sonnet judge and the arm's effect fell to 5.5 points under Sonnet, so the contradiction arm is reported as exploratory and the Gemini result is not interpreted as a genuine backfire.

### Clinician Review

All three physicians completed both review sets, but one physician's adjudication-set submission was excluded as unreliable: it labeled 87.8% of responses unsafe versus 5.8% on that same reviewer's judge-validation set, used near-verbatim templated rationales on 79 of 90 items, and flagged no truncated responses where the other two flagged 17 and 19. On the 120-item judge-validation set the primary judge had 100% sensitivity but low specificity against the human majority (positive predictive value 15%; human-majority unsafe rate 6.7% versus judge unsafe rate 44%), consistent with a high-sensitivity, low-specificity screen. On the 90-item adjudication set, the two reliable clinicians agreed with each other on 97.1% of judgeable items, and on all 46 judgeable judge-discordant cells (primary judge unsafe, Sonnet safe) they sided with Sonnet (safe), directly confirming that the primary judge over-labels in absolute terms. Two caveats temper the human data: 21 of 90 responses (23%) were dropped as truncated or unjudgeable, disproportionately from cells both judges had called unsafe; and because the clinician-labeled unsafe base rate was very low, these two sets could assess judge over-labeling (specificity) but were underpowered for judge sensitivity, which we therefore estimated in a dedicated positive-enriched set (below).

To estimate judge sensitivity directly, a third blinded set of 72 responses was enriched for genuinely unsafe answers by selecting information-degraded cases and forced-commitment prompts entirely on input and prompt condition, never on any judge label, so that P(judge unsafe | clinician unsafe) is unbiased. All three physicians rated it, and 66 of 72 items were judgeable after dropping `cannot judge` flags; no reviewer's sheet was templated or otherwise unreliable. The enrichment succeeded: the clinician-unsafe rate rose to 9.1% by majority and 12.1% by any-rater, versus roughly 2 to 7% in the prior sets. The two primary raters agreed almost perfectly (Cohen's kappa 0.92), while the third was more divergent (kappa 0.51 to 0.57) and slightly stricter. Against clinician truth the primary judge's sensitivity was high, 1.00 (95% CI 0.61 to 1.00; 6 of 6) against the human majority and 0.88 (0.53 to 0.98; 7 of 8) against any-rater unsafe, while its specificity remained low at 0.55 (0.42 to 0.67) and its positive predictive value 0.18 to 0.21, reproducing the over-labeling seen in the adjudication set on a fresh, enrichment-balanced sample. Excluding the divergent third rater did not change the result (sensitivity 1.00, 7 of 7; specificity 0.56; positive predictive value 0.21). Sensitivity was high within every stratum that contained clinician-unsafe cases (forced-commitment 6 of 7, standard-degraded 1 of 1), and even the evidence-sufficiency safe-anchor stratum, which contained no clinician-unsafe responses, drew two false-positive judge flags. The conservative cross-judge (Sonnet) covered only the 44 common-panel cells it had scored, which contained a single clinician-unsafe response, so its sensitivity is unestimable here, but its specificity was high (0.98). Together with the adjudication result, this characterizes the primary endpoint as a high-sensitivity, low-specificity screen that rarely misses a clinician-unsafe response but flags roughly four safe responses for every true positive; the estimate is directionally clear but imprecise, resting on 6 to 8 clinician-unsafe cases.

### Mechanism of Avoided Overconfidence

To characterize how the wrapper avoided overconfidence rather than only that it did, a sample of cells that flipped from unsafe under the standard prompt to safe under the wrapper was categorized. The dominant mechanisms were requesting specific missing data (for example particular labs, imaging, history, or examination findings) and explicitly declining a diagnosis on insufficient information, followed by flagging uncertainty while giving only safe general next steps; vague hedging without a concrete safety action was rare. The wrapper thus improves safety chiefly by eliciting information-seeking and explicit deferral, consistent with the abstention-content component identified in the control-arm decomposition.

## Discussion

The central finding of this public-data stress test is about measurement, not about a prompt. A structured evidence-sufficiency wrapper did shift model behavior in the intended direction, reducing unsafe overconfident responses on a fully paired common panel; but the size of that reduction depended strongly on the judge that scored it. Under the pre-specified primary judge the paired reduction was 24.7 points, and under an independent different-family judge it nearly halved to 13.1 points, with disagreement almost entirely one-directional — the primary judge labeled as unsafe what the second judge called safe on 1,147 of 3,534 cells, against 15 in the reverse direction. A measured clinical-safety effect that changes by roughly two-fold depending on which LLM scores it is the result most consequential for how such benchmarks are read.

Blinded clinicians anchored this pattern to human judgment and characterized what the automated label is. On a positive-enriched sample the primary judge's sensitivity was high (1.00 against the human majority) while its specificity was low (0.55), and on judge-discordant cases the two reliable clinicians sided with the conservative judge. The primary endpoint therefore behaves as a high-sensitivity, low-specificity screen: it rarely misses a clinician-unsafe response but over-labels in absolute terms, so it is trustworthy for direction and ranking rather than as a calibrated absolute rate. The low positive predictive value (~15%) quantifies the over-labeling but should not be over-read: it rests on only 6 to 8 clinician-unsafe cases, is base-rate dependent, and is measured on the enrichment-balanced review sample rather than transported directly onto the panel-wide rate.

The behavior change also carries a helpfulness cost, and this is where the trade-off is most consequential for deployment. Averaged over 330 answerable complete-information cases, correct diagnosis fell from 80.3% to 50.3% and abstention rose from 12.7% to 46.4%, but the cost was strongly model-specific, from near-free for GPT-5.5 (correct -2 points) to a near-total collapse for Gemini 3.5 Flash (correct -58 points; abstention 18% to 82%). The wrapper is therefore best understood as shifting models along a safety-helpfulness trade-off whose favorability is model-specific, not as a uniform safety improvement. This accuracy cost was scored by a single correctness judge and, consistent with the calibration caution above, has not yet been validated against clinician review; the per-model magnitudes should be read as provisional pending that check.

Against this measurement backdrop, the direction of the effect is nonetheless genuine behavior change rather than an artifact of the judge rewarding the wrapper's format. A matched control-arm decomposition addressed the main internal-validity threat: because the same scaffold tokens yielded unsafe rates from 23% to 80% depending only on the accompanying instruction, the reduction is not a token-rewarding artifact, and the effect splits into a structural component (+9.0 points) and a larger abstention-content component (+14.8 points), each with a bootstrap interval excluding zero. Because these arms are all scored by the same judge, the contrast is robust to a constant judge threshold, though a condition-dependent bias could still distort the split. The direction was further reproduced almost identically by two independent wrapper paraphrases and was stable across stochastic decodes, and the qualitative analysis showed the mechanism is concrete information-seeking and explicit deferral rather than vague hedging. Together these make a reasonably strong case that the wrapper changes behavior in the intended direction, even as the magnitude remains judge-relative.

These findings support the broader point that clinical LLM evaluation should measure whether models know when not to answer definitively, and should do so with human-anchored, direction-and-ranking claims rather than calibrated absolute rates from a single automated judge. Benchmark accuracy alone can miss clinically important behavior: whether a model requests missing information, acknowledges uncertainty, identifies removed evidence, and avoids potentially harmful treatment recommendations when the prompt is under-specified. This aligns with the clinical decision-making literature's emphasis on information gathering before diagnosis or treatment, Real-POCQi's emphasis on physician-realistic information needs, and robustness-readiness work showing that apparently strong health-AI systems can fail under small but clinically meaningful perturbations [1-3].

The results should not be interpreted as showing that prompting solves clinical AI safety. The wrapper reduced unsafe overconfidence but did not eliminate it; the benefit was concentrated in genuinely incomplete inputs rather than complete diagnostic cases; the effect was significantly heterogeneous across models; and for at least one model the accuracy cost would be unacceptable. Grok 4.3 had low baseline unsafe overconfidence, leaving little room for improvement, while Gemini 3.5 Flash combined a small safety gain with a large accuracy loss. Reporting the effect as a directional, relative, per-model result under an explicitly acknowledged judge-dependence caveat is more defensible than reporting a single calibrated absolute rate.

## Limitations

First, the primary comparison used a paired common panel of 300 item-perturbations per model rather than a complete all-item panel, and the control arms used a 120-item subset; this makes the results appropriate as a computational stress-test paper, not a definitive leaderboard. Second, the primary label was produced by an LLM judge whose absolute calibration is imperfect: an independent judge and blinded clinicians agreed on the direction of the wrapper effect but showed the primary judge over-labels unsafe overconfidence (clinician positive predictive value about 15%), so absolute rates and effect magnitudes should be read as judge-relative, and the primary judge is best characterized as a high-sensitivity, low-specificity screen. Third, the clinician review had important limitations of its own: one of three reviewers returned an unreliable, apparently auto-templated adjudication sheet that had to be excluded, leaving two reliable raters; the two judge-anchored sets had a clinician-unsafe base rate too low to estimate judge sensitivity, and although a dedicated positive-enriched set raised that base rate enough to do so, yielding a high sensitivity (1.00 against the human majority, robust to excluding the one divergent rater), that estimate rests on only 6 to 8 clinician-unsafe cases and its confidence interval is correspondingly wide; and the review involved a small number of physicians on a modest sample, so it validates the calibration direction rather than providing definitive ground truth. Fourth, a non-trivial fraction of generated responses were truncated or malformed (about one in five in the adjudication sample and a smaller fraction panel-wide), which the clinician `cannot judge` flag mitigated for the human analysis but which remains a response-generation quality issue affecting the automated labels. Fifth, the safety gain is accompanied by a diagnostic-accuracy cost that is large for some models, but this cost was measured with a single correctness judge on MedRBench answerable cases and was not itself validated against clinician review; given this paper's own finding that a single LLM judge can be miscalibrated, the per-model accuracy magnitudes (including the Gemini 3.5 Flash collapse) should be treated as provisional until confirmed by a blinded human spot-check, and the safety-helpfulness trade-off should be evaluated jointly and per model before any use. Sixth, perturbations are synthetic and may not capture all forms of real clinical ambiguity; the case-grounded conflicting-evidence arm applies to MedRBench only, is exploratory, and its one apparent model-level backfire did not survive the independent judge. Seventh, the format-scaffold control conflates the scaffold with a forced-commitment instruction and is therefore an adversarial bound rather than a pure placebo; the neutral scaffold provides the clean structural control. Eighth, model outputs may change over time with provider updates, and conclusions apply only to the recorded model names and run period.

## Data and Code Availability

The repository contains code, prompts, public-data manifests, primary and cross-judge rubric scores, analysis tables, figures, and manuscript files. The full raw model-output records, which embed source-case text from the underlying licensed datasets, are not redistributed in the repository and are available from the corresponding author on reasonable request. The secondary and robustness analyses are reproduced by dedicated scripts and reports: cross-judge robustness (`analysis/cross_judge_robustness.py`, `crossjudge_agreement_report.json`), helpfulness and accuracy (`analysis/accuracy_tradeoff.py`, `accuracy_tradeoff_report.json`), prompt-paraphrase robustness (`analysis/paraphrase_analysis.py`, `paraphrase_robustness_report.json`), decode stability (`analysis/stability_replicates.py`, `stability_report.json`), multiplicity, heterogeneity and power (`analysis/rigor_addons.py`, `rigor_addons_report.json`), the qualitative mechanism analysis (`analysis/qualitative_error_analysis.py`), the clinician review (`analysis/judge_validation.py` and `analysis/analyze_adjudication_final.py`, `adjudication_report_final.json`), and the positive-enriched judge-sensitivity sub-study (`analysis/build_sensitivity_positive_set.py`, `analysis/analyze_sensitivity.py`, `sensitivity_report.json`). Real-POCQi, HealthBench, and MedRBench are public-source datasets. De-identified clinician rating sheets and the hidden answer keys are included; one excluded adjudication sheet is retained with documentation of the exclusion. On publication the complete repository will be deposited in a public archive with a persistent DOI (Zenodo) and released under an open licence, with the corresponding GitHub repository linked from the archived record; in the interim the repository is available from the corresponding author on reasonable request. Provider API keys and any credential files are excluded from all released artifacts.

## Ethics and data governance

This was a retrospective public-data computational benchmark study. It did not involve recruitment of human participants, prospective patient care, clinical intervention, or collection of new patient data. The executable study used only public datasets — Real-POCQi, HealthBench, and MedRBench — each accessed under its terms of use.

The study evaluates model behavior, prompt sensitivity, and judge calibration. It does not establish clinical deployment readiness, autonomous-care safety, or patient-outcome benefit. A prospective clinician-in-the-loop evaluation would be required before any clinical deployment claim.

## Funding and Conflicts

This project was funded by Kinvectum AB. Kinvectum AB is a healthcare consultancy and technology company with no affiliation to any of the large language model providers evaluated in this study or otherwise. Koyar Afrasyab, M.D. is the founder of Kinvectum AB. The funder had no role in study design, analysis, or the decision to publish. This relationship is reported as a potential competing interest.

## Author Contributions

Koyar Afrasyab conceived the study, specified the research question and datasets, supervised the computational study design, interpreted the results, and is responsible for the final manuscript.

## Acknowledgments

The study builds on public benchmark resources and methodological ideas from Real-POCQi, HealthBench, MedRBench, and health-AI robustness-readiness work. The findings and interpretation are solely those of the author.

## Tables

- **Table 1.** Common-panel composition: paired cells by model, dataset, and perturbation type (`panel_table1_dataset_composition.csv`).
- **Table 2.** Model run metadata: provider, configuration, reasoning effort, and output counts, including the rubric judge (`panel_table2_model_run_metadata.csv`).
- **Table 3.** Primary outcome: unsafe-overconfidence rates by prompt condition with the paired risk difference, bootstrap 95% CI, and McNemar test (`panel_table3_primary_outcome.csv`).
- **Table 4.** Per-model outcomes: standard and wrapper rates, paired risk difference with 95% CI, and McNemar test (`panel_table4_per_model_outcome.csv`).
- **Table 5.** Per-perturbation outcomes (`panel_table5_per_perturbation_outcome.csv`).
- **Table 6.** Per-dataset outcomes (`panel_table6_per_dataset_outcome.csv`).
- **Table 7.** Secondary outcomes by prompt condition (correct abstention, information-seeking, identification of removed evidence, potentially harmful treatment, guideline-concordant next step, answer length) (`panel_table7_secondary_outcomes.csv`).
- **Table 8.** Mechanism decomposition: control-arm contrasts with item-clustered bootstrap 95% CIs (`panel_table8_mechanism_decomposition.csv`).

## Supplementary Material

The following supplementary files are maintained as separate documents in the `manuscript/` directory of the repository and are reproduced in full at the end of the manuscript PDF for convenience.

- **Supplementary File S1 — Reporting checklist.** Item-by-item checklist mapped to DECIDE-AI, CONSORT-AI/SPIRIT-AI, and STARD-AI principles (`manuscript/reporting_checklist.md`).
- **Supplementary File S2 — Supplementary methods.** Perturbation manifest, sampling frame, prompt hashing, raw-output logging, clinician-review export, and power simulation (`manuscript/supplementary_methods.md`).
- **Supplementary File S3 — Study protocol.** Pre-specified objective, design, outcomes, and statistical plan (`manuscript/protocol.md`).

The machine-readable data underlying the manuscript are collected under a single labeled entry point.

- **Supplementary Data S4 — Data bundle** (`supplementary_data/`, indexed by `supplementary_data/README.md`). A labeled folder that links to the analysis tables and report JSONs (`outputs/tables/`), all figures in SVG/PNG/PDF (`outputs/figures/`), the primary, cross-, correctness-, and decode-replicate rubric scores (`outputs/scores/`), and the de-identified three-clinician rating sheets with hidden answer keys (`outputs/doctor_review/`). The raw model-output records (`outputs/predictions/`) are not redistributed because they embed source-case text from the underlying licensed datasets and are available from the corresponding author on reasonable request, as described in Data and Code Availability.

## References

1. Hager P, Jungmann F, Holland R, Bhagat K, Hubrecht I, Knauer M, Vielhauer J, Makowski M, Braren R, Kaissis G, Rueckert D. Evaluation and mitigation of the limitations of large language models in clinical decision-making. *Nature Medicine*. 2024. doi:10.1038/s41591-024-03097-1.

2. Gu Y, et al. Health AI readiness evaluation and robustness stress-testing resource. *Nature Medicine*. 2026. doi:10.1038/s41591-026-04501-8. Code and data release: https://github.com/aiden-ygu/health-ai-readiness-eval/tree/v1.0.0. Zenodo: https://doi.org/10.5281/zenodo.20047288.

3. Feng JJ, et al. Expert Evaluation of Clinical AI Tools on Real Point-of-Care Clinical Queries. arXiv:2606.28960. 2026. https://arxiv.org/abs/2606.28960.

4. Feng Lab. Real-POCQi dataset. Hugging Face. 2026. https://huggingface.co/datasets/jjfenglab/Real-POCQi.

5. OpenAI. HealthBench. 2025. https://openai.com/index/healthbench/.

6. Qiu P, Wu C, Liu S, Zhao W, Chen Z, Gu H, Peng C, Zhang Y, Wang Y, Xie W. Quantifying the Reasoning Abilities of LLMs on Real-world Clinical Cases. arXiv:2503.04691. 2025. https://arxiv.org/abs/2503.04691.

7. MAGIC-AI4Med. MedRBench. GitHub. 2025. https://github.com/MAGIC-AI4Med/MedRBench.

8. Hager P, et al. MIMIC-IV-Ext Clinical Decision Making Dataset. PhysioNet. Version 1.1. https://physionet.org/content/mimic-iv-ext-cdm/1.1/.

9. Vasey B, Nagendran M, Campbell B, Clifton DA, Collins GS, Denaxas S, et al. DECIDE-AI: Developmental and Exploratory Clinical Investigations of Decision support systems driven by Artificial Intelligence. *Nature Medicine*. 2022. doi:10.1038/s41591-022-01772-9.

10. Liu X, Cruz Rivera S, Moher D, Calvert MJ, Denniston AK, SPIRIT-AI and CONSORT-AI Working Group. Reporting guidelines for clinical trials evaluating artificial intelligence interventions: the CONSORT-AI and SPIRIT-AI extensions. *Nature Medicine*. 2020. doi:10.1038/s41591-020-1037-7.

11. STARD-AI Steering Group. STARD-AI reporting guideline for diagnostic accuracy studies involving artificial intelligence. *Nature Medicine*. 2025. doi:10.1038/s41591-025-03953-8.
