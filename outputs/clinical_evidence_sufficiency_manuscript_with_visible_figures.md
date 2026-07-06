# Evidence sufficiency and unsafe overconfidence in clinical LLM decision support: a public-data computational stress test

**Author:** Koyar Afrasyab, M.D.  
**Affiliation:** Independent researcher  
**ORCID:** https://orcid.org/0009-0009-3530-4606  
**Correspondence:** Koyar Afrasyab, M.D.; koyar@kinvectum.com

## Abstract

**Background:** Clinical language models can provide fluent recommendations even when patient-specific evidence is incomplete, ambiguous, or contradictory. Prior clinical decision-making and robustness studies show that benchmark performance can fall under realistic information-gathering, context-removal, and perturbation conditions [1,2]. We evaluated whether a structured evidence-sufficiency prompt reduces unsafe overconfident clinical answers.

**Methods:** We conducted a retrospective public-data benchmark using Real-POCQi, HealthBench, and MedRBench [3-7]. MIMIC-CDM was excluded because MIMIC-derived data are currently prohibited from external API use [8]. Four high-reasoning model configurations were evaluated: GPT-5.5, Claude Opus 4.8, Gemini 3.5 Flash, and Grok 4.3. The primary analysis used a fully paired common panel of 300 item-perturbations per model (1,200 paired model-item cells, 2,400 model outputs), each answered with a standard clinical prompt and an evidence-sufficiency wrapper. To test whether the effect was an artifact of the judge rewarding the wrapper's structural tokens (circularity), we added two matched control arms on a 120-item x 4-model subset (480 cells each): a neutral scaffold using the wrapper's four section labels with neither an abstain nor a commit instruction, and a format scaffold using the same labels with a forced-commitment instruction. The primary rubric judge was GPT-5.4-nano. We pre-specified multiple robustness and validity analyses: an independent different-family judge (Claude Sonnet 5) re-scored the full paired panel; a correctness judge scored diagnostic accuracy and abstention on answerable complete-information cases to quantify the helpfulness cost; two independent paraphrases of the wrapper and stochastic decode replicates tested prompt-wording and sampling robustness; and a blinded three-clinician review tested whether the judge's threshold matches clinical judgment. The primary estimand was the paired absolute risk difference in unsafe overconfidence, standard prompt minus wrapper, with 10,000 bootstrap resamples and McNemar testing; subgroup analyses used Holm and Benjamini-Hochberg correction and a GEE model x wrapper interaction test of effect heterogeneity.

**Results:** On the common panel, unsafe overconfidence occurred in 592/1,200 standard-prompt responses (49.3%) and 296/1,200 evidence-sufficiency responses (24.7%). The wrapper reduced unsafe overconfidence by 24.7 percentage points (95% bootstrap CI, 21.8 to 27.7; McNemar 348 vs 52 discordant pairs, p=3.1e-49; adjusted GEE odds ratio 0.22, p=3.0e-39). Model-specific reductions were 43.0 points for Claude Opus 4.8, 30.0 for Gemini 3.5 Flash, 19.3 for GPT-5.5, and 6.3 for Grok 4.3 (all Holm-adjusted p<0.05), and the effect differed significantly across models (interaction joint Wald p=2.7e-11). In the control arms, identical scaffold tokens produced very different unsafe rates (neutral 37.7%, wrapper 22.9%, format 79.8% on the matched subset), so the judge scored behavior rather than tokens; the effect decomposed additively into a scaffold-structure component (+9.0 points) and a larger abstention-content component (+14.8 points). The direction of benefit was robust to the independent Sonnet judge and to two wrapper paraphrases (+28 to +29 points) and stochastic decoding, but the absolute magnitude was judge-dependent: the two judges disagreed on 1,147/3,534 cells almost entirely in one direction (GPT-5.4-nano labeled unsafe where Sonnet labeled safe, 1,147 vs 15 the reverse), and the paired reduction fell to +13.1 points under Sonnet. The wrapper's safety gain carried a helpfulness cost that was itself strongly model-dependent: on 330 answerable complete-information cases, correct diagnosis fell from 80.3% to 50.3% and abstention rose from 12.7% to 46.4%, but this ranged from near-free for GPT-5.5 (correct -2 points) to catastrophic for Gemini 3.5 Flash (correct -58 points; abstention 18% to 82%). In the blinded clinician review, the primary judge's sensitivity to human-labeled unsafe responses was high but its specificity was low (positive predictive value 15%; human-majority unsafe rate 6.7% vs judge 44%), and on judge-discordant cases the two reliable clinicians sided with the conservative Sonnet judge, confirming that the primary judge over-labels in absolute terms.

**Conclusions:** In this public-data computational benchmark, evidence-sufficiency prompting reliably and reproducibly reduced unsafe overconfident clinical responses, and a matched control-arm decomposition showed the direction of the effect reflects genuine behavior change rather than judge circularity. However, the absolute magnitude of the effect and of the unsafe-overconfidence rate itself are properties of the judge's calibration: an independent judge and blinded clinicians agreed on the direction but placed the threshold much higher, and the safety gain came at a diagnostic-accuracy cost that was acceptable for some models and unacceptable for others. These findings support evaluating clinical AI for abstention, information-seeking, and the safety-helpfulness trade-off jointly, reporting effects as directional and relative rather than as calibrated absolute rates, and do not establish clinical deployment readiness or patient-outcome benefit.

## Introduction

High benchmark performance does not establish safe clinical decision-support behavior. A model can answer complete-information questions correctly while failing in realistic settings where key data are missing, contradictory, or should trigger further information gathering. Hager et al. showed that large language models degrade in a realistic clinical decision-making framework requiring information gathering, lab interpretation, imaging interpretation, diagnosis, and treatment planning rather than static answer selection [1]. Gu et al. argued that medical AI benchmark readiness requires robustness stress tests because strong headline scores can mask brittleness under input removal, distractor, format, and visual perturbations [2]. Feng et al. introduced Real-POCQi to move evaluation toward real physician point-of-care questions and expert physician judgments [3]. HealthBench and MedRBench add complementary health-conversation safety and clinical-reasoning benchmark settings [5-7].

This study focuses on unsafe overconfidence: giving a definitive diagnosis, treatment recommendation, or management plan despite insufficient or contradictory information, without clearly stating uncertainty and without requesting missing clinically necessary information. The hypothesis was that a simple evidence-sufficiency wrapper would reduce unsafe overconfident responses compared with a standard clinical-answer prompt.

## Methods

### Study Design

This was a retrospective public-data benchmark and perturbation study with paired repeated measures, designed in the spirit of reporting and evaluation guidance for AI decision-support systems [9-11]. The unit of analysis was model x item x perturbation x prompt condition. The primary comparison was standard prompt versus evidence-sufficiency wrapper for the same model and item-perturbation.

The original protocol considered MIMIC-CDM because it is directly aligned with clinical decision-making under incomplete information [1,8]. MIMIC-derived data were excluded from external API calls after confirming the data-use restriction. MedRBench diagnostic cases replaced the MIMIC arm for the executable public-data study [6,7].

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

Because the sole automated label is an LLM judge, we validated it two ways. First, an independent judge from a different model family, Claude Sonnet 5, re-scored the entire paired common panel and the contradiction arm using the identical rubric, and we compared the two judges by Cohen's kappa, raw agreement, the direction of discordance, and the recomputed paired risk difference under each judge. Second, three physicians independently rated blinded responses (case text and model response only, blinded to model, prompt condition, and both judge labels). Two review sets were prepared: a 120-item judge-validation set balanced across models and conditions, and a 90-item adjudication set that oversampled judge-discordant cells (primary judge unsafe, cross judge safe) so clinicians could arbitrate which threshold is clinically correct. Both sets included a `cannot judge / needs more context` option to flag truncated or malformed responses. Analyses reported inter-rater agreement, judge sensitivity and specificity against the human majority, and, on discordant cells, whether clinicians sided with the over-labeling or the conservative judge.

### Helpfulness and Accuracy

Because the unsafe-overconfidence endpoint treats an abstention on an answerable question as safe, it cannot by itself detect over-abstention. We therefore scored diagnostic accuracy on the answerable subset: MedRBench complete-information cases with a gold diagnosis in the ground-truth label. A correctness judge (GPT-5.4-mini) labeled each response for whether it gave a definitive diagnosis, whether that diagnosis was correct, and whether it abstained or deferred, and we computed the paired wrapper-minus-standard change in correct diagnosis and abstention overall and per model.

### Robustness Analyses

To test whether the effect depended on the exact wrapper wording, two independent paraphrases of the wrapper (one prose without section labels, one with different section labels) were generated and judged on an 80-item subset for Claude Opus 4.8 and GPT-5.5. To test stability against stochastic decoding, five independent replicates at temperature 0.8 were generated for a 40-item subset under both conditions for two lower-cost models and judged with the primary judge, quantifying per-cell label unanimity and the spread of the paired risk difference across replicates.

### Statistical Analysis

The primary estimand was the paired absolute risk difference in unsafe overconfidence, standard prompt minus evidence-sufficiency wrapper, on the common panel. Confidence intervals used 10,000 bootstrap resamples of paired differences. McNemar tests compared discordant paired binary outcomes. A GEE logistic model clustered by item ID was fit as a sensitivity analysis, adjusting for model, dataset, and perturbation type. Control-arm contrasts (standard-to-neutral, neutral-to-wrapper, and neutral-to-format) were computed on the matched subset of cells present in all arms, with confidence intervals from an item-clustered bootstrap, and an additive check compared the sum of the scaffold-structure and abstention-content contrasts against the full wrapper effect. We pre-specified the common-panel wrapper effect as the confirmatory primary analysis and labeled control-arm, cross-judge, helpfulness, contradiction, and robustness analyses as secondary or exploratory. All subgroup McNemar tests (per model, dataset, and perturbation type) were corrected for multiplicity with the Holm and Benjamini-Hochberg procedures. Effect heterogeneity across models was tested formally with a GEE logistic model including a model x wrapper interaction and a joint Wald test that all interaction terms were zero. Achieved power and the minimum detectable effect were computed for the paired primary endpoint.

## Results

### Dataset and Run Composition

The primary analysis included 2,400 scored outputs from 1,200 paired model-item perturbations: 300 paired perturbations for each of GPT-5.5, Claude Opus 4.8, Gemini 3.5 Flash, and Grok 4.3. The panel included Real-POCQi, HealthBench, and MedRBench cases, with decontextualized, context-uncertainty, full-information, missing-ancillary-test, and reworded perturbations. The two control arms added 960 further outputs (480 neutral-scaffold and 480 format-scaffold cells). The judged data contained no failed judge calls.

![Figure 1. Study design](figures/panel_figure1_study_design.png)

*Figure 1. Study design. Public clinical benchmark sources were transformed into evidence-sufficiency stress variants, answered by four high-reasoning model configurations under standard and evidence-sufficiency prompts, and scored with a separate rubric judge before clinician adjudication.*


### Primary Outcome

Unsafe overconfidence occurred in 592/1,200 standard-prompt responses (49.3%) and 296/1,200 evidence-sufficiency responses (24.7%). The absolute paired reduction was 24.7 percentage points (95% bootstrap CI, 21.8 to 27.7). McNemar testing showed 348 pairs where the standard prompt was unsafe and the wrapper was safe, versus 52 pairs where the standard prompt was safe and the wrapper was unsafe (p=3.1e-49).

In the adjusted GEE sensitivity model, the evidence-sufficiency wrapper was associated with lower odds of unsafe overconfidence (odds ratio 0.22; p=3.0e-39), adjusted for model, dataset, and perturbation type.

![Figure 2. Unsafe overconfidence by prompt](figures/panel_figure2_unsafe_overconfidence_by_prompt.png)

*Figure 2. Unsafe overconfidence by prompt. Primary common-panel automated-rubric outcome. Unsafe overconfidence labels were lower under the evidence-sufficiency wrapper than under the standard clinical-answer prompt.*


### Per-Model Results

The wrapper reduced unsafe overconfidence in all four model subsets, but the magnitude varied:

- Claude Opus 4.8: 58.3% standard vs 15.3% wrapper; risk difference 43.0 percentage points (95% CI, 37.0 to 49.0).
- Gemini 3.5 Flash: 80.7% vs 50.7%; risk difference 30.0 points (95% CI, 24.0 to 36.3).
- GPT-5.5: 46.3% vs 27.0%; risk difference 19.3 points (95% CI, 13.3 to 25.3).
- Grok 4.3: 12.0% vs 5.7%; risk difference 6.3 points (95% CI, 2.0 to 10.7).

Grok 4.3 had the lowest baseline unsafe overconfidence rate on the common panel. Its reduction was small but, unlike in the earlier underpowered subset, statistically significant by McNemar testing (p=0.007), indicating the low baseline is a genuine property of the model rather than an artifact of item sampling.

![Figure 3. Prompt effect by model](figures/panel_figure3_risk_difference_by_model.png)

*Figure 3. Prompt effect by model. Absolute risk difference by model in the common-panel analysis. Positive values favor the evidence-sufficiency wrapper. Model comparisons should not be interpreted as a leaderboard.*


### Dataset and Perturbation Results

Risk differences were observed across all datasets:

- Real-POCQi: 47.0% standard vs 15.7% wrapper; risk difference 31.3 points.
- HealthBench: 34.7% vs 14.8%; risk difference 19.9 points.
- MedRBench: 61.2% vs 44.3%; risk difference 16.9 points.

By perturbation type, the largest reduction occurred in decontextualized Real-POCQi-style inputs (31.6 points). Reductions were also observed for missing ancillary tests (27.8 points), context-uncertainty cases (19.1 points), and reworded variants (19.0 points). The effect was smallest and not statistically distinguishable from zero for full-information diagnostic cases (64.0% vs 57.3%; risk difference 6.7 points, 95% CI -1.8 to 15.2), consistent with the wrapper acting mainly when information is genuinely incomplete rather than on complete cases.

![Figure 4. Prompt effect by perturbation type](figures/panel_figure4_risk_difference_by_perturbation.png)

*Figure 4. Prompt effect by perturbation type. Absolute risk difference by perturbation type. The largest reductions were seen when information was genuinely incomplete, consistent with the wrapper's intended effect of discouraging definitive answers when case-specific evidence is missing.*


![Figure 5. Prompt effect by dataset](figures/panel_figure5_risk_difference_by_dataset.png)

*Figure 5. Prompt effect by dataset. Absolute risk difference by dataset in the common-panel analysis. Positive values indicate lower automated unsafe-overconfidence labels under the evidence-sufficiency wrapper than under the standard prompt.*


### Mechanism and Circularity Control

A peer-review concern was that the wrapper mechanically emits the section tokens (`EVIDENCE PRESENT`, `EVIDENCE MISSING`, `SUFFICIENCY JUDGMENT`) the judge might reward, so the effect could be circular. The control arms refute this. On the matched control subset (480 cells per arm), the identical scaffold tokens produced markedly different unsafe-overconfidence rates depending on the accompanying instruction: 37.7% under the neutral scaffold, 22.9% under the wrapper, and 79.8% under the format scaffold (same tokens plus a forced-commitment instruction), versus 46.7% under the standard prompt. Because the same tokens map to unsafe rates spanning 23% to 80%, the judge is scoring response behavior, not the presence of scaffold tokens.

The wrapper effect decomposed additively into two components, each with a bootstrap confidence interval excluding zero. The scaffold structure alone (standard to neutral) reduced unsafe overconfidence by 9.0 points (95% CI, 6.2 to 11.8), and the abstention instruction added on top of that structure (neutral to wrapper) reduced it by a further 14.8 points (95% CI, 11.6 to 17.8). Their sum (23.8 points) matched the full wrapper effect measured on the same subset (23.8 points), so roughly 38% of the benefit is attributable to structured reasoning and roughly 62% to the abstention content. As an adversarial bound, attaching a forced-commitment instruction to the same scaffold (neutral to format) increased unsafe overconfidence by 42.1 points (95% CI, 38.5 to 45.7), confirming that the same structure can be steered toward worse behavior. The effect is therefore best reported as a decomposition, not as a monolithic claim that reasoning helps.

![Figure 6. Control-arm unsafe-overconfidence rates](figures/panel_figure6_control_arm_rates.png)

*Figure 6. Control-arm unsafe-overconfidence rates. Matched control arms using the same scaffold labels showed that unsafe-overconfidence rates depended on instruction content, not merely the presence of wrapper section tokens.*


![Figure 7. Mechanism decomposition](figures/panel_figure7_mechanism_decomposition.png)

*Figure 7. Mechanism decomposition. The full wrapper effect decomposed into a scaffold-structure component and a larger abstention-content component, while a forced-commitment scaffold worsened unsafe overconfidence.*


### Secondary Outcomes

The wrapper improved several safety-relevant behaviors on the common panel. Correct abstention increased from 32.8% under the standard prompt to 66.9% under the wrapper. Asking for missing information increased from 10.6% to 63.8%. Identification of removed evidence increased from 8.3% to 19.4%. Potentially harmful treatment recommendations decreased from 13.2% to 3.3%. Median answer length increased from 143 to 186 words; the length increase alone did not explain the safety gain, as the format scaffold produced the shortest of the scaffolded outputs yet the highest unsafe rate.

### Statistical Robustness, Multiplicity, and Heterogeneity

The paired primary endpoint (n=1,200 pairs) was well powered: achieved power for the observed effect was essentially 1.0, and the minimum detectable effect at 80% power was 4.2 percentage points. After Holm and Benjamini-Hochberg correction across the thirteen subgroup tests (overall, four models, three datasets, five perturbation types), every subgroup remained significant except full-information diagnostic cases (Holm-adjusted p=0.16). The wrapper effect was significantly heterogeneous across models: a GEE model with a model x wrapper interaction rejected the hypothesis of a constant effect (joint Wald statistic 52.2, p=2.7e-11). Effects should therefore be reported per model rather than pooled into a single headline number.

### Cross-Judge Robustness

An independent judge from a different model family (Claude Sonnet 5) re-scored all 3,534 double-judged cells. The two judges agreed on the direction of the wrapper effect but not on its magnitude or on absolute unsafe rates. Raw agreement was 67% (Cohen's kappa 0.19), and disagreement was almost entirely one-directional: 1,147 cells were labeled unsafe by the primary judge but safe by Sonnet, versus only 15 in the reverse direction. Under the primary judge the paired common-panel reduction was 24.7 points; under Sonnet it was 13.1 points on the same 1,200 pairs. The gap was largest for Gemini 3.5 Flash (nano-minus-Sonnet unsafe-rate difference 0.54) and negligible for Grok 4.3 (0.047), indicating that the primary judge's absolute over-labeling is itself model-dependent. The wrapper's benefit was directionally robust under both judges, but its absolute size is a function of judge calibration.

![Figure 8. Cross-judge calibration](figures/panel_figure8_cross_judge_overlabeling.png)

*Figure 8. Cross-judge calibration. Independent judging showed the same directional wrapper benefit but lower absolute unsafe-overconfidence rates, consistent with primary-judge over-labeling.*


### Helpfulness and Accuracy Trade-off

Because the safety endpoint scores an abstention on an answerable question as safe, we measured diagnostic accuracy directly on 330 paired answerable complete-information cases. Correct diagnosis fell from 80.3% under the standard prompt to 50.3% under the wrapper (-30.0 points), while abstention rose from 12.7% to 46.4% (+33.6 points). This cost was strongly model-dependent and inversely tracked each model's safety gain: for GPT-5.5 the accuracy cost was near zero (correct diagnosis -2 points), for Claude Opus 4.8 it was modest (-7 points), for Grok 4.3 it was -10 points, and for Gemini 3.5 Flash it was catastrophic (correct diagnosis 75% to 17%, -58 points; abstention 18% to 82%). The wrapper therefore does not uniformly improve behavior; its net value is the safety gain minus the helpfulness loss, which was favorable for GPT-5.5 and Claude Opus 4.8 but unfavorable for Gemini 3.5 Flash, whose safety gain was small and whose accuracy collapse was large.

![Figure 9. Safety-helpfulness trade-off](figures/panel_figure9_safety_helpfulness_tradeoff.png)

*Figure 9. Safety-helpfulness trade-off. Diagnostic accuracy and abstention changed differently by model, showing that the safety gain carried a model-specific helpfulness cost.*


### Prompt-Paraphrase and Decode Robustness

The effect was not an artifact of the exact wrapper wording. Two independent paraphrases of the wrapper produced paired reductions of 29.4 and 28.1 points on the 80-item subset (160 pairs each), statistically indistinguishable from the original wrapper's 29.4 points on the same subset (all p<1e-8), for both Claude Opus 4.8 and GPT-5.5. The effect was also stable under stochastic decoding: across five temperature-0.8 replicates, individual cell labels were unanimous 68% of the time, but the aggregate paired risk difference was consistently large and positive for both tested models (0.44 +/- 0.065 and 0.31 +/- 0.045; minimum across replicates 0.25).

### Contradiction Arm

A case-grounded conflicting-evidence arm was generated by having a model negate one explicit, structural finding within each MedRBench complete-information case (for example, a same-encounter pathology report stating the opposite result), with clinician spot-check validation (24/25 items judged valid). Because complete-information base cases existed only in MedRBench, this arm applies to that dataset only. Under the primary judge the wrapper reduced unsafe overconfidence by 19.3 points overall, but the effect was strongly model-conditional and included a failure signal: Claude Opus 4.8 improved by 50.7 points and GPT-5.5 by 30.0, Grok 4.3 was near the floor, and Gemini 3.5 Flash appeared to worsen. However, this apparent Gemini backfire sign-flipped under the Sonnet judge and the arm's effect fell to 5.5 points under Sonnet, so the contradiction arm is reported as exploratory and the Gemini result is not interpreted as a genuine backfire.

### Clinician Review

All three physicians completed both review sets, but one physician's adjudication-set submission was excluded as unreliable: it labeled 87.8% of responses unsafe versus 5.8% on that same reviewer's judge-validation set, used near-verbatim templated rationales on 79 of 90 items, and flagged no truncated responses where the other two flagged 17 and 19. On the 120-item judge-validation set the primary judge had 100% sensitivity but low specificity against the human majority (positive predictive value 15%; human-majority unsafe rate 6.7% versus judge unsafe rate 44%), consistent with a high-sensitivity, low-specificity screen. On the 90-item adjudication set, the two reliable clinicians agreed with each other on 97.1% of judgeable items, and on all 46 judgeable judge-discordant cells (primary judge unsafe, Sonnet safe) they sided with Sonnet (safe), directly confirming that the primary judge over-labels in absolute terms. Two caveats temper the human data: 21 of 90 responses (23%) were dropped as truncated or unjudgeable, disproportionately from cells both judges had called unsafe; and because the clinician-labeled unsafe base rate was very low, this set could assess judge over-labeling (specificity) but not judge sensitivity.

### Mechanism of Avoided Overconfidence

To characterize how the wrapper avoided overconfidence rather than only that it did, a sample of cells that flipped from unsafe under the standard prompt to safe under the wrapper was categorized. The dominant mechanisms were requesting specific missing data (for example particular labs, imaging, history, or examination findings) and explicitly declining a diagnosis on insufficient information, followed by flagging uncertainty while giving only safe general next steps; vague hedging without a concrete safety action was rare. The wrapper thus improves safety chiefly by eliciting information-seeking and explicit deferral, consistent with the abstention-content component identified in the control-arm decomposition.

## Discussion

In this public-data stress test, a simple evidence-sufficiency wrapper reduced unsafe overconfident clinical responses by about one quarter of all paired cases on a fully paired common panel. The effect was observed across datasets and across all four model subsets, although the baseline unsafe overconfidence rate and effect size varied substantially by model.

A matched control-arm decomposition addressed the main internal-validity threat to this design. Because the same scaffold tokens yielded unsafe rates from 23% to 80% depending on the accompanying instruction, the reduction is not an artifact of the judge rewarding the wrapper's format. The effect instead splits into a structural component and a larger abstention-content component, each significant, which clarifies why the intervention works and what a minimal version would need to retain.

These findings support the central hypothesis that clinical LLM evaluation should measure whether models know when not to answer definitively. Benchmark accuracy alone can miss clinically important behavior: whether a model requests missing information, acknowledges uncertainty, identifies removed evidence, and avoids potentially harmful treatment recommendations when the prompt is under-specified. This aligns with MIMIC-CDM's emphasis on information gathering before diagnosis or treatment, Real-POCQi's emphasis on physician-realistic information needs, and robustness-readiness work showing that apparently strong health-AI systems can fail under small but clinically meaningful perturbations [1-3].

Two robustness results strengthen the causal reading of the direction of effect. The reduction was reproduced almost identically by two independent paraphrases of the wrapper and was stable across stochastic decodes, so it is not an artifact of a particular prompt string or of sampling noise, and the qualitative analysis showed the mechanism is concrete information-seeking and explicit deferral rather than vague hedging. Together with the control-arm decomposition, this makes a reasonably strong case that the wrapper changes model behavior in the intended direction.

At the same time, three findings caution strongly against over-reading the absolute numbers. First, the magnitude of the effect and the absolute unsafe-overconfidence rate are properties of the judge's calibration, not of the models alone: an independent different-family judge agreed on the direction but nearly halved the effect, and disagreement was almost entirely the primary judge labeling as unsafe what the second judge called safe. Second, blinded clinicians confirmed this over-labeling directly, agreeing with the conservative judge on every judge-discordant case they could rate and implying a human unsafe rate several-fold below the primary judge's. Third, and most important for deployment, the safety gain is not free: it is bought with abstention, and the diagnostic-accuracy cost ranged from negligible for GPT-5.5 to a near-total collapse of correct diagnosis for Gemini 3.5 Flash. The wrapper is therefore best understood as shifting models along a safety-helpfulness trade-off whose favorability is model-specific, not as a uniform safety improvement.

The results should not be interpreted as showing that prompting solves clinical AI safety. The wrapper reduced unsafe overconfidence but did not eliminate it; the benefit was concentrated in genuinely incomplete inputs rather than complete diagnostic cases; the effect was significantly heterogeneous across models; and for at least one model the accuracy cost would be unacceptable. Grok 4.3 had low baseline unsafe overconfidence, leaving little room for improvement, while Gemini 3.5 Flash combined a small safety gain with a large accuracy loss. Reporting the effect as a directional, relative, per-model result under an explicitly acknowledged judge-calibration caveat is more defensible than reporting a single calibrated absolute rate.

## Limitations

First, the primary comparison used a paired common panel of 300 item-perturbations per model rather than a complete all-item panel, and the control arms used a 120-item subset; this makes the results appropriate as a computational stress-test paper, not a definitive leaderboard. Second, the primary label was produced by an LLM judge whose absolute calibration is imperfect: an independent judge and blinded clinicians agreed on the direction of the wrapper effect but showed the primary judge over-labels unsafe overconfidence (clinician positive predictive value about 15%), so absolute rates and effect magnitudes should be read as judge-relative, and the primary judge is best characterized as a high-sensitivity, low-specificity screen. Third, the clinician review had important limitations of its own: one of three reviewers returned an unreliable, apparently auto-templated adjudication sheet that had to be excluded, leaving two reliable raters; the clinician-labeled unsafe base rate was low enough that the review could assess judge over-labeling but not judge sensitivity; and it involved a small number of physicians on a modest sample, so it validates the calibration direction rather than providing definitive ground truth. Fourth, a non-trivial fraction of generated responses were truncated or malformed (about one in five in the adjudication sample and a smaller fraction panel-wide), which the clinician `cannot judge` flag mitigated for the human analysis but which remains a response-generation quality issue affecting the automated labels. Fifth, the safety gain is accompanied by a diagnostic-accuracy cost that is large for some models, measured here with a single correctness judge on MedRBench answerable cases; the safety-helpfulness trade-off should be evaluated jointly and per model before any use. Sixth, perturbations are synthetic and may not capture all forms of real clinical ambiguity; the case-grounded conflicting-evidence arm applies to MedRBench only, is exploratory, and its one apparent model-level backfire did not survive the independent judge. Seventh, the format-scaffold control conflates the scaffold with a forced-commitment instruction and is therefore an adversarial bound rather than a pure placebo; the neutral scaffold provides the clean structural control. Eighth, model outputs may change over time with provider updates, and conclusions apply only to the recorded model names and run period. Ninth, MIMIC-CDM was not used because MIMIC-derived data are prohibited from external API use; this improves data-use compliance but reduces comparability with EHR-derived clinical decision-making simulations.

## Data and Code Availability

The repository contains code, prompts, public-data manifests, primary and cross-judge rubric scores, analysis tables, figures, and manuscript files. The full raw model-output records, which embed source-case text from the underlying licensed datasets, are not redistributed in the repository and are available from the corresponding author on reasonable request. The secondary and robustness analyses are reproduced by dedicated scripts and reports: cross-judge robustness (`analysis/cross_judge_robustness.py`, `crossjudge_agreement_report.json`), helpfulness and accuracy (`analysis/accuracy_tradeoff.py`, `accuracy_tradeoff_report.json`), prompt-paraphrase robustness (`analysis/paraphrase_analysis.py`, `paraphrase_robustness_report.json`), decode stability (`analysis/stability_replicates.py`, `stability_report.json`), multiplicity, heterogeneity and power (`analysis/rigor_addons.py`, `rigor_addons_report.json`), the qualitative mechanism analysis (`analysis/qualitative_error_analysis.py`), and the clinician review (`analysis/judge_validation.py` and `analysis/analyze_adjudication_final.py`, `adjudication_report_final.json`). Real-POCQi, HealthBench, and MedRBench are public-source datasets. MIMIC-derived data are not included and were not sent to external APIs. De-identified clinician rating sheets and the hidden answer keys are included; one excluded adjudication sheet is retained with documentation of the exclusion.

## Ethics

This was a retrospective computational benchmark using public non-MIMIC data. It does not involve prospective patient care and does not support clinical deployment claims. A prospective clinician-in-the-loop evaluation would be required before clinical deployment claims.

## Funding and Conflicts

This work was funded by Kinvectum AB. Koyar Afrasyab is founder of Kinvectum AB. The author reports this relationship as a potential competing interest. No MIMIC-derived records were sent to external APIs.

## Author Contributions

Koyar Afrasyab conceived the study, specified the research question and datasets, supervised the computational study design, interpreted the results, and is responsible for the final manuscript.

## Acknowledgments

The study builds on public benchmark resources and methodological ideas from MIMIC-CDM, Real-POCQi, HealthBench, MedRBench, and health-AI robustness-readiness work. The findings and interpretation are solely those of the author.

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
