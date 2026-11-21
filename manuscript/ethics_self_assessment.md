# Research ethics and data-governance self-assessment

**Study:** *Judge-dependent safety gains and model-specific helpfulness costs of evidence-sufficiency prompting in clinical LLMs*

**Author:** Koyar Afrasyab, MD; Kinvectum AB, Sweden  
**Basis:** Retrospective computational benchmark using public-source clinical vignettes/questions, model-generated responses, and blinded clinician assessments of AI-generated text.

## Scope of the assessment

The study did not recruit patients, intervene in care, access medical records for identified individuals, or report patient outcomes. It used the publicly released portions of Real-POCQi, HealthBench, and MedRBench; MIMIC-derived cases were excluded from external-API execution. Source-data permissions and deidentification depend on the source datasets' release terms, not on this self-assessment.

Three clinicians assessed blinded AI-generated answers as expert raters. This **was human expert participation in an evaluation workflow**: it should not be described as a study with no human contributors. The repository contains deidentified reviewer rating worksheets; no patient-level identifying data were intended to be collected. Reviewer correspondence and administrative attribution, if retained outside the analytical datasets, must be handled under applicable privacy and institutional policies. A reviewer reference set is not automatically clinical ground truth.

## Swedish ethical-review considerations

The author previously assessed the work as not requiring mandatory Swedish ethical review under the Ethical Review Act (2003:460) because the computational study does not involve patient interventions, biological material, or planned processing of sensitive personal health data about identifiable research subjects. **This is an investigator self-assessment, not a formal decision by Etikprövningsmyndigheten, a legal opinion, or an institutional ethics approval.** The expert-rating component and any retained reviewer information should be disclosed accurately when a receiving journal or institution evaluates its requirements.

Before a claim of formal exemption is made, confirm the applicable rules and affiliation-specific procedures with the responsible institution or competent authority, if required. The manuscript should say that no formal ethics approval was obtained, rather than imply one was issued.

## Consent, confidentiality, and audit

The clinical review comprises professional judgments on text outputs, not individual patient-care data. Preserve reviewer privacy in publicly released sheets and exclude names, emails, credentials, and keys from analytical archives. Retain the unedited scoring sheets, adjudication decisions, exclusion rationale, code, and timestamps necessary for an independent audit, subject to legal and participant-data restrictions. Do not equate absence of identifiable patient data with absence of all potential governance obligations.

## Research boundary

This benchmark does not demonstrate clinical safety, treatment effectiveness, diagnostic-device accuracy, or deployment readiness. Its primary endpoint is assigned by an LLM judge, with a limited clinician anchoring sample and an independent cross-judge analysis.

**Status:** investigator self-assessment; no formal authority determination documented in this repository.
