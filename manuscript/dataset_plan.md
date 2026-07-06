# Public API-Compatible Dataset Plan

MIMIC-CDM is excluded from the executable API-based study because MIMIC-derived data are currently prohibited from being used with external APIs.

| Dataset | Items | Role |
| --- | ---: | --- |
| Real-POCQi | all 620 | Primary real physician point-of-care question set |
| HealthBench | 200-300 selected uncertainty/context cases | Safety and context-seeking set |
| MedRBench | 200 diagnostic cases | Public diagnostic-case set |

## Dataset notes

Real-POCQi contributes real physician point-of-care questions and is the main source for treatment, drug-safety, diagnosis, and management queries.

HealthBench contributes health-conversation cases enriched for uncertainty, context seeking, emergency referral, and safety-sensitive tasks.

MedRBench replaces MIMIC-CDM for diagnostic case evaluation in external API runs. Use `MAGIC-AI4Med/MedRBench`, specifically `data/MedRBench/diagnosis_957_cases_with_rare_disease_491.json`, and sample 200 diagnostic cases.

## MIMIC exclusion

MIMIC-CDM remains relevant background literature, but it is not part of the API-run sampling frame. Any future MIMIC analysis must use a compliant local or approved environment that does not transmit MIMIC-derived content to prohibited external APIs.
