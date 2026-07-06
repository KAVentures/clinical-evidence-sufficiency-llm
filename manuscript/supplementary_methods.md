# Supplementary Methods

## Perturbation manifest

Each perturbation row stores item identifier, dataset, source text hash, perturbation type, removed fields, synthetic added text, expected missing evidence, ground-truth label where available, creation timestamp, and script version.

## Sampling frame

The executable study uses all Real-POCQi questions, 200-300 HealthBench uncertainty/context examples, and 200 public diagnostic cases from MedRBench. MIMIC-derived datasets are excluded from external API calls.

## Prompt hashing

Prompts are stored as plain-text files and hashed with SHA-256. Each inference record stores the prompt condition and prompt hash.

## Raw output logging

Inference outputs are written as JSONL with run ID, item ID, perturbation ID, provider, model name, version where available, prompt condition, prompt hash, temperature, timestamp, response text, token usage, latency, error status, and raw response payload.

## Clinician review export

The annotation export blinds prompt condition labels as A/B and includes clinician-label columns for primary and secondary rubric fields.

## Power simulation

`analysis/power_simulation.py` simulates paired binary outcomes across item, model, and perturbation combinations. The default assumes 200 items, 4 models, 4 perturbations, 40% baseline unsafe overconfidence, and a 10 percentage point absolute reduction.
