# Supplementary Methods

## Perturbation manifest

Each perturbation row stores item identifier, dataset, source text hash, perturbation type, removed fields, synthetic added text, expected missing evidence, ground-truth label where available, creation timestamp, and script version.

## Sampling frame

The executable study uses all Real-POCQi questions, 200-300 HealthBench uncertainty/context examples, and 200 public diagnostic cases from MedRBench. MIMIC-derived datasets are excluded from external API calls.

## Prompt hashing

Prompts are stored as plain-text files and hashed with SHA-256. Each inference record stores the prompt condition and prompt hash.

## Raw output logging

Inference outputs are written as JSONL with run ID, item ID, perturbation ID, provider, model name, version where available, prompt condition, prompt hash, temperature, timestamp, response text, token usage, latency, error status, and raw response payload.

## Historical reviewer packet export — not human validation

The legacy annotation export blinded prompt condition labels as A/B and included columns for hypothetical clinician review. The public CSV worksheets are **unfilled templates**. Completed workbooks from earlier AI-assisted sessions must not be called physician ratings. All old apparent clinician majority, sensitivity, specificity, and adjudication statistics are excluded from the revised scientific analyses. See `REPLICATION_AUDIT.md` for the provenance audit. Genuine physician validation would require a new authenticated and blinded external reviewer protocol.

## Power simulation

`analysis/power_simulation.py` simulates paired binary outcomes across item, model, and perturbation combinations. The default assumes 200 items, 4 models, 4 perturbations, 40% baseline unsafe overconfidence, and a 10 percentage point absolute reduction.
