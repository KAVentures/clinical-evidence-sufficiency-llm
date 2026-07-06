# Round 4: Reproducibility Review

## Major concerns

- Prompts, hashes, API versions, run dates, and raw outputs must be preserved.
- MIMIC-derived data must be excluded from external API execution and public archives.

## Minor concerns

- Provider APIs can change; exact package versions and run timestamps should be stored with analysis outputs.

## Required fixes applied

- Inference JSONL schema stores prompt hash, provider, model, model version where available, timestamp, token usage, latency, errors, and raw response.
- `.gitignore` excludes raw data and generated prediction logs.
- Final analyzed outputs, judge labels, tables, figures, and subset manifests are stored under `outputs/`, `data/processed/`, and `data/annotations/`.
