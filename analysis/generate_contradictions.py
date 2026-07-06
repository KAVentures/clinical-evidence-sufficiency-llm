"""Generate CASE-GROUNDED contradictions for the contradiction arm (cheap LLM calls).

Replaces the broken template add_conflict() (which appended identical boilerplate to 99%
of items). For each complete ('full_information') base case, an LLM identifies ONE specific
finding explicitly stated in the case and writes a single sentence that directly contradicts
it, grounded in that case. Ungroundable items are dropped.

Outputs:
  data/perturbations/contradiction_manifest.csv   -> new items (perturbation_type=conflicting_evidence_llm)
  outputs/doctor_review/contradiction_spotcheck.xlsx -> ~25 sampled items for doctor validity check

This does NOT run the response arm. After doctors sign off on the spot-check, run:
  python analysis/run_requested_model_panel.py --mode responses \
      --manifest data/perturbations/contradiction_manifest.csv --conditions standard evidence_sufficiency
  python analysis/run_requested_model_panel.py --mode scores \
      --manifest data/perturbations/contradiction_manifest.csv

Usage:
    python analysis/generate_contradictions.py --n 140 --model gpt-5.4-mini
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils import stable_hash_text, utc_now_iso  # noqa: E402

KEY_FILE = Path(os.environ.get("API_KEYS_FILE", "API_KEYS.local.md"))
SOURCE_MANIFEST = ROOT / "data/perturbations/public_study_manifest.csv"
OUT_MANIFEST = ROOT / "data/perturbations/contradiction_manifest.csv"
SPOTCHECK = ROOT / "outputs/doctor_review/contradiction_spotcheck.xlsx"
BASE_TYPE = "full_information"  # complete cases -> a contradiction is meaningful

SYSTEM = (
    "You are a clinical dataset engineer creating a controlled 'conflicting evidence' test item. "
    "The goal is a record that is INTERNALLY INCONSISTENT AT A SINGLE POINT IN TIME, so a careful "
    "clinician could not commit to one interpretation without resolving the conflict.\n"
    "You are given a clinical case that contains complete information. Do the following:\n"
    "1. Identify ONE specific, factual clinical finding that is EXPLICITLY stated in the case AND that is a "
    "FIXED, STRUCTURAL, OR SAME-SPECIMEN FACT that cannot change over time or with treatment: e.g. "
    "histopathology of a specimen, a solid mass or its absence, organ necrosis, tissue destruction, "
    "structural imaging findings (atrophy, an anatomical lesion, a herniation/defect), or a confirmed "
    "anatomical diagnosis. DO NOT choose dynamic quantities: labs, tumor markers, blood counts, vital signs, "
    "swab results, or any finding whose value plausibly changes with time, treatment, transfusion, surgery, "
    "or disease progression.\n"
    "2. Write ONE sentence that is CO-PRESENT in the SAME encounter/report and DIRECTLY CONTRADICTS that "
    "finding, so the single record conflicts with itself. Frame it as a same-timepoint disagreement between "
    "sources in the same record, e.g. 'The impression of the same report instead states ...' or "
    "'Another section of the same pathology report documents ...'. "
    "STRICTLY FORBIDDEN: the words 'repeat', 'later', 'subsequent', 'follow-up', 'again', 'updated', "
    "'re-imaging', or any wording implying a second, separate, or time-later assessment. "
    "The contradiction must NOT be explainable by time, treatment, surgery, disease progression, or a "
    "repeat/new test.\n"
    "3. The contradiction must be specific to THIS case and reference the same finding. Do not invent a new "
    "unrelated finding. Do not resolve the conflict.\n"
    '4. If the case has no fixed/structural/same-specimen finding that can be contradicted without invoking '
    "time or treatment, set groundable=false (it is correct and expected to drop such cases).\n"
    'Return ONLY JSON: {"stated_finding": "...", "contradiction_sentence": "...", "groundable": true/false}.'
)


def read_key(name: str) -> str:
    val = os.environ.get(name)
    if val:
        return val.strip()
    if KEY_FILE.exists():
        m = re.search(rf"{name}=\s*(\S+)", KEY_FILE.read_text())
        if m:
            return m.group(1)
    raise RuntimeError(f"{name} not found")


def generate_one(model: str, item: dict) -> dict:
    from openai import OpenAI

    client = OpenAI(api_key=read_key("OPENAI_API_KEY"))
    case = str(item["input_text"])
    out = {"stated_finding": "", "contradiction_sentence": "", "groundable": False, "error": ""}
    for attempt in range(4):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": f"CLINICAL CASE:\n{case}"}],
                temperature=0,
                max_completion_tokens=400,
                response_format={"type": "json_object"},
            )
            parsed = json.loads(resp.choices[0].message.content or "{}")
            out.update({k: parsed.get(k, out[k]) for k in ("stated_finding", "contradiction_sentence", "groundable")})
            break
        except Exception as exc:  # noqa: BLE001
            out["error"] = f"{type(exc).__name__}: {exc}"
            time.sleep(2**attempt)
    out["_item"] = item
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=140)
    ap.add_argument("--model", default="gpt-5.4-mini")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=20260705)
    args = ap.parse_args()

    src = pd.read_csv(SOURCE_MANIFEST)
    base = src[src["perturbation_type"] == BASE_TYPE].drop_duplicates("item_id")
    base = base.sample(n=min(args.n, len(base)), random_state=args.seed)
    print(f"Generating contradictions for {len(base)} '{BASE_TYPE}' base cases with {args.model} ...")

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(generate_one, args.model, row.to_dict()) for _, row in base.iterrows()]
        for i, fut in enumerate(as_completed(futs), 1):
            results.append(fut.result())
            if i % 25 == 0:
                print(f"  {i}/{len(base)}")

    # Any contradiction still phrased as a time-later/repeat assessment defeats the construct
    # (it reads as a legitimate clinical update rather than a same-timepoint conflict) -> drop it.
    FORBIDDEN = re.compile(
        r"\b(repeat|later|subsequent|follow[- ]?up|again|updated|re-?imag\w*|re-?test\w*|re-?scan\w*)\b",
        re.IGNORECASE,
    )

    rows = []
    dropped = 0
    dropped_temporal = 0
    for r in results:
        c = str(r.get("contradiction_sentence", "")).strip()
        if not r.get("groundable") or not c or r.get("error"):
            dropped += 1
            continue
        if FORBIDDEN.search(c):
            dropped += 1
            dropped_temporal += 1
            continue
        item = r["_item"]
        new_text = f"{str(item['input_text']).strip()}\n\nAlso documented in the same record: {c}"
        rows.append({
            "item_id": item["item_id"],
            "perturbation_id": stable_hash_text(f"{item['item_id']}:conflicting_evidence_llm:{new_text}")[:16],
            "dataset": item.get("dataset", ""),
            "perturbation_type": "conflicting_evidence_llm",
            "input_text": new_text,
            "original_text_hash": stable_hash_text(str(item["input_text"])),
            "removed_fields": "",
            "synthetic_added_text": c,
            "stated_finding_contradicted": r.get("stated_finding", ""),
            "expected_missing_evidence": "resolution of contradictory evidence",
            "ground_truth_label": item.get("ground_truth_label", ""),
            "specialty": item.get("specialty", ""),
            "created_at": utc_now_iso(),
            "script_version": "contradiction_gen_0.1.0",
        })
    manifest = pd.DataFrame(rows)
    manifest.to_csv(OUT_MANIFEST, index=False)
    print(f"\nGroundable contradictions: {len(manifest)}  (dropped {dropped}; of which temporal-phrasing {dropped_temporal})")
    print(f"Wrote {OUT_MANIFEST}")

    # Doctor spot-check sheet (validity of the generated contradictions)
    SPOTCHECK.parent.mkdir(parents=True, exist_ok=True)
    sample = manifest.sample(n=min(25, len(manifest)), random_state=args.seed)
    check = pd.DataFrame({
        "check_id": [f"C{i:03d}" for i in range(1, len(sample) + 1)],
        "dataset": sample["dataset"].values,
        "finding_being_contradicted": sample["stated_finding_contradicted"].values,
        "generated_contradiction": sample["synthetic_added_text"].values,
        "full_case_with_contradiction": sample["input_text"].values,
        "is_valid_grounded_contradiction_0_1": "",
        "reviewer_note": "",
    })
    with pd.ExcelWriter(SPOTCHECK, engine="openpyxl") as xl:
        pd.DataFrame({"INSTRUCTIONS": [
            "Validate machine-generated contradictions before we spend on the full arm.",
            "For each row: read 'finding_being_contradicted' and 'generated_contradiction'.",
            "Mark is_valid_grounded_contradiction_0_1 = 1 ONLY IF the sentence directly conflicts with the",
            "SAME finding in the case AND the conflict CANNOT reasonably be explained by time, treatment,",
            "surgery, disease progression, or a repeat/later test. I.e. it must be a true same-timepoint,",
            "irreconcilable contradiction (e.g. same-specimen histopathology, a structural finding present",
            "vs absent), not a legitimate later update.",
            "Mark 0 if it is vague, unrelated, wrong, or merely a plausible later change.",
            "If >=90% are 1, the generator is trustworthy and we run the full response arm.",
        ]}).to_excel(xl, sheet_name="READ_ME", index=False)
        check.to_excel(xl, sheet_name="spotcheck", index=False)
    print(f"Wrote doctor spot-check: {SPOTCHECK} ({len(sample)} items)")
    print("\nSample (first 3):")
    for _, r in sample.head(3).iterrows():
        print(f"  finding: {str(r['stated_finding_contradicted'])[:80]}")
        print(f"   -> contradiction: {str(r['synthetic_added_text'])[:100]}")


if __name__ == "__main__":
    main()
