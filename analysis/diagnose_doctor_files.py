"""One-off diagnostics for the returned doctor packets.

(a) Confirm Doctor C's ADJUDICATION submission is templated/unreliable (vs C's own
    generic packet), by comparing unsafe rates and counting templated rationales.
(b) Quantify TRUNCATION across all four frontier panel prediction files so we know how
    big the response-generation QA confound is.

No API. Pure local read.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DL = Path(os.environ.get("DOCTOR_FILLED_DIR", str(ROOT / "outputs" / "doctor_review")))
PRED_DIR = ROOT / "outputs/predictions"
SLUGS = {
    "gpt-5.5": "openai_gpt55",
    "claude-opus-4-8": "anthropic_claude_opus_48",
    "gemini-3.5-flash": "google_gemini_35_flash",
    "grok-4.3": "xai_grok_43",
}


def load_ratings(path: Path) -> pd.DataFrame:
    xl = pd.ExcelFile(path)
    sheet = "ratings" if "ratings" in xl.sheet_names else xl.sheet_names[0]
    return xl.parse(sheet)


def find_col(df: pd.DataFrame, *cands: str) -> str | None:
    low = {c.lower().strip(): c for c in df.columns}
    for c in cands:
        if c in low:
            return low[c]
    for c in cands:
        for k, v in low.items():
            if c in k:
                return v
    return None


def unsafe_rate(df: pd.DataFrame) -> tuple[float, int, str]:
    col = find_col(df, "unsafe_overconfident", "unsafe", "rating", "label")
    if col is None:
        return (float("nan"), 0, "?")
    s = df[col].astype(str).str.strip().str.lower()
    mapping = {"1": 1, "yes": 1, "true": 1, "unsafe": 1, "unsafe_overconfident": 1,
               "0": 0, "no": 0, "false": 0, "safe": 0}
    v = s.map(mapping)
    n = int(v.notna().sum())
    return (float(v.mean()) if n else float("nan"), n, col)


def rationale_templating(df: pd.DataFrame) -> dict:
    col = find_col(df, "doctor_rationale", "rationale", "notes", "comment")
    if col is None:
        return {"rationale_col": None}
    txt = df[col].astype(str)
    n = len(txt)
    starts_overconf = int(txt.str.strip().str.startswith("**Overconfident").sum())
    has_uses_language = int(txt.str.contains(r"uses language like", case=False, na=False).sum())
    uniq = int(txt.str.strip().nunique())
    return {"rationale_col": col, "n": n, "starts_**Overconfident**": starts_overconf,
            "contains_'uses language like'": has_uses_language,
            "distinct_rationales": uniq, "distinct_frac": round(uniq / n, 3) if n else None}


def truncation_scan() -> dict:
    """Heuristics for truncated/malformed generations across the frontier panel files."""
    pat_midword = re.compile(r"[A-Za-z,;:\-]$")  # ends without terminal punctuation
    results = {}
    total = {"n": 0, "no_terminal_punct": 0, "very_short": 0, "empty": 0, "unclosed_md": 0}
    for model, slug in SLUGS.items():
        p = PRED_DIR / f"{slug}_public_study.jsonl"
        if not p.exists():
            results[model] = {"error": "missing"}
            continue
        n = no_term = short = empty = unclosed = 0
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = (r.get("response_text") or "").strip()
            n += 1
            if not t:
                empty += 1
                continue
            words = len(t.split())
            last = t[-1]
            if last not in ".!?)\"'”’*`]}":
                no_term += 1
            if words < 20:
                short += 1
            # unbalanced markdown bold markers -> often mid-render truncation
            if t.count("**") % 2 == 1:
                unclosed += 1
        results[model] = {"n": n, "empty": empty, "no_terminal_punct": no_term,
                          "no_terminal_punct_frac": round(no_term / n, 3) if n else None,
                          "very_short_lt20w": short, "unclosed_bold": unclosed}
        total["n"] += n
        total["no_terminal_punct"] += no_term
        total["very_short"] += short
        total["empty"] += empty
        total["unclosed_md"] += unclosed
    results["_panel_total"] = total
    return results


def main() -> None:
    print("=" * 70)
    print("(a) DOCTOR C ADJUDICATION RELIABILITY")
    print("=" * 70)
    for who, adj_name, gen_name in [
        ("A", "adjudication_A_reviewed.xlsx", "doctor_review_A_completed.xlsx"),
        ("B", "adjudication_B_reviewed.xlsx", "doctor_review_B_filled.xlsx"),
        ("C", "adjudication_C_filled.xlsx", "doctor_review_C_draft.xlsx"),
    ]:
        adj = DL / adj_name
        gen = DL / gen_name
        line = f"Doctor {who}: "
        if adj.exists():
            r, n, col = unsafe_rate(load_ratings(adj))
            line += f"adjudication unsafe={r:.3f} (n={n}, col={col})  "
        if gen.exists():
            rg, ng, cg = unsafe_rate(load_ratings(gen))
            line += f"generic unsafe={rg:.3f} (n={ng})"
        print(line)
        if adj.exists():
            print("        templating:", rationale_templating(load_ratings(adj)))

    print()
    print("=" * 70)
    print("(b) TRUNCATION / MALFORMED-GENERATION SCAN (frontier panel)")
    print("=" * 70)
    trunc = truncation_scan()
    print(json.dumps(trunc, indent=2))

    out = ROOT / "outputs/tables/doctor_file_diagnostics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"truncation": trunc}, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
