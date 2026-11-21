#!/usr/bin/env python3
"""Offline verification of computational results and exclusion of invalid reviewer files.

Earlier AI-generated reviewer workbooks must never be interpreted as physician
reference data. This check validates archived computational summaries only.
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs" / "tables"
REVIEW = ROOT / "outputs" / "doctor_review"


def report_json(filename):
    return json.loads((TABLES / filename).read_text(encoding="utf-8"))


def approx(value, expected, tol=1e-8):
    assert abs(value - expected) <= tol, (value, expected)


def check_computational():
    panel = report_json("panel_analysis_summary.json")
    p = panel["primary_common_panel"]
    assert p["n_pairs"] == 1200
    assert panel["mcnemar"]["standard_unsafe_wrapper_safe"] == 348
    assert panel["mcnemar"]["standard_safe_wrapper_unsafe"] == 52
    approx(p["risk_difference"], 296 / 1200)
    approx(panel["primary_rates"]["standard"], 592 / 1200)
    approx(panel["primary_rates"]["evidence_sufficiency"], 296 / 1200)
    assert sum(x["n_pairs"] for x in panel["per_model"]) == 1200

    cross = report_json("crossjudge_agreement_report.json")
    assert cross["n_cells_double_judged"] == 3534
    assert cross["agreement"]["nano_unsafe_cross_safe"] == 1147
    assert cross["agreement"]["nano_safe_cross_unsafe"] == 15
    approx(cross["primary_rd_common_panel"]["cross_rd"], 157 / 1200)

    accuracy = report_json("accuracy_tradeoff_report.json")
    correct = accuracy["overall"]["correct_diagnosis"]
    assert correct["n_pairs"] == 330
    assert accuracy["answerable_item_count"] == 200
    assert sum(x["correct_diagnosis"]["n_pairs"] for x in
               accuracy["per_model"].values()) == 330
    approx(correct["std_rate"], 265 / 330)
    approx(correct["es_rate"], 166 / 330)

    print("PASS: paired computational, cross-judge and single-judge correctness summary checks")


def check_review_sources():
    out = {}
    for family, expected in [("doctor_review", 120), ("adjudication", 90),
                             ("sensitivity", 72)]:
        for rater in "ABC":
            name = f"{family}_{rater}.csv"
            with (REVIEW / name).open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            assert len(rows) == expected, (name, len(rows), expected)
            labeled = sum(row["unsafe_overconfident_0_1"].strip() in ("0", "1")
                          for row in rows)
            out[name] = {"rows": len(rows), "filled_unsafe_labels": labeled}
    if any(x["filled_unsafe_labels"] != 0 for x in out.values()):
        raise AssertionError("Unexpected populated reviewer templates: provenance is "
                             "unauthenticated. Do not call these human labels.")
    assert (REVIEW / "PROVENANCE_WARNING.md").is_file()
    assert (TABLES / "PROVENANCE_WARNING.md").is_file()
    print("PASS: historical AI-review artifacts explicitly separated from scientific results")
    return out


def main():
    check_computational()
    check_review_sources()


if __name__ == "__main__":
    main()
