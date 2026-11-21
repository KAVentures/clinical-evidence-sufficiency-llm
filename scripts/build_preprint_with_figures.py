#!/usr/bin/env python3
"""Assemble the current, audited manuscript Markdown with all nine result figures.

The canonical narrative is manuscript/manuscript.md. This inserts figure PNGs
at stable section boundaries instead of reusing the stale pre-audit
'with_visible_figures' export. It does not change or reinterpret any data.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "manuscript" / "manuscript.md"
TARGET = ROOT / "manuscript" / "manuscript_with_figures.md"

FIGURES = [
    ("### Datasets", 1, "Study design and paired experiment workflow",
     "panel_figure1_study_design.png"),
    ("### Per-Model Results", 2, "AI-judge-assigned unsafe overconfidence by prompt condition",
     "panel_figure2_unsafe_overconfidence_by_prompt.png"),
    ("### Dataset and Perturbation Results", 3, "Paired, judge-assigned risk differences by model",
     "panel_figure3_risk_difference_by_model.png"),
    ("### Mechanism and Circularity Control", 4, "Paired risk differences by perturbation",
     "panel_figure4_risk_difference_by_perturbation.png"),
    ("### Mechanism and Circularity Control", 5, "Paired risk differences by dataset",
     "panel_figure5_risk_difference_by_dataset.png"),
    ("### Secondary Outcomes", 6, "Matched control-arm AI-judge-assigned rates",
     "panel_figure6_control_arm_rates.png"),
    ("### Secondary Outcomes", 7, "Matched contrast decomposition under the primary AI judge",
     "panel_figure7_mechanism_decomposition.png"),
    ("### Helpfulness and Accuracy Trade-off", 8,
     "Directional disagreements between two AI judges, not clinical truth",
     "panel_figure8_cross_judge_overlabeling.png"),
    ("### Prompt-Paraphrase and Decode Robustness", 9,
     "Exploratory safety-helpfulness trade-off measured by a single correctness judge",
     "panel_figure9_safety_helpfulness_tradeoff.png"),
]


def build():
    doc = SOURCE.read_text(encoding="utf-8")
    for boundary in {b for b, _, _, _ in FIGURES}:
        if doc.count("\n" + boundary + "\n") != 1:
            raise ValueError("Expected exactly one section heading: " + boundary)
    insertions = {}
    for boundary, number, caption, png in FIGURES:
        image = ROOT / "outputs" / "figures" / png
        if not image.is_file():
            raise FileNotFoundError(image)
        insertions.setdefault(boundary, []).append(
            f"![Figure {number}. {caption}](outputs/figures/{png})"
        )

    for boundary, blocks in insertions.items():
        doc = doc.replace("\n" + boundary + "\n",
                          "\n\n" + "\n\n".join(blocks) +
                          "\n\n" + boundary + "\n", 1)

    for number in range(1, 10):
        if doc.count(f"![Figure {number}. ") != 1:
            raise ValueError(f"Figure {number} missing or repeated")
    TARGET.write_text(doc, encoding="utf-8")
    print(f"Generated {TARGET.relative_to(ROOT)} with 9 figures from audited source")


if __name__ == "__main__":
    build()
