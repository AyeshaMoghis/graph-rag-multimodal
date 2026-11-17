"""Calibration plot: confidence bucket vs. actual keyword recall.

A calibrated confidence signal shows recall falling from high to medium to
low. The original pipeline reported "high" for 49 of 50 answers, so its
self-reported confidence could not separate good answers from bad ones.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

LEVELS = ("high", "medium", "low")
SOURCES = {
    "confidence_self_reported": "Model self-report",
    "confidence": "Retrieval evidence",
}


def calibration_table(results: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, float]]]:
    good = [r for r in results if "error" not in r]
    table: dict[str, dict[str, dict[str, float]]] = {}
    for key in SOURCES:
        table[key] = {}
        for level in LEVELS:
            rows = [r for r in good if r.get(key) == level]
            if rows:
                table[key][level] = {
                    "n": len(rows),
                    "keyword_recall": round(sum(r["keyword_recall"] for r in rows) / len(rows), 3),
                }
    return table


def plot_calibration(results_path: Path, out_path: Path) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = json.loads(results_path.read_text(encoding="utf-8"))["results"]
    table = calibration_table(results)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, (key, title) in zip(axes, SOURCES.items(), strict=True):
        levels = [lv for lv in LEVELS if lv in table[key]]
        recalls = [table[key][lv]["keyword_recall"] for lv in levels]
        counts = [int(table[key][lv]["n"]) for lv in levels]
        bars = ax.bar(levels, recalls, color="#4c72b0")
        for bar, n in zip(bars, counts, strict=True):
            ax.annotate(
                f"n={n}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center",
                va="bottom",
                fontsize=9,
            )
        ax.set_title(title)
        ax.set_xlabel("Confidence bucket")
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Mean keyword recall")
    fig.suptitle("Does confidence track answer quality?")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return table
