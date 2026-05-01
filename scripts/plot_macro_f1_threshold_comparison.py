#!/usr/bin/env python3
"""
Render a compact SVG comparing raw macro-F1 versus per-class threshold tuning.

The figure focuses on three representative experiments:
1. exp004_bertimbau        - early strong transformer baseline
2. exp010_bertimbau_es     - best single-model leaky-fold run
3. exp023_bertimbau_dedup  - honest group-aware baseline

For each experiment:
- metrics.json        = no threshold tuning
- metrics_tuned.json  = per-class threshold tuning
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from html import escape
from pathlib import Path


@dataclass(frozen=True)
class ThresholdComparison:
    title: str
    experiment_name: str
    raw_label: str
    tuned_label: str


COMPARISONS = [
    ThresholdComparison(
        title="Original BERTimbau baseline",
        experiment_name="exp004_bertimbau",
        raw_label="No threshold tuning",
        tuned_label="Per-class threshold tuning",
    ),
    ThresholdComparison(
        title="BERTimbau + early stopping",
        experiment_name="exp010_bertimbau_es",
        raw_label="No threshold tuning",
        tuned_label="Per-class threshold tuning",
    ),
    ThresholdComparison(
        title="Honest dedup + group-aware baseline",
        experiment_name="exp023_bertimbau_dedup",
        raw_label="No threshold tuning",
        tuned_label="Per-class threshold tuning",
    ),
]

RAW_COLOR = "#64748B"
TUNED_COLOR = "#0F766E"
DELTA_POSITIVE_COLOR = "#15803D"
DELTA_NEGATIVE_COLOR = "#B91C1C"
INK = "#0F172A"
MUTED = "#475569"
GRID = "#CBD5E1"
CONNECTOR = "#94A3B8"
BACKGROUND = "#FFFFFF"
PANEL = "#FCFCFD"
PANEL_STROKE = "#E2E8F0"


def load_macro_f1(repo_root: Path, experiment_name: str, metrics_name: str) -> float:
    metrics_path = repo_root / "experiments" / experiment_name / metrics_name
    with metrics_path.open() as handle:
        return float(json.load(handle)["macro_f1"])


def axis_bounds(values: list[float]) -> tuple[float, float]:
    min_value = min(values)
    max_value = max(values)
    axis_min = math.floor((min_value - 0.01) * 100.0) / 100.0
    axis_max = math.ceil((max_value + 0.01) * 100.0) / 100.0
    return axis_min, axis_max


def x_scale(value: float, axis_min: float, axis_max: float, x0: float, x1: float) -> float:
    return x0 + (value - axis_min) * (x1 - x0) / (axis_max - axis_min)


def format_delta(delta: float) -> str:
    return f"{delta:+.4f}"


def svg_text(x: float, y: float, text: str, **attrs: str) -> str:
    attributes = " ".join(f'{key}="{escape(str(value))}"' for key, value in attrs.items())
    return f'<text x="{x:.1f}" y="{y:.1f}" {attributes}>{escape(text)}</text>'


def render_svg(repo_root: Path) -> str:
    results = []
    for comparison in COMPARISONS:
        raw_macro_f1 = load_macro_f1(repo_root, comparison.experiment_name, "metrics.json")
        tuned_macro_f1 = load_macro_f1(repo_root, comparison.experiment_name, "metrics_tuned.json")
        results.append((comparison, raw_macro_f1, tuned_macro_f1))

    all_values = [value for _, raw_value, tuned_value in results for value in (raw_value, tuned_value)]
    axis_min, axis_max = axis_bounds(all_values)

    width = 1280
    height = 780
    left_margin = 72
    top_margin = 78
    bottom_margin = 72
    label_column_width = 420
    delta_column_width = 170
    plot_left = left_margin + label_column_width
    plot_right = width - 72 - delta_column_width
    plot_top = top_margin + 156
    plot_bottom = height - bottom_margin - 46
    row_gap = (plot_bottom - plot_top) / (len(results) - 1)
    row_ys = [plot_top + i * row_gap for i in range(len(results))]

    ticks = []
    tick = axis_min
    while tick <= axis_max + 1e-9:
        ticks.append(round(tick, 2))
        tick += 0.01

    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img" '
            f'aria-labelledby="title subtitle">'
        ),
        f'<rect width="{width}" height="{height}" fill="{BACKGROUND}"/>',
        (
            f'<rect x="36" y="28" width="{width - 72}" height="{height - 56}" rx="28" '
            f'fill="{PANEL}" stroke="{PANEL_STROKE}" stroke-width="1.5"/>'
        ),
        '<title id="title">Performance comparison: threshold tuning versus none</title>',
        (
            '<desc id="subtitle">'
            'Comparison of raw macro-F1 versus tuned macro-F1 after per-class threshold offsets '
            'for three representative experiments.'
            '</desc>'
        ),
    ]

    parts.append(
        svg_text(
            72,
            90,
            "Performance Comparison: Threshold Tuning vs None",
            fill=INK,
            **{"font-size": "30", "font-family": "Arial, Helvetica, sans-serif", "font-weight": "700"},
        )
    )
    parts.append(
        svg_text(
            72,
            124,
            "Macro-F1 before and after per-class threshold offsets on OOF probabilities",
            fill=MUTED,
            **{"font-size": "17", "font-family": "Arial, Helvetica, sans-serif"},
        )
    )

    legend_y = 126
    parts.append(f'<circle cx="974" cy="{legend_y - 6}" r="7" fill="{RAW_COLOR}"/>')
    parts.append(
        svg_text(
            990,
            legend_y,
            "No tuning",
            fill=INK,
            **{"font-size": "15", "font-family": "Arial, Helvetica, sans-serif"},
        )
    )
    parts.append(f'<circle cx="1080" cy="{legend_y - 6}" r="7" fill="{TUNED_COLOR}"/>')
    parts.append(
        svg_text(
            1096,
            legend_y,
            "Threshold tuning",
            fill=INK,
            **{"font-size": "15", "font-family": "Arial, Helvetica, sans-serif"},
        )
    )

    axis_title_y = plot_top - 58
    parts.append(
        svg_text(
            (plot_left + plot_right) / 2,
            axis_title_y,
            "Macro-F1",
            fill=INK,
            **{
                "font-size": "16",
                "font-family": "Arial, Helvetica, sans-serif",
                "font-weight": "700",
                "text-anchor": "middle",
            },
        )
    )

    grid_y0 = plot_top - 28
    grid_y1 = plot_bottom + 48
    for tick_value in ticks:
        x = x_scale(tick_value, axis_min, axis_max, plot_left, plot_right)
        parts.append(
            f'<line x1="{x:.1f}" y1="{grid_y0:.1f}" x2="{x:.1f}" y2="{grid_y1:.1f}" '
            f'stroke="{GRID}" stroke-width="1.3" stroke-dasharray="4 8"/>'
        )
        parts.append(
            svg_text(
                x,
                plot_top - 8,
                f"{tick_value:.2f}",
                fill=MUTED,
                **{
                    "font-size": "14",
                    "font-family": "Arial, Helvetica, sans-serif",
                    "text-anchor": "middle",
                },
            )
        )

    parts.append(
        f'<line x1="{plot_left:.1f}" y1="{grid_y0:.1f}" x2="{plot_left:.1f}" y2="{grid_y1:.1f}" '
        f'stroke="{CONNECTOR}" stroke-width="2"/>'
    )

    for (comparison, raw_macro_f1, tuned_macro_f1), y in zip(results, row_ys):
        raw_x = x_scale(raw_macro_f1, axis_min, axis_max, plot_left, plot_right)
        tuned_x = x_scale(tuned_macro_f1, axis_min, axis_max, plot_left, plot_right)
        delta = tuned_macro_f1 - raw_macro_f1
        delta_color = DELTA_POSITIVE_COLOR if delta >= 0 else DELTA_NEGATIVE_COLOR

        label_x = 72
        parts.append(
            svg_text(
                label_x,
                y - 26,
                comparison.title,
                fill=INK,
                **{"font-size": "22", "font-family": "Arial, Helvetica, sans-serif", "font-weight": "700"},
            )
        )
        parts.append(
            svg_text(
                label_x,
                y + 4,
                f"{comparison.raw_label} ({comparison.experiment_name})",
                fill=RAW_COLOR,
                **{"font-size": "16", "font-family": "Arial, Helvetica, sans-serif"},
            )
        )
        parts.append(
            svg_text(
                label_x,
                y + 28,
                comparison.tuned_label,
                fill=TUNED_COLOR,
                **{"font-size": "16", "font-family": "Arial, Helvetica, sans-serif"},
            )
        )

        parts.append(
            f'<line x1="{raw_x:.1f}" y1="{y:.1f}" x2="{tuned_x:.1f}" y2="{y:.1f}" '
            f'stroke="{CONNECTOR}" stroke-width="7" stroke-linecap="round"/>'
        )
        parts.append(f'<circle cx="{raw_x:.1f}" cy="{y:.1f}" r="11" fill="{RAW_COLOR}"/>')
        parts.append(f'<circle cx="{tuned_x:.1f}" cy="{y:.1f}" r="11" fill="{TUNED_COLOR}"/>')

        parts.append(
            svg_text(
                raw_x,
                y - 20,
                f"{raw_macro_f1:.4f}",
                fill=RAW_COLOR,
                **{
                    "font-size": "16",
                    "font-family": "Arial, Helvetica, sans-serif",
                    "font-weight": "700",
                    "text-anchor": "middle",
                },
            )
        )
        parts.append(
            svg_text(
                tuned_x,
                y + 34,
                f"{tuned_macro_f1:.4f}",
                fill=TUNED_COLOR,
                **{
                    "font-size": "16",
                    "font-family": "Arial, Helvetica, sans-serif",
                    "font-weight": "700",
                    "text-anchor": "middle",
                },
            )
        )

        parts.append(
            svg_text(
                plot_right + 36,
                y + 6,
                format_delta(delta),
                fill=delta_color,
                **{"font-size": "22", "font-family": "Arial, Helvetica, sans-serif", "font-weight": "700"},
            )
        )
        parts.append(
            svg_text(
                plot_right + 36,
                y + 28,
                "delta",
                fill=MUTED,
                **{"font-size": "13", "font-family": "Arial, Helvetica, sans-serif"},
            )
        )

    footnote = (
        "Metric source: experiments/*/metrics.json and metrics_tuned.json | "
        "Experiments: exp004, exp010, exp023"
    )
    parts.append(
        svg_text(
            72,
            height - 50,
            footnote,
            fill=MUTED,
            **{"font-size": "14", "font-family": "Arial, Helvetica, sans-serif"},
        )
    )
    parts.append("</svg>")
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="notebooks/figures/macro_f1_threshold_tuning_comparison.svg",
        help="Path to the output SVG file",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    output_path = repo_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_svg(repo_root), encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
