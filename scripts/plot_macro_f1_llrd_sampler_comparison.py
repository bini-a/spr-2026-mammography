#!/usr/bin/env python3
"""
Render a compact SVG comparing two honest ablations against the same baseline:

1. No LLRD vs LLRD
2. No weighted sampler vs WeightedRandomSampler

The clean reference for both is exp023_bertimbau_dedup, since exp017 and exp018
explicitly describe themselves as using the same honest pipeline as exp023 with
only the respective ablation changed.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from html import escape
from pathlib import Path


@dataclass(frozen=True)
class Comparison:
    title: str
    reference_label: str
    reference_experiment: str
    variant_label: str
    variant_experiment: str


COMPARISONS = [
    Comparison(
        title="Layer-wise LR decay",
        reference_label="No LLRD (exp023)",
        reference_experiment="exp023_bertimbau_dedup",
        variant_label="LLRD decay=0.9 (exp017)",
        variant_experiment="exp017_bertimbau_llrd",
    ),
    Comparison(
        title="Batch sampling",
        reference_label="No weighted sampler (exp023)",
        reference_experiment="exp023_bertimbau_dedup",
        variant_label="WeightedRandomSampler (exp018)",
        variant_experiment="exp018_bertimbau_weighted_sampler",
    ),
]

REFERENCE_COLOR = "#0F766E"
VARIANT_COLOR = "#C2410C"
DELTA_POSITIVE_COLOR = "#15803D"
DELTA_NEGATIVE_COLOR = "#B91C1C"
INK = "#0F172A"
MUTED = "#475569"
GRID = "#CBD5E1"
CONNECTOR = "#94A3B8"
BACKGROUND = "#FFFFFF"
PANEL = "#FCFCFD"
PANEL_STROKE = "#E2E8F0"


def load_macro_f1(repo_root: Path, experiment_name: str) -> float:
    metrics_path = repo_root / "experiments" / experiment_name / "metrics_tuned.json"
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
        reference_macro_f1 = load_macro_f1(repo_root, comparison.reference_experiment)
        variant_macro_f1 = load_macro_f1(repo_root, comparison.variant_experiment)
        results.append((comparison, reference_macro_f1, variant_macro_f1))

    all_values = [value for _, left_value, right_value in results for value in (left_value, right_value)]
    axis_min, axis_max = axis_bounds(all_values)

    width = 1280
    height = 620
    left_margin = 72
    top_margin = 78
    bottom_margin = 72
    label_column_width = 420
    delta_column_width = 170
    plot_left = left_margin + label_column_width
    plot_right = width - 72 - delta_column_width
    plot_top = top_margin + 156
    plot_bottom = height - bottom_margin - 40
    row_gap = (plot_bottom - plot_top) / (len(results) - 1)
    row_ys = [plot_top + i * row_gap for i in range(len(results))]

    ticks = []
    tick = axis_min
    while tick <= axis_max + 1e-9:
        ticks.append(round(tick, 2))
        tick += 0.02

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
        '<title id="title">Performance comparison: LLRD and weighted sampling</title>',
        '<desc id="subtitle"></desc>',
    ]

    parts.append(
        svg_text(
            72,
            90,
            "Performance Comparison: LLRD and Weighted Sampling",
            fill=INK,
            **{"font-size": "30", "font-family": "Arial, Helvetica, sans-serif", "font-weight": "700"},
        )
    )

    legend_y = 126
    parts.append(f'<circle cx="924" cy="{legend_y - 6}" r="7" fill="{REFERENCE_COLOR}"/>')
    parts.append(
        svg_text(
            940,
            legend_y,
            "Reference setup",
            fill=INK,
            **{"font-size": "15", "font-family": "Arial, Helvetica, sans-serif"},
        )
    )
    parts.append(f'<circle cx="1074" cy="{legend_y - 6}" r="7" fill="{VARIANT_COLOR}"/>')
    parts.append(
        svg_text(
            1090,
            legend_y,
            "Compared variant",
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
            f'stroke="{GRID}" stroke-width="1.5" stroke-dasharray="4 8"/>'
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

    for (comparison, reference_macro_f1, variant_macro_f1), y in zip(results, row_ys):
        reference_x = x_scale(reference_macro_f1, axis_min, axis_max, plot_left, plot_right)
        variant_x = x_scale(variant_macro_f1, axis_min, axis_max, plot_left, plot_right)
        delta = variant_macro_f1 - reference_macro_f1
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
                comparison.reference_label,
                fill=REFERENCE_COLOR,
                **{"font-size": "16", "font-family": "Arial, Helvetica, sans-serif"},
            )
        )
        parts.append(
            svg_text(
                label_x,
                y + 28,
                comparison.variant_label,
                fill=VARIANT_COLOR,
                **{"font-size": "16", "font-family": "Arial, Helvetica, sans-serif"},
            )
        )

        parts.append(
            f'<line x1="{reference_x:.1f}" y1="{y:.1f}" x2="{variant_x:.1f}" y2="{y:.1f}" '
            f'stroke="{CONNECTOR}" stroke-width="7" stroke-linecap="round"/>'
        )
        parts.append(f'<circle cx="{reference_x:.1f}" cy="{y:.1f}" r="11" fill="{REFERENCE_COLOR}"/>')
        parts.append(f'<circle cx="{variant_x:.1f}" cy="{y:.1f}" r="11" fill="{VARIANT_COLOR}"/>')

        parts.append(
            svg_text(
                reference_x,
                y - 20,
                f"{reference_macro_f1:.4f}",
                fill=REFERENCE_COLOR,
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
                variant_x,
                y + 34,
                f"{variant_macro_f1:.4f}",
                fill=VARIANT_COLOR,
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
        "Metric source: experiments/*/metrics_tuned.json | "
        "Pairs: exp023 vs exp017, exp023 vs exp018"
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
        default="notebooks/figures/macro_f1_llrd_sampler_comparison.svg",
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
