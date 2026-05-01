"""
Analyze exact-duplicate reports in the training set.

Usage:
    uv run python scripts/analyze_duplicates.py
    uv run python scripts/analyze_duplicates.py --top-k 10
    uv run python scripts/analyze_duplicates.py --show-examples --max-example-chars 400
"""
import argparse
import hashlib
import os
import sys

import pandas as pd

# Ensure project root is on path regardless of working directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import load_train


def normalize_report(text: str) -> str:
    """Match the quick audit logic: lowercase + strip surrounding whitespace."""
    if pd.isna(text):
        return ""
    return str(text).strip().lower()


def stable_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def truncate(text: str, max_chars: int) -> str:
    text = text.replace("\r", "\\r").replace("\n", "\\n")
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze exact-duplicate mammography reports")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Number of largest duplicate groups to print (default: 10)")
    parser.add_argument("--show-examples", action="store_true",
                        help="Print one report snippet for each displayed duplicate group")
    parser.add_argument("--show-conflicts", action="store_true", default=True,
                        help="Print exact-duplicate groups with conflicting labels (default: on)")
    parser.add_argument("--hide-conflicts", dest="show_conflicts", action="store_false",
                        help="Do not print conflicting-label groups")
    parser.add_argument("--max-conflicts", type=int, default=10,
                        help="Max conflicting groups to print (default: 10)")
    parser.add_argument("--max-example-chars", type=int, default=300,
                        help="Max characters to show for each example snippet (default: 300)")
    args = parser.parse_args()

    df = load_train().copy()
    df["report_norm"] = df["report"].apply(normalize_report)
    df["report_hash"] = df["report_norm"].apply(stable_hash)

    group_sizes = df["report_hash"].value_counts().rename("count")
    duplicate_sizes = group_sizes[group_sizes > 1]

    n_rows = len(df)
    n_groups = int(df["report_hash"].nunique())
    n_dup_groups = int((group_sizes > 1).sum())
    n_dup_rows = int(duplicate_sizes.sum())
    dup_ratio = (n_dup_rows / n_rows) if n_rows else 0.0

    print("=" * 70)
    print("Exact-Duplicate Report Audit")
    print("=" * 70)
    print(f"Rows total                 : {n_rows:,}")
    print(f"Unique normalized reports  : {n_groups:,}")
    print(f"Duplicate groups (>1 row)  : {n_dup_groups:,}")
    print(f"Rows in duplicate groups   : {n_dup_rows:,} ({dup_ratio:.1%})")
    print(f"Largest duplicate group    : {int(group_sizes.max() if len(group_sizes) else 0):,}")

    target_counts = df["target"].value_counts().sort_index()
    print("\nClass distribution:")
    print(target_counts.to_string())

    if n_dup_groups:
        print(f"\nTop {min(args.top_k, n_dup_groups)} duplicate groups:")
        top_hashes = duplicate_sizes.head(args.top_k)
        for i, (report_hash, count) in enumerate(top_hashes.items(), start=1):
            group = df[df["report_hash"] == report_hash]
            targets = sorted(group["target"].unique().tolist())
            print(f"{i:>2}. hash={report_hash}  rows={count}  targets={targets}")
            if args.show_examples:
                example = truncate(group["report"].iloc[0], args.max_example_chars)
                print(f"    example: {example}")

    target_uniques = df.groupby("report_hash")["target"].nunique().rename("n_unique_targets")
    conflict_hashes = target_uniques[target_uniques > 1].index.tolist()
    n_conflicts = len(conflict_hashes)

    print(f"\nConflicting duplicate groups: {n_conflicts}")
    if n_conflicts:
        summary = (
            df[df["report_hash"].isin(conflict_hashes)]
            .groupby("report_hash")
            .agg(rows=("target", "size"),
                 unique_targets=("target", lambda s: sorted(set(int(x) for x in s))))
            .reset_index()
            .sort_values(["rows", "report_hash"], ascending=[False, True])
        )
        target_unique_dist = target_uniques.value_counts().sort_index()
        print("\nUnique targets per normalized report:")
        print(target_unique_dist.to_string())

        if args.show_conflicts:
            print(f"\nShowing up to {min(args.max_conflicts, n_conflicts)} conflicting groups:")
            for _, row in summary.head(args.max_conflicts).iterrows():
                group = df[df["report_hash"] == row["report_hash"]][["target", "report"]]
                print(
                    f"\nHash={row['report_hash']}  rows={int(row['rows'])}  "
                    f"targets={row['unique_targets']}"
                )
                for rec in group.head(5).itertuples(index=False):
                    snippet = truncate(rec.report, args.max_example_chars)
                    print(f"  target={int(rec.target)}  report={snippet}")


if __name__ == "__main__":
    main()
