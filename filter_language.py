"""
Strip clips from metadata.jsonl that detect_language.py flagged as non-English
or low-confidence, so training doesn't chase a WER floor caused by
audio/reference language mismatches.

Backs up each split's metadata.jsonl to metadata.jsonl.bak before rewriting,
so this is safe to rerun (rerun overwrites metadata.jsonl again from the same
.bak-preserved original — running it twice does not double-filter, since the
second run reads whatever is currently in metadata.jsonl; to re-filter from
scratch, restore metadata.jsonl.bak -> metadata.jsonl first).

Usage:
    python filter_language.py --dataset ./dataset --report ./checkpoints/language_report.jsonl
    python filter_language.py --dataset ./dataset --report ./checkpoints/language_report.jsonl --dry-run
"""

import argparse
import json
import os


def load_excluded(report_path: str, min_confidence: float) -> set:
    excluded = set()

    with open(report_path, "r", encoding="utf-8") as report_file:
        for line in report_file:
            line = line.strip()

            if not line:
                continue

            record = json.loads(line)
            is_non_english = record["language"] != "en"
            is_uncertain = record["language_probability"] < min_confidence

            if is_non_english or is_uncertain:
                excluded.add((record["split"], record["file_name"]))

    return excluded


def filter_split(dataset_dir: str, split: str, excluded: set, dry_run: bool) -> None:
    metadata_path = os.path.join(dataset_dir, split, "metadata.jsonl")

    if not os.path.isfile(metadata_path):
        print(f"  (no metadata.jsonl for split '{split}', skipping)")
        return

    kept_lines = []
    total = 0
    removed = 0

    with open(metadata_path, "r", encoding="utf-8") as metadata_file:
        for line in metadata_file:
            stripped = line.strip()

            if not stripped:
                continue

            total += 1
            example = json.loads(stripped)

            if (split, example["file_name"]) in excluded:
                removed += 1
                continue

            kept_lines.append(stripped)

    print(f"  {split}: {total} total, {removed} removed, {len(kept_lines)} kept")

    if dry_run:
        return

    backup_path = metadata_path + ".bak"

    if not os.path.isfile(backup_path):
        os.replace(metadata_path, backup_path)
    else:
        os.remove(metadata_path)

    with open(metadata_path, "w", encoding="utf-8") as metadata_file:
        for line in kept_lines:
            metadata_file.write(line + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter non-English / low-confidence clips out of metadata.jsonl."
    )

    parser.add_argument("--dataset", default="./dataset")
    parser.add_argument("--report", default="./checkpoints/language_report.jsonl")
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.5,
        help="Clips below this language_probability are dropped regardless of detected language.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print counts without modifying metadata.jsonl.",
    )

    args = parser.parse_args()

    excluded = load_excluded(args.report, args.min_confidence)
    print(f"Excluding {len(excluded)} clips (non-English or confidence < {args.min_confidence})")

    splits = [split.strip() for split in args.splits.split(",")]

    for split in splits:
        filter_split(args.dataset, split, excluded, args.dry_run)


if __name__ == "__main__":
    main()
