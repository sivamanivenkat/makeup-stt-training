"""
Drop specific segments from a packaged dataset split by their audiofolder
row index (as reported by eval_moonshine.py --sort wer_desc), plus the
source checkpoints/cleaned.jsonl so a future re-run of `node orchestrator.js
package` doesn't resurrect them.

Usage:
    python remove_segments.py --dataset ./dataset --split val --indices 24,26,82
    python remove_segments.py --dataset ./dataset --split val --indices 24,26,82 --cleaned checkpoints/cleaned.jsonl
"""

import argparse
import json
import os

from datasets import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove segments by dataset row index.")
    parser.add_argument("--dataset", default="./dataset", help="Dataset directory in audiofolder format.")
    parser.add_argument("--split", required=True, help="Split to remove from (e.g. val, train).")
    parser.add_argument("--indices", required=True, help="Comma-separated row indices within --split to drop.")
    parser.add_argument(
        "--cleaned",
        default=None,
        help="Optional checkpoints/cleaned.jsonl to also strip these segment_ids from.",
    )
    args = parser.parse_args()

    indices = [int(i) for i in args.indices.split(",")]

    raw_dataset = load_dataset("audiofolder", data_dir=args.dataset)
    dataset = raw_dataset[args.split]

    segment_ids = {dataset[i]["segment_id"] for i in indices}
    print(f"Dropping {len(segment_ids)} segment(s): {sorted(segment_ids)}")

    meta_path = os.path.join(args.dataset, args.split, "metadata.jsonl")
    with open(meta_path, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]

    kept = [line for line in lines if line["segment_id"] not in segment_ids]
    removed = len(lines) - len(kept)

    with open(meta_path, "w", encoding="utf-8") as f:
        for line in kept:
            f.write(json.dumps(line) + "\n")

    print(f"{meta_path}: removed {removed}, kept {len(kept)}")

    if args.cleaned:
        with open(args.cleaned, "r", encoding="utf-8") as f:
            cleaned_lines = [json.loads(line) for line in f if line.strip()]

        cleaned_kept = [line for line in cleaned_lines if line["segment_id"] not in segment_ids]
        cleaned_removed = len(cleaned_lines) - len(cleaned_kept)

        with open(args.cleaned, "w", encoding="utf-8") as f:
            for line in cleaned_kept:
                f.write(json.dumps(line) + "\n")

        print(f"{args.cleaned}: removed {cleaned_removed}, kept {len(cleaned_kept)}")


if __name__ == "__main__":
    main()
