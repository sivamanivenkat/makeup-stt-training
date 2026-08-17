"""
Build a manifest for agents/3-transcriber.py directly from the already-
packaged dataset (dataset/{split}/metadata.jsonl + dataset/{split}/audio/),
instead of the original checkpoints/downloaded.jsonl.

Why: re-transcription only needs to touch the audio files actually used for
training. The original per-segment raw extracts (temp/audio/segments/) are
gone from local disk and weren't backed up (only temp/audio/raw/, the raw
per-video downloads, got zipped to Drive) -- but the packaged dataset WAVs
are already sitting in Colab from dataset_clean.zip and are functionally
identical for transcription purposes (just resampled to 16kHz mono, which
Whisper does internally anyway).

Usage:
    python build_transcribe_manifest.py --dataset ./dataset --output ./checkpoints/retranscribe_manifest.jsonl
"""

import argparse
import json
import os


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="./dataset")
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--output", default="./checkpoints/retranscribe_manifest.jsonl")

    args = parser.parse_args()

    splits = [s.strip() for s in args.splits.split(",")]
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    total = 0

    with open(args.output, "w", encoding="utf-8") as out_f:
        for split in splits:
            metadata_path = os.path.join(args.dataset, split, "metadata.jsonl")

            if not os.path.isfile(metadata_path):
                print(f"  (no metadata.jsonl for split '{split}', skipping)")
                continue

            with open(metadata_path, "r", encoding="utf-8") as metadata_file:
                for line in metadata_file:
                    line = line.strip()

                    if not line:
                        continue

                    example = json.loads(line)
                    wav_path = os.path.join(args.dataset, split, example["file_name"])

                    record = {
                        "segmentId": example["segment_id"],
                        "videoId": example.get("video_id", ""),
                        "stepNum": example["segment_id"].split("_step")[-1]
                        if "_step" in example["segment_id"]
                        else "",
                        "split": split,
                        "area": example.get("area", []),
                        "caption": example.get("caption", ""),
                        "wavPath": wav_path,
                    }

                    out_f.write(json.dumps(record) + "\n")
                    total += 1

    print(f"Wrote {total} manifest entries -> {args.output}")


if __name__ == "__main__":
    main()
