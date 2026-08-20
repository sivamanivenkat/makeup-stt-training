"""
Convert a packaged dataset split's metadata.jsonl into the {audio, text}
manifest format expected by `moonshine-voice lora --train-manifest` /
`--eval-manifest`.

Field names only change here (file_name -> audio, transcription -> text);
the audio path stays relative to the split directory, so pass
`--data-root ./dataset/<split>` to the lora CLI to match.

Usage:
    python build_lora_manifest.py --dataset ./dataset --split train --output checkpoints/lora_train_manifest.jsonl
    python build_lora_manifest.py --dataset ./dataset --split val --output checkpoints/lora_eval_manifest.jsonl
"""

import argparse
import json
import os

import soundfile as sf

# Reproducibly crashes moonshine-voice lora's loader (soundfile.LibsndfileError:
# "System error" at open time) despite reading fine in every local check here --
# unexplained, environment-specific flakiness rather than a real corrupt file.
# Hard-excluded rather than spending more time chasing one row out of 13k+.
_KNOWN_BAD_FILES = {
    "audio/az8Tzf4OyJY_step2.wav",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a moonshine-voice lora manifest from a packaged split.")
    parser.add_argument("--dataset", default="./dataset", help="Dataset directory in audiofolder format.")
    parser.add_argument("--split", required=True, help="Split folder to convert (e.g. train, val).")
    parser.add_argument("--output", required=True, help="Output manifest path.")
    args = parser.parse_args()

    meta_path = os.path.join(args.dataset, args.split, "metadata.jsonl")
    with open(meta_path, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    split_dir = os.path.join(args.dataset, args.split)

    written = 0
    skipped_empty = 0
    skipped_missing = 0
    skipped_known_bad = 0
    with open(args.output, "w", encoding="utf-8") as f:
        for entry in lines:
            text = (entry.get("transcription") or "").strip()
            if not text:
                skipped_empty += 1
                continue
            if entry["file_name"] in _KNOWN_BAD_FILES:
                skipped_known_bad += 1
                continue
            audio_path = os.path.join(split_dir, entry["file_name"])
            try:
                # Full decode, not sf.info() -- a file truncated mid-write by
                # an interrupted unzip can have a valid header (info() passes)
                # while the actual audio data is short/corrupt (read() fails).
                # moonshine-voice's own loader calls sf.read(), so this needs
                # to match or it won't catch what actually crashes training.
                sf.read(audio_path, dtype="float32")
            except Exception:
                skipped_missing += 1
                continue
            f.write(json.dumps({"audio": entry["file_name"], "text": text}) + "\n")
            written += 1

    print(
        f"{args.output}: wrote {written} rows "
        f"(skipped {skipped_empty} empty, {skipped_missing} unreadable audio, "
        f"{skipped_known_bad} known-bad)"
    )
    print(f"Pass --data-root {os.path.join(args.dataset, args.split)} to moonshine-voice lora")


if __name__ == "__main__":
    main()
