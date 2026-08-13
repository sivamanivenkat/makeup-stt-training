"""
Scan dataset audio clips for spoken language, to find examples where the
reference transcript is an English translation/caption of non-English
audio rather than an actual transcription of what's said.

Moonshine (and any English ASR model) can't produce a correct output for
those clips no matter how well trained -- the "correct" text doesn't match
the audio. This is a real, structural WER floor if present in any volume,
not ordinary label noise.

Uses faster-whisper's built-in language detection (cheap: processes only
the first ~30s per clip, no full transcription). Deliberately does not use
whisperx, which pulls in a transformers version that conflicts with the
Moonshine training environment.

Requires:
    pip install faster-whisper

Usage:
    python detect_language.py --dataset ./dataset --output ./checkpoints/language_report.jsonl
    python detect_language.py --dataset ./dataset --split train --model small
"""

import argparse
import json
import os

from faster_whisper import WhisperModel


def iter_examples(dataset_dir: str, split: str):
    metadata_path = os.path.join(dataset_dir, split, "metadata.jsonl")

    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(
            f"No metadata.jsonl for split '{split}': {metadata_path}"
        )

    with open(metadata_path, "r", encoding="utf-8") as metadata_file:
        for line in metadata_file:
            line = line.strip()

            if not line:
                continue

            example = json.loads(line)
            example["split"] = split
            example["audio_path"] = os.path.join(
                dataset_dir, split, example["file_name"]
            )

            yield example


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect spoken language per clip to find audio/reference language mismatches."
    )

    parser.add_argument(
        "--dataset",
        default="./dataset",
        help="Dataset directory in Hugging Face audiofolder format.",
    )

    parser.add_argument(
        "--splits",
        default="train,val,test",
        help="Comma-separated splits to scan. Directory names as on disk.",
    )

    parser.add_argument(
        "--model",
        default="tiny",
        help="faster-whisper model size for language ID. tiny is enough.",
    )

    parser.add_argument(
        "--device",
        default="cuda",
        help="cuda or cpu.",
    )

    parser.add_argument(
        "--compute-type",
        default="float16",
        help="float16 on GPU, int8 or float32 on CPU.",
    )

    parser.add_argument(
        "--output",
        default="./checkpoints/language_report.jsonl",
        help="Where to write the per-clip language report.",
    )

    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.5,
        help="Below this language_probability, flag the clip as uncertain rather than trusting the label.",
    )

    args = parser.parse_args()

    print(
        f"Loading faster-whisper '{args.model}' on {args.device} ({args.compute_type})..."
    )

    model = WhisperModel(
        args.model,
        device=args.device,
        compute_type=args.compute_type,
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    splits = [split.strip() for split in args.splits.split(",")]

    total = 0
    non_english = 0
    uncertain = 0
    non_english_by_video = {}

    with open(args.output, "w", encoding="utf-8") as output_file:
        for split in splits:
            print(f"Scanning split: {split}")

            for example in iter_examples(args.dataset, split):
                total += 1

                _, info = model.transcribe(
                    example["audio_path"],
                    task="transcribe",
                    without_timestamps=True,
                )

                is_uncertain = info.language_probability < args.min_confidence
                is_non_english = info.language != "en"

                if is_uncertain:
                    uncertain += 1

                if is_non_english:
                    non_english += 1
                    video_id = example.get("video_id", "unknown")
                    non_english_by_video[video_id] = (
                        non_english_by_video.get(video_id, 0) + 1
                    )

                record = {
                    "segment_id": example.get("segment_id"),
                    "video_id": example.get("video_id"),
                    "split": split,
                    "file_name": example["file_name"],
                    "language": info.language,
                    "language_probability": round(info.language_probability, 3),
                }

                output_file.write(json.dumps(record) + "\n")

                if total % 500 == 0:
                    print(
                        f"  {total} scanned  non_english={non_english}  "
                        f"uncertain={uncertain}"
                    )

    print("=" * 70)
    print(f"Total clips scanned: {total}")
    print(f"Non-English detected: {non_english} ({100 * non_english / total:.1f}%)")
    print(f"Low-confidence (<{args.min_confidence}): {uncertain}")
    print(f"Report written to: {args.output}")

    if non_english_by_video:
        print("-" * 70)
        print("Non-English clips by video_id (top 20):")

        top_videos = sorted(
            non_english_by_video.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:20]

        for video_id, count in top_videos:
            print(f"  {video_id}: {count} clips")

    print("=" * 70)


if __name__ == "__main__":
    main()
