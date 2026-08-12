"""
WhisperX batch transcriber.
Called by orchestrator.js as a subprocess — loads model once, processes all segments.

Usage:
    python agents/3-transcriber.py \
        --manifest  checkpoints/downloaded.jsonl \
        --output    checkpoints/transcribed.jsonl \
        --model     large-v3 \
        --device    cuda \
        --batch-size 16
"""

import argparse
import json
import os
import sys

import torch
import whisperx


def load_done(output_path):
    done = set()
    if os.path.exists(output_path):
        with open(output_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        done.add(json.loads(line)["segment_id"])
                    except Exception:
                        pass
    return done


def transcribe_segment(model, align_model, align_metadata, entry, device, batch_size):
    audio = whisperx.load_audio(entry["wavPath"])
    result = model.transcribe(audio, batch_size=batch_size, language="en")

    if not result.get("segments"):
        return {"raw_text": "", "words": [], "segments": []}

    aligned = whisperx.align(
        result["segments"],
        align_model,
        align_metadata,
        audio,
        device,
        return_char_alignments=False,
    )

    raw_text = " ".join(s["text"].strip() for s in aligned.get("segments", []))
    words = [
        {
            "word": w["word"],
            "start": w.get("start", 0),
            "end": w.get("end", 0),
            "score": w.get("score", 0),
        }
        for seg in aligned.get("segments", [])
        for w in seg.get("words", [])
    ]

    return {
        "raw_text": raw_text.strip(),
        "words": words,
        "segments": aligned.get("segments", []),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="large-v3")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    compute_type = "float16" if args.device == "cuda" else "int8"
    print(
        f"[transcriber] device={args.device} model={args.model} compute={compute_type}",
        flush=True,
    )

    manifest = []
    with open(args.manifest) as f:
        for line in f:
            line = line.strip()
            if line:
                manifest.append(json.loads(line))

    done = load_done(args.output)
    todo = [e for e in manifest if e["segmentId"] not in done]
    print(
        f"[transcriber] {len(todo)} segments to transcribe ({len(done)} already done)",
        flush=True,
    )
    if not todo:
        print("[transcriber] Nothing to do.", flush=True)
        return

    print("[transcriber] Loading WhisperX model...", flush=True)
    model = whisperx.load_model(args.model, args.device, compute_type=compute_type)
    align_model, align_metadata = whisperx.load_align_model(
        language_code="en", device=args.device
    )
    print("[transcriber] Model loaded. Starting transcription...", flush=True)

    ok, failed = 0, 0
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    with open(args.output, "a") as out_f:
        for i, entry in enumerate(todo):
            try:
                result = transcribe_segment(
                    model,
                    align_model,
                    align_metadata,
                    entry,
                    args.device,
                    args.batch_size,
                )
                record = {
                    "segment_id": entry["segmentId"],
                    "video_id": entry["videoId"],
                    "step_num": entry["stepNum"],
                    "split": entry["split"],
                    "wav_path": entry["wavPath"],
                    "area": entry.get("area", []),
                    "caption": entry.get("caption", ""),
                    "raw_text": result["raw_text"],
                    "words": result["words"],
                }
                out_f.write(json.dumps(record) + "\n")
                out_f.flush()
                ok += 1
            except Exception as e:
                print(
                    f"[transcriber] SKIP {entry['segmentId']}: {e}",
                    file=sys.stderr,
                    flush=True,
                )
                failed += 1

            if (i + 1) % 100 == 0 or i == len(todo) - 1:
                print(
                    f"[transcriber] {i + 1}/{len(todo)} — {ok} OK / {failed} failed",
                    flush=True,
                )

    print(f"[transcriber] Done. {ok} transcribed, {failed} failed.", flush=True)


if __name__ == "__main__":
    main()
