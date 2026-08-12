# SKILL: YouMakeup STT Pipeline

## Purpose
Run the YouTube-to-transcript pipeline for MakeUP AI STT training data.
Produces a HuggingFace audiofolder dataset ready for Whisper fine-tuning.

## Entry point
node orchestrator.js [download|transcribe|cleanup|package]

Omit phase to run all four sequentially.

## Pre-flight checks (always verify before running)
1. yt-dlp installed?    →  yt-dlp --version
2. ffmpeg installed?    →  ffmpeg -version
3. CUDA available?      →  python -c "import torch; print(torch.cuda.is_available())"
4. .env configured?     →  grep ANTHROPIC_API_KEY .env
5. YouMakeup data here? →  ls ./data/train/train_steps.json ./data/val/val_steps.json

## Resume after failure
All phases are idempotent. Re-run any phase — it skips already-processed entries.
Check ./checkpoints/failed.jsonl for permanently-skipped entries and their reasons.

## Tuning knobs
| Symptom               | Fix                             |
|-----------------------|---------------------------------|
| yt-dlp rate limited   | Set WORKERS=1 in .env           |
| WhisperX CUDA OOM     | Set WHISPER_MODEL=medium in .env|
| Claude 429 errors     | Auto-handled with backoff        |
| Video unavailable     | Logged to failed.jsonl, skipped  |

## Expected output volume (1000 videos)
- ~9,000 WAV segments, ~8-10 GB audio
- ~9,000 lines in cleaned.jsonl
- Train: 700 videos, Val: 150, Test: 150

## Fine-tune after pipeline
python train.py
Expects dataset/ directory from package phase.
Saves model to ./model/final/.
