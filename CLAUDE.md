# MakeUP AI — STT Training Data Pipeline

## Purpose
Download 1,000 YouMakeup YouTube videos, extract makeup-step audio segments, transcribe
with WhisperX, clean with Claude, and output a HuggingFace-compatible Whisper fine-tuning
dataset split 70/15/15 (700 train / 150 val / 150 test).

## Quick start
```bash
npm install
pip install -r requirements.txt
cp .env.example .env        # add ANTHROPIC_API_KEY
node orchestrator.js        # all 4 phases, auto-resumes on failure
```

## Run a single phase
```bash
node orchestrator.js download      # yt-dlp + ffmpeg audio extraction
node orchestrator.js transcribe    # WhisperX batch GPU transcription
node orchestrator.js cleanup       # Claude ASR correction
node orchestrator.js package       # write HuggingFace audiofolder splits
```

## System dependencies (must be installed globally)
- yt-dlp:  pip install yt-dlp
- ffmpeg:  brew install ffmpeg  /  apt install ffmpeg
- CUDA GPU: strongly recommended — CPU fallback is ~100× slower

## Architecture
```
agents/1-discovery.js    reads YouMakeup NDJSON, samples 1000, seeds 70/15/15 split
agents/2-extractor.js    yt-dlp per video, ffmpeg per segment → 16kHz mono WAV
agents/3-transcriber.py  WhisperX + forced alignment, batch processes all WAVs
agents/4-cleanup.js      Claude claude-sonnet-4-6, makeup-domain ASR correction
orchestrator.js          phase coordinator, checkpoint-aware
```

## Checkpoint files (./checkpoints/)
```
segments.jsonl      all discovered segments with split label
downloaded.jsonl    successfully extracted WAV paths
transcribed.jsonl   raw WhisperX output per segment
cleaned.jsonl       Claude-corrected transcripts (final)
failed.jsonl        skipped entries with error reason
```

## Output
```
dataset/
  train/  metadata.jsonl + audio/*.wav   (~6,300 segments)
  val/    metadata.jsonl + audio/*.wav   (~1,350 segments)
  test/   metadata.jsonl + audio/*.wav   (~1,350 segments)
```

## Fine-tune Whisper after pipeline
```bash
python train.py
```
Model saved to ./model/final — loads with WhisperForConditionalGeneration.from_pretrained()

## Environment variables
OPENAIUse _API_KEY   required
WORKERS             parallel yt-dlp workers, default 2 (max 4)
WHISPER_MODEL       large-v3 | medium | small, default large-v3
DEVICE              cuda | cpu, default cuda
TARGET_VIDEOS       how many videos to sample, default 1000
SEED                random seed for reproducible split, default 42
DATA_DIR            path to YouMakeup JSON files, default ./data
