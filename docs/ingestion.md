# Data Ingestion Phase

This document explains the **Ingestion** process of the MakeUP AI speech-to-text pipeline, detailing how video steps are discovered, sampled, downloaded, and segmented.

---

## 1. Step Discovery (`agents/1-discovery.js`)

The pipeline ingest process starts by reading annotation files from the **YouMakeup** dataset. These annotations associate YouTube video IDs with specific makeup tutorial step sequences (e.g., step number, start/end timestamps, facial target area, and a description).

### Key Processes:

- **JSON Parsing:** Reads train, validation, and test source step records from the target directory (`DATA_DIR`).
- **Timestamp Normalization:** Converts time annotations (formatted as `HH:MM:SS`, `MM:SS`, or floating seconds) to numerical seconds.
- **Filtering:** Retains only valid steps with positive durations (`end_time > start_time`) and non-empty step descriptions.
- **Dataset Splitting & Sampling:**
  - Performs a seeded, deterministic shuffle of all valid videos (default seed `42`) to ensure reproducibility.
  - Samples a configurable target number of videos (default `1000`).
  - Assigns sampled videos to Train, Validation, or Test groups based on a **70% / 15% / 15%** split ratio.
- **Metadata Checkpointing:** Generates and saves the discovered step list to `checkpoints/segments.jsonl` to ensure phase restarts do not shift split assignments.

---

## 2. Audio Download and Extraction (`agents/2-extractor.js`)

Once target segments are discovered, the extractor handles audio downloading and segment trimming.

```
YouTube Video Page ──> [yt-dlp] ──> Raw WAV (16kHz mono) ──> [ffmpeg] ──> Segmented WAVs
```

### Process Specifications:

1. **Downloading Streams (`yt-dlp`):**
   - Uses `yt-dlp` to extract high-quality audio streams.
   - Restricts processing to one video at a time per worker to avoid YouTube IP throttling or blocking.
   - If a video has been deleted, set to private, or region-locked, the system flags the video steps in `checkpoints/failed.jsonl` as `unavailable` and skips them.
2. **Standardization & Downsampling (`ffmpeg`):**
   - Audio streams are converted, downsampled, and saved as standard `16kHz`, `mono`, `16-bit PCM` WAV files.
   - Raw audio tracks are cached under `temp/audio/raw/` to avoid redownloading if multiple steps are extracted from the same video.
3. **Clip Segmentation (`ffmpeg`):**
   - Trims individual step clips from the raw video audio using the start and end timestamps.
   - Saves clips to `temp/audio/segments/` with naming convention `{videoId}_step{stepNum}.wav`.
   - Appends successful records to `checkpoints/downloaded.jsonl`.
   - After all step segments for a video are successfully generated, the raw full-length WAV file is purged to conserve disk space.
