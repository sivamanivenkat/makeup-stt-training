# Data Curation & Refinement Phase

This document explains the **Curation** process of the MakeUP AI pipeline, covering transcription, forced alignment, domain-specific text cleanup, and Hugging Face dataset packaging.

---

## 1. Batch Transcription & Alignment (`agents/3-transcriber.py`)

Using the extracted segment WAV files, the pipeline performs transcription and word-level temporal alignment using **WhisperX**.

### Processes:
* **Batch Processing:** Reads the manifest of successfully trimmed WAV files from `checkpoints/downloaded.jsonl`.
* **Transcription:** Loads a Whisper model (default `large-v3` running on `cuda`) and processes the audio using batching to maximize GPU usage.
* **Forced Alignment:** Applies WhisperX's phoneme-based alignment model to calculate precise start and end times for individual words.
* **Checkpoint Saving:** Outputs transcriptions and word alignment lists to `checkpoints/transcribed.jsonl`.

---

## 2. Domain-Specific Text Cleanup (`agents/4-cleanup.js`)

Raw automatic speech recognition (ASR) transcripts often contain phonetic transcription mistakes when encountering specialized vocabulary, brands, or product categories. The curation phase corrects these using an LLM.

### AI Correction Prompt:
Uses OpenAI's API (default: `gpt-4o-mini`) configured with a specialized system prompt. It is supplied with the step description from the original metadata and the target facial area to provide contextual hints.

### Target Areas of Correction:
1. **Product Names:** e.g., foundation, concealer, highlighter, setting spray, BB cream.
2. **Tool Names:** e.g., kabuki brush, beauty blender, blending brush, spoolie.
3. **Application Techniques:** e.g., strobing, cut crease, halo eye, baking, stippling, buffing.
4. **Facial Areas:** e.g., waterline, lid, cupid's bow, inner corner, T-zone.
5. **Brand Names:** e.g., MAC, Fenty, NARS, Charlotte Tilbury, Too Faced, ABH.

### Post-Processing Rules:
* **Structural Preservation:** The clean-up is strictly instructed to correct spelling, spacing, and brand capitalization (e.g., `"eye liner"` &rarr; `"eyeliner"`, `"under eye"` &rarr; `"under-eye"`) but **never** rephrase, summarize, or edit the sentence meaning.
* Checkpoints are written to `checkpoints/cleaned.jsonl`.

---

## 3. Dataset Packaging & Split Structuring

Once cleanup is complete, the final phase packages the data to be consumed by standard machine learning toolkits.

```
dataset/
├── train/
│   ├── metadata.jsonl
│   └── audio/
│       └── [segment_id].wav
├── val/
│   ├── metadata.jsonl
│   └── audio/
│       └── [segment_id].wav
└── test/
    ├── metadata.jsonl
    └── audio/
        └── [segment_id].wav
```

### Process Specifications:
1. **Directory Routing:** Splits the cleaned transcripts back into their assigned divisions (Train, Val, Test).
2. **Audio Transfer:** Copies the corresponding WAV files from `temp/audio/segments/` to the split-specific audio folders (`dataset/{split}/audio/`).
3. **Metadata Indexing:** Writes `metadata.jsonl` files for each split. Each line contains a JSON object mapping the audio file name to its target text label:
   ```json
   {"file_name": "audio/video_step1.wav", "transcription": "Apply a liquid foundation using a damp beauty blender."}
   ```

---

## 4. Integrity Validation (`validate.js`)

A validation script is run prior to training (`npm run validate`) to ensure data consistency:
* Confirms `metadata.jsonl` files are populated and formatted.
* Checks that every WAV file referenced in `metadata.jsonl` physically exists on disk.
* Verifies no empty or whitespace-only transcriptions exist.
* Summarizes dataset statistics and reports any permanent failures.
