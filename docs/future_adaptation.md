# Pipeline Porting & Future Model Adaptation Guide

This document lists the critical components, parameters, and code paths you must modify when adapting this pipeline to train other speech-to-text models (e.g., different domains, languages, or model architectures).

---

## 1. Domain Adaptations (Changing the Subject Matter)

If you are training on a domain other than makeup (e.g., medical, cooking, automotive, coding tutorials):

### Modify the LLM Cleanup Dictionary
In [4-cleanup.js](file:///d:/UserFolders/Downloads/makeup-stt-pipeline/makeup-stt-pipeline/agents/4-cleanup.js#L13-L33):
* Update the `SYSTEM_PROMPT` to replace the makeup domain vocabulary (products, tools, techniques, brands) with your target domain's vocabulary.
* Update `userPrompt` formatting in the `cleanOne` function if your source data uses metadata keys other than `caption` or `area`.

---

## 2. Language Adaptations (Training Non-English Models)

The current pipeline is optimized exclusively for English. For other languages:

### Transcription and Phoneme Alignment
In [3-transcriber.py](file:///d:/UserFolders/Downloads/makeup-stt-pipeline/makeup-stt-pipeline/agents/3-transcriber.py#L38):
* Modify `model.transcribe(..., language="en")` to target your language code (e.g., `"es"`, `"fr"`).
* Modify the WhisperX forced-alignment call in [3-transcriber.py](file:///d:/UserFolders/Downloads/makeup-stt-pipeline/makeup-stt-pipeline/agents/3-transcriber.py#L94) to load the correct alignment model:
  ```python
  align_model, align_metadata = whisperx.load_align_model(language_code="your_lang_code", device=args.device)
  ```

### Whisper Configuration
In [train.py](file:///d:/UserFolders/Downloads/makeup-stt-pipeline/makeup-stt-pipeline/train.py#L249-L253) and [train.py](file:///d:/UserFolders/Downloads/makeup-stt-pipeline/makeup-stt-pipeline/train.py#L318-L321):
* Modify the language specifications for the processor and model's generation config:
  ```python
  processor = WhisperProcessor.from_pretrained(..., language="Spanish", task="transcribe")
  model.generation_config.language = "spanish"
  ```

---

## 3. Dataset Format Adaptations (Non-YouTube or Non-YouMakeup Sources)

### JSON Structure
In [1-discovery.js](file:///d:/UserFolders/Downloads/makeup-stt-pipeline/makeup-stt-pipeline/agents/1-discovery.js#L29-L50):
* The parser expects metadata structure formatted as `videoId`, `stepNum`, `startime`/`endtime`, `area`, and `caption`. 
* If you ingest standard datasets (e.g., LibriSpeech, CommonVoice, or custom annotation CSVs), rewrite the parsing loop inside `parseVideoEntry` to extract these fields from your source format.

### Audio Extraction Sources
In [2-extractor.js](file:///d:/UserFolders/Downloads/makeup-stt-pipeline/makeup-stt-pipeline/agents/2-extractor.js#L26-L61):
* The current extractor assumes video segments are hosted on YouTube and uses `yt-dlp` to download them.
* If you have local video/audio directories, bypass `downloadVideo` and point the `trimSegment` function directly to your local file paths.

---

## 4. Architecture Adaptations (Non-Whisper Models)

If you decide to train a model architecture other than Whisper (e.g., **Wav2Vec2**, **Hubert**, or **MMS**):

### Incompatibility with `train.py`
The [train.py](file:///d:/UserFolders/Downloads/makeup-stt-pipeline/makeup-stt-pipeline/train.py) file is explicitly written for Seq2Seq (Encoder-Decoder) training. Connectionist Temporal Classification (CTC) models (like Wav2Vec2) do not use decoders.

If you port the training script to a CTC model, you must:
1. Replace `Seq2SeqTrainer` and `Seq2SeqTrainingArguments` with Hugging Face's standard `Trainer` and `TrainingArguments`.
2. Swap the `WhisperProcessor` and tokenizer for the respective CTC model processor.
3. Rewrite the data collator to handle 1D padding and not include decoder attention masks.
4. Alter the compute metrics function to use simple decoder token search instead of `predict_with_generate`.

---

## 5. Hyperparameter & Scaling Limits

### GPU VRAM Restrictions
* **Whisper Model Sizes:** Fine-tuning larger Whisper checkpoints (e.g., `whisper-medium` or `whisper-large-v3`) requires significantly more VRAM.
* **VRAM Mitigations:** If you run into CUDA Out-Of-Memory (OOM) errors when training larger models, update [train.py](file:///d:/UserFolders/Downloads/makeup-stt-pipeline/makeup-stt-pipeline/train.py#L380-L400) to:
  * Reduce `--batch` to `4` or `2`.
  * Increase `--grad-accum` to `4` or `8` (maintaining an effective batch size of 16 or 32).
  * Set `gradient_checkpointing=True` in `Seq2SeqTrainingArguments` to trade computation steps for VRAM.
