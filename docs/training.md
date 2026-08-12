# Model Training & Fine-Tuning Phase

This document explains the **Training** phase of the MakeUP AI pipeline, which fine-tunes Whisper models on the curated domain-specific dataset using the `train.py` script.

---

## 1. Fine-Tuning Script (`train.py`)

The training script utilizes Hugging Face's `Seq2SeqTrainer` to train a sequence-to-sequence model (by default, `openai/whisper-small`).

### Core Model & Tokenizer Components:
* **`WhisperForConditionalGeneration`**: Loads the pre-trained weights. Attention implementation is forced to use PyTorch's native Scaled Dot-Product Attention (`sdpa`) for improved speed and lower memory usage.
* **`WhisperProcessor`**: Handles audio spectrogram extraction and tokenization. 
* **`DataCollatorSpeechSeq2SeqWithPadding`**: Dynamically pads the input audio features and target text tokens per batch to minimize memory usage, replacing target padding tokens with `-100` so they are ignored by the cross-entropy loss function.

---

## 2. Text Normalization & Evaluation

To accurately evaluate performance, Word Error Rate (WER) is computed on normalized text strings.

### Normalization Logic (`normalize_text`):
* Text labels and model predictions are lowercased.
* Punctuation is stripped.
* Consecutive spaces are collapsed.
* *Why:* In short instruction clips, differences in commas, periods, or capitalization (e.g. `"Apply foundation"` vs `"apply foundation."`) would artificially inflate WER without representing real transcription errors.

---

## 3. Training & Hyperparameter Configuration

The model is configured with hyperparameters adjusted for fine-tuning stability and performance:

| Parameter | Configuration Value | Explanation |
| :--- | :--- | :--- |
| **Optimizer** | `AdamW` (`adamw_torch`) | Adam optimizer with weight decay. |
| **Learning Rate** | `5e-6` | Conservative learning rate to avoid catastrophic forgetting of base speech representations. |
| **Weight Decay** | `0.01` | Regularization to prevent model overfitting on training steps. |
| **Warmup Steps** | `500` | Linearly increases learning rate from zero to stable level over initial steps. |
| **Batch Size** | `8` per GPU | Local training batch size. |
| **Gradient Accumulation** | `2` | Multiplies effective batch size to `16` before applying gradients, stabilizing training. |
| **Mixed Precision** | `fp16` (automatic if CUDA is present) | Reduces model memory footprint and accelerates throughput. |
| **Metric for Best Model** | `wer` (Word Error Rate) | Model checkpoints are evaluated and selected based on validation set WER. |
| **Early Stopping** | Patience of `4` evaluation cycles | Training terminates early if validation WER fails to improve over 4 evaluation cycles. |

---

## 4. Training Stability & Google Drive Backup

Speech-to-text fine-tuning is resource-intensive and often run on cloud runtimes (like Google Colab) that are subject to abrupt disconnections. The training script implements several safety mechanisms:

### Memory Management:
* Invokes regular Python Garbage Collection (`gc.collect()`).
* Clears PyTorch CUDA memory cache (`torch.cuda.empty_cache()`) between steps.
* Configures CUDA allocation settings: `expandable_segments:True` to reduce fragmentation.

### Resilience and Backup (`DriveBackupCallback`):
* **Local Checkpointing:** Training checkpoints are saved to the local folder at regular intervals (every 500 steps).
* **Cloud Backup:** A custom `TrainerCallback` clones each checkpoint directory to a durable backup directory (e.g., a mounted Google Drive folder `/content/drive/MyDrive/checkpoints`) immediately after it is saved.
* **Auto-Resume:** On startup, the script searches the local directory for checkpoints. If none exist (due to a runtime restart), it automatically restores the latest checkpoint from the backup directory and resumes the exact training state (learning rate, optimizer, epoch) from that step.
* **Checkpoint Rotation:** Retains only the best checkpoint and the last 3 checkpoints in both local and backup directories to manage storage capacity.
