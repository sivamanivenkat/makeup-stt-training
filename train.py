"""
Whisper fine-tuning on the MakeUP AI STT dataset.

Expects:
    dataset/ in Hugging Face audiofolder format.

Usage:
    python train.py \
        --model openai/whisper-small \
        --dataset ./dataset \
        --output ./model_v3 \
        --backup-dir /content/drive/MyDrive/makeup-stt/checkpoints
"""

import argparse
import gc
import os
import re
import shutil
from dataclasses import dataclass
from typing import Any, Dict, List, Union

# Set environment variables before importing Hugging Face libraries.
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True",
)

# Disable Hugging Face Xet because it was hanging while downloading.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")

# PyTorch 2.6 defaults torch.load to weights_only=True, which breaks
# Trainer._load_rng_state() unpickling the checkpoint's numpy RNG state.
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")


import evaluate
import numpy as np
import torch

from datasets import Audio, load_dataset

from transformers import (
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from transformers.trainer_utils import get_last_checkpoint

# ─────────────────────────────────────────────────────────────────────────────
# Text normalization
# ─────────────────────────────────────────────────────────────────────────────


def normalize_text(text: str) -> str:
    """
    Lowercase and remove punctuation before WER calculation.

    Without normalization, punctuation and capitalization can incorrectly
    increase WER, especially for short makeup instruction transcripts.
    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\balright\b", "all right", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────


def checkpoint_dirs(root: str) -> List[str]:
    """
    Return checkpoint directories sorted by checkpoint step.
    """
    if not os.path.isdir(root):
        return []

    checkpoints = []

    for name in os.listdir(root):
        if not name.startswith("checkpoint-"):
            continue

        step_text = name.split("-")[-1]

        if not step_text.isdigit():
            continue

        checkpoint_path = os.path.join(root, name)
        checkpoints.append((int(step_text), checkpoint_path))

    checkpoints.sort(key=lambda item: item[0])

    return [path for _, path in checkpoints]


def restore_checkpoint_from_backup(
    output_dir: str,
    backup_dir: str | None,
) -> str | None:
    """
    Find a local checkpoint.

    If no local checkpoint exists, restore the latest checkpoint from the
    backup directory into the local output directory.
    """
    os.makedirs(output_dir, exist_ok=True)

    local_checkpoint = get_last_checkpoint(output_dir)

    if local_checkpoint:
        print(f"Found local checkpoint: {local_checkpoint}")
        return local_checkpoint

    if not backup_dir:
        print("No backup directory provided.")
        return None

    if not os.path.isdir(backup_dir):
        print(f"Backup directory does not exist: {backup_dir}")
        return None

    backup_checkpoint = get_last_checkpoint(backup_dir)

    if backup_checkpoint is None:
        print(f"No checkpoint found in backup directory: {backup_dir}")
        return None

    restored_checkpoint = os.path.join(
        output_dir,
        os.path.basename(backup_checkpoint),
    )

    print(
        f"No local checkpoint — restoring "
        f"{backup_checkpoint} -> {restored_checkpoint}"
    )

    shutil.copytree(
        backup_checkpoint,
        restored_checkpoint,
        dirs_exist_ok=True,
    )

    print(f"Checkpoint restored successfully: {restored_checkpoint}")

    return restored_checkpoint


class DriveBackupCallback(TrainerCallback):
    """
    Copy each newly saved local checkpoint to durable storage.

    This protects training progress if the Colab runtime disconnects.
    """

    def __init__(
        self,
        backup_dir: str,
        save_total_limit: int,
    ):
        self.backup_dir = backup_dir
        self.save_total_limit = save_total_limit

        os.makedirs(self.backup_dir, exist_ok=True)

    def on_save(
        self,
        args,
        state,
        control,
        **kwargs,
    ):
        checkpoint_name = f"checkpoint-{state.global_step}"

        source_checkpoint = os.path.join(
            args.output_dir,
            checkpoint_name,
        )

        destination_checkpoint = os.path.join(
            self.backup_dir,
            checkpoint_name,
        )

        if os.path.isdir(source_checkpoint):
            print(f"Backing up {checkpoint_name} " f"to {destination_checkpoint}")

            shutil.copytree(
                source_checkpoint,
                destination_checkpoint,
                dirs_exist_ok=True,
            )

            print(f"Backed up {checkpoint_name} " f"-> {destination_checkpoint}")

        # Never delete the best checkpoint — load_best_model_at_end needs it
        # on disk at the end of training, and it may not be one of the most
        # recent N checkpoints.
        protected_name = None

        if state.best_model_checkpoint:
            protected_name = os.path.basename(state.best_model_checkpoint)

        backup_checkpoints = [
            path
            for path in checkpoint_dirs(self.backup_dir)
            if os.path.basename(path) != protected_name
        ]

        number_to_delete = max(
            0,
            len(backup_checkpoints) - self.save_total_limit,
        )

        for stale_checkpoint in backup_checkpoints[:number_to_delete]:
            print(f"Removing old backup checkpoint: {stale_checkpoint}")

            shutil.rmtree(
                stale_checkpoint,
                ignore_errors=True,
            )

        return control


# ─────────────────────────────────────────────────────────────────────────────
# Data collator
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(
        self,
        features: List[
            Dict[
                str,
                Union[List[int], torch.Tensor],
            ]
        ],
    ) -> Dict[str, torch.Tensor]:

        input_features = [
            {
                "input_features": feature["input_features"],
            }
            for feature in features
        ]

        batch = self.processor.feature_extractor.pad(
            input_features,
            return_tensors="pt",
        )

        label_features = [
            {
                "input_ids": feature["labels"],
            }
            for feature in features
        ]

        labels_batch = self.processor.tokenizer.pad(
            label_features,
            return_tensors="pt",
        )

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1),
            -100,
        )

        if labels.shape[1] > 0:
            decoder_start_tokens = labels[:, 0] == self.decoder_start_token_id

            if decoder_start_tokens.all().cpu().item():
                labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


# ─────────────────────────────────────────────────────────────────────────────
# Global processor used by dataset.map
# ─────────────────────────────────────────────────────────────────────────────

processor = None


def prepare(batch: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert one audio/transcription example into Whisper input features
    and tokenized labels.
    """
    if processor is None:
        raise RuntimeError("Processor has not been initialized.")

    audio = batch["audio"]

    batch["input_features"] = processor.feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
    ).input_features[0]

    batch["labels"] = processor.tokenizer(
        batch["transcription"],
        truncation=True,
        max_length=448,
    ).input_ids

    return batch


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    global processor

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    parser = argparse.ArgumentParser(
        description="Fine-tune Whisper on the MakeUP AI STT dataset."
    )

    parser.add_argument(
        "--model",
        default="openai/whisper-small",
        help="Base Whisper model or local model directory.",
    )

    parser.add_argument(
        "--dataset",
        default="./dataset",
        help="Dataset directory in Hugging Face audiofolder format.",
    )

    parser.add_argument(
        "--output",
        default="./model",
        help="Local output directory.",
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=4000,
        help="Total number of training steps.",
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=8,
        help="Training batch size per GPU.",
    )

    parser.add_argument(
        "--grad-accum",
        type=int,
        default=2,
        help="Gradient accumulation steps.",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=5e-6,
        help="Learning rate.",
    )

    parser.add_argument(
        "--hub-id",
        default=None,
        help="Optional Hugging Face Hub repository ID.",
    )

    parser.add_argument(
        "--backup-dir",
        default=None,
        help=(
            "Durable checkpoint directory, such as Google Drive. "
            "Checkpoints are restored from this directory when the "
            "local output directory is empty."
        ),
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Whisper fine-tuning")
    print("=" * 70)
    print(f"Model:        {args.model}")
    print(f"Dataset:      {args.dataset}")
    print(f"Output:       {args.output}")
    print(f"Backup:       {args.backup_dir}")
    print(f"Max steps:    {args.steps}")
    print(f"Batch size:   {args.batch}")
    print(f"Grad accum:   {args.grad_accum}")
    print(f"Learning rate:{args.lr}")
    print(f"CUDA:         {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU:          {torch.cuda.get_device_name(0)}")

    print("=" * 70)

    if not os.path.isdir(args.dataset):
        raise FileNotFoundError(f"Dataset directory was not found: {args.dataset}")

    # Restore checkpoint before loading the processor and model.
    last_checkpoint = restore_checkpoint_from_backup(
        output_dir=args.output,
        backup_dir=args.backup_dir,
    )

    # Use the checkpoint as the model and processor source when available.
    # This prevents downloading openai/whisper-small again.
    model_source = last_checkpoint if last_checkpoint is not None else args.model

    print(f"Loading processor from: {model_source}")

    processor = WhisperProcessor.from_pretrained(
        model_source,
        language="English",
        task="transcribe",
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Load dataset
    # ─────────────────────────────────────────────────────────────────────────

    print("Loading dataset...")

    raw_dataset = load_dataset(
        "audiofolder",
        data_dir=args.dataset,
    )

    if "val" in raw_dataset.keys() and "validation" not in raw_dataset.keys():
        raw_dataset["validation"] = raw_dataset.pop("val")

    required_splits = {
        "train",
        "validation",
        "test",
    }

    missing_splits = required_splits.difference(raw_dataset.keys())

    if missing_splits:
        raise ValueError(
            "Dataset is missing required splits: " + ", ".join(sorted(missing_splits))
        )

    raw_dataset = raw_dataset.cast_column(
        "audio",
        Audio(sampling_rate=16_000),
    )

    train_columns = raw_dataset["train"].column_names

    if "audio" not in train_columns:
        raise ValueError("Dataset must contain an 'audio' column.")

    if "transcription" not in train_columns:
        raise ValueError("Dataset must contain a 'transcription' column.")

    columns_to_remove = [
        column for column in train_columns if column not in ("input_features", "labels")
    ]

    print("Preparing dataset features...")

    dataset = raw_dataset.map(
        prepare,
        remove_columns=columns_to_remove,
        num_proc=4,
        desc="Preparing Whisper features",
    )

    print(
        f"Train: {len(dataset['train'])} | "
        f"Val: {len(dataset['validation'])} | "
        f"Test: {len(dataset['test'])}"
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Load model
    # ─────────────────────────────────────────────────────────────────────────

    print(f"Loading model weights from: {model_source}")

    model = WhisperForConditionalGeneration.from_pretrained(
        model_source,
        attn_implementation="sdpa",
    )

    model.generation_config.language = "english"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Metrics
    # ─────────────────────────────────────────────────────────────────────────

    print("Loading WER metric...")

    wer_metric = evaluate.load("wer")

    def compute_metrics(prediction) -> Dict[str, float]:
        prediction_ids = prediction.predictions

        if isinstance(prediction_ids, tuple):
            prediction_ids = prediction_ids[0]

        label_ids = np.copy(prediction.label_ids)

        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        prediction_strings = processor.tokenizer.batch_decode(
            prediction_ids,
            skip_special_tokens=True,
        )

        label_strings = processor.tokenizer.batch_decode(
            label_ids,
            skip_special_tokens=True,
        )

        prediction_strings = [normalize_text(text) for text in prediction_strings]

        label_strings = [normalize_text(text) for text in label_strings]

        wer = 100 * wer_metric.compute(
            predictions=prediction_strings,
            references=label_strings,
        )

        return {
            "wer": round(wer, 2),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Training arguments
    # ─────────────────────────────────────────────────────────────────────────

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_steps=500,
        max_steps=args.steps,
        gradient_checkpointing=False,
        fp16=torch.cuda.is_available(),
        eval_strategy="steps",
        per_device_eval_batch_size=max(
            args.batch // 2,
            1,
        ),
        predict_with_generate=True,
        generation_max_length=225,
        generation_num_beams=4,
        save_steps=500,
        eval_steps=500,
        logging_steps=50,
        save_total_limit=3,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=args.hub_id is not None,
        hub_model_id=args.hub_id,
        remove_unused_columns=False,
    )

    callbacks = [
        EarlyStoppingCallback(
            early_stopping_patience=4,
        ),
    ]

    if args.backup_dir:
        callbacks.append(
            DriveBackupCallback(
                backup_dir=args.backup_dir,
                save_total_limit=training_args.save_total_limit,
            )
        )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor,
        callbacks=callbacks,
    )

    if last_checkpoint:
        print(f"Resuming full Trainer state from: " f"{last_checkpoint}")
    else:
        print("No checkpoint found. Starting from base model weights.")

    print("Starting training...")

    # Do not silently restart from step zero if checkpoint resume fails.
    trainer.train(
        resume_from_checkpoint=last_checkpoint,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Save final model
    # ─────────────────────────────────────────────────────────────────────────

    final_directory = os.path.join(
        args.output,
        "final",
    )

    os.makedirs(
        final_directory,
        exist_ok=True,
    )

    trainer.save_model(final_directory)
    processor.save_pretrained(final_directory)

    print("=" * 70)
    print(f"Model saved to: {final_directory}")
    print(
        "Load with:\n"
        "WhisperForConditionalGeneration.from_pretrained("
        f"'{final_directory}')"
    )
    print("=" * 70)

    if args.hub_id:
        trainer.push_to_hub()
        print(f"Pushed to Hugging Face Hub: {args.hub_id}")


if __name__ == "__main__":
    main()
