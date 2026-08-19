"""
Moonshine fine-tuning on the MakeUP AI STT dataset.

Moonshine has no fixed 30s input window like Whisper, so inference latency
scales with actual clip length — this is the real-time-friendly alternative
to the Whisper model trained by train.py.

Supports both plain Moonshine checkpoints (e.g. UsefulSensors/moonshine-base)
and the streaming variants (UsefulSensors/moonshine-streaming-{tiny,small,
medium}) — the model class is picked automatically from the checkpoint name.
The streaming variants add sliding-window encoder attention for low-latency
incremental transcription; fine-tuning them uses the same input_values/labels
interface as plain Moonshine, so no other code paths differ.

Adds duration-based curriculum learning (train on short clips first, then
progressively longer ones) and the schedule-free AdamW optimizer.

Expects:
    dataset/ in Hugging Face audiofolder format (see train.py / orchestrator.js).

Requires (on top of requirements.txt):
    pip install -r requirements-moonshine.txt

    MoonshineStreamingForConditionalGeneration is very recent. If it fails to
    import, upgrade transformers:
        pip install -U transformers
        # or, if not yet in a release:
        pip install -U git+https://github.com/huggingface/transformers.git

Usage:
    python train_moonshine.py \
        --model UsefulSensors/moonshine-streaming-medium \
        --dataset ./dataset \
        --output ./model_moonshine \
        --backup-dir /content/drive/MyDrive/makeup-stt/checkpoints-moonshine
"""

import argparse
import gc
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Union

# Set environment variables before importing Hugging Face libraries.
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True",
)

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")


import evaluate
import numpy as np
import torch

from datasets import Audio, load_dataset

from transformers import (
    AutoConfig,
    AutoProcessor,
    EarlyStoppingCallback,
    MoonshineForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

try:
    from transformers import MoonshineStreamingForConditionalGeneration
except ImportError:
    MoonshineStreamingForConditionalGeneration = None

from transformers.trainer_utils import get_last_checkpoint

from train import (
    DriveBackupCallback,
    normalize_text,
    restore_checkpoint_from_backup,
)


def align_tokenizer_special_tokens(tokenizer: Any, model_config: Any) -> None:
    """Align tokenizer special tokens with IDs already used by the model.

    Some Moonshine checkpoints declare the special-token IDs in config.json,
    but omit them from the tokenizer metadata. Reusing the tokens at those IDs
    preserves the pretrained vocabulary and avoids resizing model embeddings.
    """
    changed_tokens = []

    for token_name in ("bos_token", "eos_token", "pad_token"):
        token_id_name = f"{token_name}_id"
        model_token_id = getattr(model_config, token_id_name, None)
        tokenizer_token_id = getattr(tokenizer, token_id_name, None)

        if model_token_id is None:
            if tokenizer_token_id is None:
                raise ValueError(
                    f"Neither the model nor tokenizer defines {token_id_name}."
                )
            continue

        if tokenizer_token_id == model_token_id:
            continue

        token = tokenizer.convert_ids_to_tokens(model_token_id)

        if token is None:
            raise ValueError(
                f"Model {token_id_name}={model_token_id} does not map to a "
                "token in the tokenizer vocabulary."
            )

        setattr(tokenizer, token_name, token)

        if getattr(tokenizer, token_id_name, None) != model_token_id:
            raise ValueError(
                f"Could not align tokenizer {token_id_name} with model value "
                f"{model_token_id}."
            )

        changed_tokens.append(f"{token_name}={token!r} (id={model_token_id})")

    if changed_tokens:
        print("Aligned tokenizer special tokens: " + ", ".join(changed_tokens))


# ─────────────────────────────────────────────────────────────────────────────
# Data collator
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class DataCollatorMoonshineSeq2SeqWithPadding:
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

        input_values = [
            {
                "input_values": feature["input_values"],
            }
            for feature in features
        ]

        batch = self.processor.feature_extractor.pad(
            input_values,
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
    Convert one audio/transcription example into Moonshine input values,
    tokenized labels, and clip duration (used for curriculum filtering).
    """
    if processor is None:
        raise RuntimeError("Processor has not been initialized.")

    audio = batch["audio"]

    batch["input_values"] = processor.feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
    ).input_values[0]

    batch["labels"] = processor.tokenizer(
        batch["transcription"],
        truncation=True,
        max_length=448,
    ).input_ids

    batch["duration"] = len(audio["array"]) / audio["sampling_rate"]

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
        description="Fine-tune Moonshine on the MakeUP AI STT dataset."
    )

    parser.add_argument(
        "--model",
        default="UsefulSensors/moonshine-streaming-medium",
        help="Base Moonshine model or local model directory.",
    )

    parser.add_argument(
        "--dataset",
        default="./dataset",
        help="Dataset directory in Hugging Face audiofolder format.",
    )

    parser.add_argument(
        "--output",
        default="./model_moonshine",
        help="Local output directory.",
    )

    parser.add_argument(
        "--epochs",
        type=float,
        default=3.0,
        help="Total training epochs, split evenly across curriculum stages.",
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
        default=5e-5,
        help="Learning rate. Schedule-free AdamW does not decay this.",
    )

    parser.add_argument(
        "--curriculum-stages",
        default="5,10,15",
        help=(
            "Comma-separated max clip duration (seconds) per curriculum "
            "stage. Each stage trains on all clips at or below its cutoff, "
            "widening the pool as training progresses."
        ),
    )

    parser.add_argument(
        "--no-curriculum",
        action="store_true",
        help="Disable curriculum learning and train on the full dataset directly.",
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

    stage_cutoffs = (
        [None]
        if args.no_curriculum
        else [float(value) for value in args.curriculum_stages.split(",")]
    )

    print("=" * 70)
    print("Moonshine fine-tuning")
    print("=" * 70)
    print(f"Model:            {args.model}")
    print(f"Dataset:          {args.dataset}")
    print(f"Output:           {args.output}")
    print(f"Backup:           {args.backup_dir}")
    print(f"Epochs (total):   {args.epochs}")
    print(f"Batch size:       {args.batch}")
    print(f"Grad accum:       {args.grad_accum}")
    print(f"Learning rate:    {args.lr}")
    print(f"Curriculum:       {stage_cutoffs}")
    print(f"CUDA:             {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU:              {torch.cuda.get_device_name(0)}")

    print("=" * 70)

    if not os.path.isdir(args.dataset):
        raise FileNotFoundError(f"Dataset directory was not found: {args.dataset}")

    # Restore checkpoint before loading the processor and model.
    last_checkpoint = restore_checkpoint_from_backup(
        output_dir=args.output,
        backup_dir=args.backup_dir,
    )

    model_source = last_checkpoint if last_checkpoint is not None else args.model

    # Local checkpoint dirs don't carry "streaming" in their path, so decide
    # the model class from the original --model argument, not model_source.
    is_streaming = "streaming" in args.model.lower()

    if is_streaming and MoonshineStreamingForConditionalGeneration is None:
        raise ImportError(
            "MoonshineStreamingForConditionalGeneration is not available in "
            "the installed transformers version. Upgrade with:\n"
            "  pip install -U transformers\n"
            "or, if not yet in a release:\n"
            "  pip install -U git+https://github.com/huggingface/transformers.git"
        )

    model_class = (
        MoonshineStreamingForConditionalGeneration
        if is_streaming
        else MoonshineForConditionalGeneration
    )

    print(f"Loading processor from: {model_source}")

    processor = AutoProcessor.from_pretrained(model_source)
    model_config = AutoConfig.from_pretrained(model_source)
    align_tokenizer_special_tokens(processor.tokenizer, model_config)

    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
        print(
            f"Tokenizer had no pad_token — reusing eos_token as pad "
            f"({processor.tokenizer.eos_token!r}, "
            f"id={processor.tokenizer.eos_token_id})"
        )

    if model_config.pad_token_id is None:
        model_config.pad_token_id = processor.tokenizer.pad_token_id
        print(
            "Model config had no pad_token_id — set to "
            f"{model_config.pad_token_id} to match tokenizer."
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
        column
        for column in train_columns
        if column not in ("input_values", "labels", "duration")
    ]

    print("Preparing dataset features...")

    dataset = raw_dataset.map(
        prepare,
        remove_columns=columns_to_remove,
        num_proc=4,
        desc="Preparing Moonshine features",
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

    model = model_class.from_pretrained(
        model_source,
        config=model_config,
        attn_implementation="sdpa",
    )

    if model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = model.config.pad_token_id

    # Blocks exact n-gram repetition during beam search. Without this, hard
    # or long inputs can make the decoder loop on a repeated phrase instead
    # of emitting EOS, running out to max_length and wrecking that example's
    # WER badly enough to skew the whole eval-set metric.
    model.generation_config.no_repeat_ngram_size = 3

    # model.generation_config.max_length ships with a pretrained default
    # (194 for moonshine-base) that's independent of Seq2SeqTrainingArguments'
    # generation_max_length below. If left unset, eval-time generation silently
    # truncates at 194 regardless of what generation_max_length says, which
    # inflates eval_wer (and corrupts load_best_model_at_end / early stopping,
    # both keyed on that metric) without raising an error — just a "friendly
    # reminder" warning easy to miss in a long log.
    generation_max_length = 225
    model.generation_config.max_length = generation_max_length

    data_collator = DataCollatorMoonshineSeq2SeqWithPadding(
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

        prediction_ids = np.copy(prediction_ids)

        # Variable-length generated sequences across eval batches get padded
        # with -100 when concatenated. The tokenizer's Rust decoder can't
        # cast -100 to its internal unsigned int type and raises an
        # OverflowError, so it needs the same cleanup as label_ids below.
        prediction_ids[prediction_ids == -100] = processor.tokenizer.pad_token_id

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
    # Curriculum stages — train sequentially on widening duration cutoffs.
    # Each stage resumes from whatever checkpoint the previous stage (or a
    # prior run) left behind, so the whole run is restart-safe.
    # ─────────────────────────────────────────────────────────────────────────

    epochs_per_stage = max(
        args.epochs / len(stage_cutoffs),
        0.1,
    )

    eval_dataset = dataset["validation"]
    train_durations = np.array(dataset["train"]["duration"])

    for stage_index, cutoff in enumerate(stage_cutoffs, start=1):
        # Cheap pre-check: if a checkpoint already exists and its global_step
        # already meets or exceeds what this stage would train to, skip the
        # (slow) filter and the no-op train() call entirely. Without this,
        # resuming after all stages already completed re-filters the full
        # dataset for every earlier stage before Trainer notices there's
        # nothing left to do for it.
        resume_checkpoint = get_last_checkpoint(args.output)

        if resume_checkpoint:
            state_path = os.path.join(resume_checkpoint, "trainer_state.json")

            with open(state_path, "r", encoding="utf-8") as state_file:
                resumed_global_step = json.load(state_file)["global_step"]

            stage_example_count = (
                len(dataset["train"])
                if cutoff is None
                else int((train_durations <= cutoff).sum())
            )

            dataloader_len = math.ceil(stage_example_count / args.batch)
            steps_per_epoch = max(dataloader_len // args.grad_accum, 1)
            target_steps = math.ceil(steps_per_epoch * epochs_per_stage)

            if resumed_global_step >= target_steps:
                print(
                    f"Skipping stage {stage_index}/{len(stage_cutoffs)} "
                    f"(<= {cutoff}s clips): checkpoint already at step "
                    f"{resumed_global_step}, stage target is {target_steps}."
                )
                continue

        if cutoff is None:
            stage_train_dataset = dataset["train"]
            stage_label = "full dataset (no curriculum)"
        else:
            stage_train_dataset = dataset["train"].filter(
                lambda duration: duration <= cutoff,
                input_columns=["duration"],
                desc=f"Filtering stage {stage_index} (<= {cutoff}s)",
            )
            stage_label = f"<= {cutoff}s clips"

        print("=" * 70)
        print(
            f"Curriculum stage {stage_index}/{len(stage_cutoffs)}: {stage_label} "
            f"({len(stage_train_dataset)} clips, {epochs_per_stage:.2f} epochs)"
        )
        print("=" * 70)

        training_args = Seq2SeqTrainingArguments(
            output_dir=args.output,
            per_device_train_batch_size=args.batch,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            optim="schedule_free_adamw",
            lr_scheduler_type="constant",
            warmup_steps=500,
            num_train_epochs=epochs_per_stage,
            gradient_checkpointing=False,
            fp16=torch.cuda.is_available(),
            eval_strategy="steps",
            per_device_eval_batch_size=max(
                args.batch // 2,
                1,
            ),
            predict_with_generate=True,
            generation_max_length=generation_max_length,
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
            train_dataset=stage_train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            processing_class=processor,
            callbacks=callbacks,
        )

        stage_checkpoint = get_last_checkpoint(args.output)

        if stage_checkpoint:
            print(f"Resuming stage {stage_index} from: {stage_checkpoint}")
        else:
            print(f"Starting stage {stage_index} from current model weights.")

        trainer.train(
            resume_from_checkpoint=stage_checkpoint,
        )

        model = trainer.model

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
        f"{model_class.__name__}.from_pretrained("
        f"'{final_directory}')"
    )
    print("=" * 70)

    if args.hub_id:
        trainer.push_to_hub()
        print(f"Pushed to Hugging Face Hub: {args.hub_id}")


if __name__ == "__main__":
    main()
