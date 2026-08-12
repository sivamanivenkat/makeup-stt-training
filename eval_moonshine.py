"""
Print Moonshine predictions vs. reference transcripts side by side, for
manually inspecting failure modes (truncation, repetition, hallucination)
that an aggregate WER number hides.

Defaults to the longest clips in the validation split first, since that's
where a generation max-length cap would bite hardest.

Usage:
    python eval_moonshine.py --checkpoint ./model_moonshine_base/checkpoint-364
    python eval_moonshine.py --checkpoint ./model_moonshine_base/checkpoint-79 --num-examples 10
"""

import argparse

import evaluate
import torch

from datasets import Audio, load_dataset

from transformers import AutoConfig, AutoProcessor, MoonshineForConditionalGeneration

try:
    from transformers import MoonshineStreamingForConditionalGeneration
except ImportError:
    MoonshineStreamingForConditionalGeneration = None

from train import normalize_text
from train_moonshine import align_tokenizer_special_tokens


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect Moonshine predictions vs. references on real examples."
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to a trained checkpoint directory, or a Hub model id.",
    )

    parser.add_argument(
        "--dataset",
        default="./dataset",
        help="Dataset directory in Hugging Face audiofolder format.",
    )

    parser.add_argument(
        "--split",
        default="validation",
        help="Dataset split to sample from.",
    )

    parser.add_argument(
        "--num-examples",
        type=int,
        default=10,
        help="Number of examples to print.",
    )

    parser.add_argument(
        "--sort",
        choices=["duration_desc", "duration_asc", "none"],
        default="duration_desc",
        help=(
            "duration_desc (default) samples the longest clips first — the "
            "case most likely to hit a generation max-length cap."
        ),
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Generation cap. Set high on purpose to rule out truncation.",
    )

    parser.add_argument(
        "--num-beams",
        type=int,
        default=4,
        help="Beam search width, matching training's eval config.",
    )

    parser.add_argument(
        "--streaming",
        action="store_true",
        help=(
            "Force the streaming model class. Needed for local checkpoint "
            "paths, since they don't carry 'streaming' in the path name."
        ),
    )

    args = parser.parse_args()

    is_streaming = args.streaming or "streaming" in args.checkpoint.lower()

    if is_streaming and MoonshineStreamingForConditionalGeneration is None:
        raise ImportError(
            "MoonshineStreamingForConditionalGeneration is not available in "
            "the installed transformers version."
        )

    model_class = (
        MoonshineStreamingForConditionalGeneration
        if is_streaming
        else MoonshineForConditionalGeneration
    )

    print(f"Loading processor and model from: {args.checkpoint}")

    processor = AutoProcessor.from_pretrained(args.checkpoint)
    model_config = AutoConfig.from_pretrained(args.checkpoint)
    align_tokenizer_special_tokens(processor.tokenizer, model_config)

    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    if model_config.pad_token_id is None:
        model_config.pad_token_id = processor.tokenizer.pad_token_id

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model_class.from_pretrained(
        args.checkpoint,
        config=model_config,
    ).to(device)

    if model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = model.config.pad_token_id

    model.eval()

    print(f"Loading dataset split '{args.split}' from: {args.dataset}")

    raw_dataset = load_dataset("audiofolder", data_dir=args.dataset)

    if (
        args.split == "validation"
        and args.split not in raw_dataset
        and "val" in raw_dataset
    ):
        dataset = raw_dataset["val"]
    else:
        dataset = raw_dataset[args.split]

    dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))

    durations = [len(row["array"]) / row["sampling_rate"] for row in dataset["audio"]]
    indices = list(range(len(dataset)))

    if args.sort == "duration_desc":
        indices.sort(key=lambda i: durations[i], reverse=True)
    elif args.sort == "duration_asc":
        indices.sort(key=lambda i: durations[i])

    indices = indices[: args.num_examples]

    wer_metric = evaluate.load("wer")

    print("=" * 100)
    print(
        f"{args.num_examples} examples, sorted by {args.sort}, "
        f"max_new_tokens={args.max_new_tokens}, num_beams={args.num_beams}"
    )
    print("=" * 100)

    for rank, index in enumerate(indices, start=1):
        example = dataset[index]
        audio = example["audio"]
        reference = example["transcription"]
        duration = durations[index]

        inputs = processor(
            audio["array"],
            sampling_rate=audio["sampling_rate"],
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
            )

        prediction = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0]

        generated_length = generated_ids.shape[-1]
        hit_cap = generated_length >= args.max_new_tokens

        example_wer = 100 * wer_metric.compute(
            predictions=[normalize_text(prediction)],
            references=[normalize_text(reference)],
        )

        print(
            f"[{rank}] duration={duration:.2f}s  wer={example_wer:.1f}%  "
            f"generated_tokens={generated_length}"
            f"{'  <-- HIT MAX_NEW_TOKENS CAP' if hit_cap else ''}"
        )
        print(f"    REF:  {reference}")
        print(f"    PRED: {prediction}")
        print("-" * 100)


if __name__ == "__main__":
    main()
