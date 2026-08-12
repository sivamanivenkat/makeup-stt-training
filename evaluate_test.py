"""
Run the fine-tuned model against the held-out test split and report WER.

Usage (Colab cell or script):
    python evaluate_test.py --model ./model_v3/final --dataset ./dataset
"""

import argparse
import re

import evaluate
import torch
from datasets import Audio, load_dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="./model_v3/final")
    parser.add_argument("--dataset", default="./dataset")
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model from {args.model} on {device}...")
    processor = WhisperProcessor.from_pretrained(
        args.model, language="English", task="transcribe"
    )
    model = WhisperForConditionalGeneration.from_pretrained(args.model).to(device)
    model.eval()

    print("Loading test split...")
    dataset = load_dataset("audiofolder", data_dir=args.dataset, split="test")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))

    wer_metric = evaluate.load("wer")

    predictions = []
    references = []

    for start in range(0, len(dataset), args.batch):
        batch = dataset[start : start + args.batch]

        audio_arrays = [a["array"] for a in batch["audio"]]

        inputs = processor(
            audio_arrays,
            sampling_rate=16_000,
            return_tensors="pt",
            padding=True,
        )
        input_features = inputs.input_features.to(device)

        with torch.no_grad():
            predicted_ids = model.generate(
                input_features,
                max_length=225,
            )

        batch_predictions = processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )

        predictions.extend(normalize_text(p) for p in batch_predictions)
        references.extend(normalize_text(t) for t in batch["transcription"])

        print(f"{min(start + args.batch, len(dataset))}/{len(dataset)}")

    wer = 100 * wer_metric.compute(predictions=predictions, references=references)
    print("=" * 60)
    print(f"Test WER: {wer:.2f}")
    print(f"Test samples: {len(dataset)}")
    print("=" * 60)

    print("\nSample predictions:")
    for i in range(min(3, len(predictions))):
        print(f"\n[{i}] REF:  {references[i][:150]}")
        print(f"[{i}] HYP:  {predictions[i][:150]}")


if __name__ == "__main__":
    main()
