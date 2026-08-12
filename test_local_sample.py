"""
Local sanity check: transcribe 10 random test-split files and compare
against the reference transcription.

Usage:
    python test_local_sample.py
"""

import json
import random
import re

import librosa
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

MODEL_DIR = "./model/final"
TEST_DIR = "./dataset/test"
SEED = 7
N = 10


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


print("Loading model...")
model = WhisperForConditionalGeneration.from_pretrained(MODEL_DIR)
processor = WhisperProcessor.from_pretrained(
    MODEL_DIR, language="English", task="transcribe"
)
model.eval()

rows = []
with open(f"{TEST_DIR}/metadata.jsonl", encoding="utf-8") as f:
    for line in f:
        rows.append(json.loads(line))

random.seed(SEED)
sample = random.sample(rows, N)

for i, row in enumerate(sample):
    wav_path = f"{TEST_DIR}/{row['file_name']}"
    reference = row["transcription"]

    audio, sr = librosa.load(wav_path, sr=16000)
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt").input_features

    with torch.no_grad():
        predicted_ids = model.generate(inputs, max_length=225)

    hypothesis = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    print(f"\n[{i+1}/{N}] {row['file_name']}")
    print(f"REF: {normalize_text(reference)[:200]}")
    print(f"HYP: {normalize_text(hypothesis)[:200]}")

print("\nDone.")
