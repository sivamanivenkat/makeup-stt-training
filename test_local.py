"""
Quick local sanity check: load the fine-tuned model and transcribe one
audio file from the test split.

Usage:
    python test_local.py
"""

from transformers import WhisperForConditionalGeneration, WhisperProcessor
import torch
import librosa

MODEL_DIR = "./model/final"
AUDIO_FILE = "./dataset/test/audio/-RXs_SSl8QA_step1.wav"

print("Loading model...")
model = WhisperForConditionalGeneration.from_pretrained(MODEL_DIR)
processor = WhisperProcessor.from_pretrained(
    MODEL_DIR, language="English", task="transcribe"
)
model.eval()

print(f"Loading audio: {AUDIO_FILE}")
audio, sr = librosa.load(AUDIO_FILE, sr=16000)

inputs = processor(audio, sampling_rate=16000, return_tensors="pt").input_features

print("Transcribing...")
with torch.no_grad():
    predicted_ids = model.generate(inputs, max_length=225)

transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

print("=" * 60)
print("TRANSCRIPTION:")
print(transcription)
print("=" * 60)
