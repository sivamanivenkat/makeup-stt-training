"""
Extract a domain vocabulary (brand names, product names, technique terms)
from the training transcripts, for use as a generation-time sequence bias
in eval_moonshine.py / inference. Targets exactly the failure mode found in
manual review: rare proper nouns and product names ("Kajal pencil", "NARS",
"Voluminous Smoldering") get mis-transcribed because they're the words the
model has seen least often, even though aggregate WER looks fine.

Heuristic: capitalized words/phrases that are NOT at the start of a sentence
are almost always proper nouns (brand/product names) in these cleaned
transcripts, not just capitalization noise -- the Claude cleanup phase
capitalizes real proper nouns and leaves normal words lowercase. Frequency
filtering keeps this from becoming a giant list of one-off transcription
noise.

Usage:
    python build_vocabulary.py --dataset ./dataset --output ./vocabulary.json
    python build_vocabulary.py --min-count 2 --max-terms 300
"""

import argparse
import json
import os
import re
from collections import Counter

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
CONNECTOR_WORDS = {"of", "the", "and", "by", "for", "de", "&"}
STOPWORDS = {"i", "i'm", "i've", "i'll", "i'd"}

# Common technique/product-type words that matter for a makeup-tutorial
# transcript but aren't proper nouns, so the capitalization-based extraction
# above won't reliably surface them (they're lowercase in normal usage and
# some, like "Kajal", only ever appear embedded inside a specific product
# name in this corpus, never standalone). Always included regardless of
# corpus frequency.
CURATED_TECHNIQUE_TERMS = [
    "translucent", "Kajal", "waterline", "crease", "contour", "contouring",
    "concealer", "foundation", "primer", "mascara", "eyeliner", "eyeshadow",
    "eyebrow", "eyelid", "eyelash", "eyelashes", "bronzer", "blush",
    "highlighter", "setting spray", "lash line", "sponge", "blending",
    "buffing", "baking", "strobing", "cut crease", "smokey eye", "ombre",
    "matte", "dewy", "shimmer", "glitter", "pigment", "palette", "swatch",
    "undertone", "color correcting",
]


def strip_punctuation(word: str) -> str:
    return word.strip(".,!?;:\"'()")


def is_candidate_word(word: str) -> bool:
    cleaned = strip_punctuation(word)

    if len(cleaned) < 2:
        return False

    if not cleaned[0].isupper():
        return False

    if cleaned.lower() in STOPWORDS:
        return False

    return True


def extract_phrases(transcription: str) -> list:
    phrases = []

    for sentence in SENTENCE_SPLIT_RE.split(transcription.strip()):
        words = sentence.split()

        if len(words) < 2:
            continue

        # Skip the sentence-initial word -- capitalization there is just
        # normal sentence casing, not a signal of a proper noun.
        current_run = []

        for word in words[1:]:
            cleaned = strip_punctuation(word)

            if is_candidate_word(word):
                current_run.append(cleaned)
            elif cleaned.lower() in CONNECTOR_WORDS and current_run:
                current_run.append(cleaned)
            else:
                if current_run and is_candidate_word(current_run[-1]) is False:
                    current_run.pop()

                if current_run:
                    phrases.append(" ".join(current_run))

                current_run = []

        if current_run:
            while current_run and not is_candidate_word(current_run[-1]):
                current_run.pop()

            if current_run:
                phrases.append(" ".join(current_run))

    return phrases


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract a domain vocabulary for generation-time sequence biasing."
    )

    parser.add_argument("--dataset", default="./dataset")
    parser.add_argument("--splits", default="train")
    parser.add_argument(
        "--min-count",
        type=int,
        default=2,
        help="Drop terms seen fewer than this many times -- filters one-off transcription noise.",
    )
    parser.add_argument(
        "--max-terms",
        type=int,
        default=300,
        help="Cap the vocabulary size, keeping the most frequent terms. Curated terms are always kept regardless of this cap.",
    )
    parser.add_argument("--output", default="./vocabulary.json")

    args = parser.parse_args()

    splits = [split.strip() for split in args.splits.split(",")]
    counts = Counter()

    for split in splits:
        metadata_path = os.path.join(args.dataset, split, "metadata.jsonl")

        if not os.path.isfile(metadata_path):
            print(f"  (no metadata.jsonl for split '{split}', skipping)")
            continue

        with open(metadata_path, "r", encoding="utf-8") as metadata_file:
            for line in metadata_file:
                line = line.strip()

                if not line:
                    continue

                example = json.loads(line)
                transcription = example.get("transcription", "")

                for phrase in extract_phrases(transcription):
                    counts[phrase] += 1

    curated_lower = {t.lower() for t in CURATED_TECHNIQUE_TERMS}

    extracted = [
        {"term": term, "count": count}
        for term, count in counts.items()
        if count >= args.min_count and term.lower() not in curated_lower
    ]

    extracted.sort(key=lambda item: item["count"], reverse=True)
    extracted = extracted[: args.max_terms]

    curated = [
        {"term": term, "count": counts.get(term, 0)}
        for term in CURATED_TECHNIQUE_TERMS
    ]

    filtered = curated + extracted
    filtered.sort(key=lambda item: item["count"], reverse=True)

    with open(args.output, "w", encoding="utf-8") as output_file:
        json.dump(filtered, output_file, indent=2)

    print(f"Extracted {len(filtered)} vocabulary terms (min_count={args.min_count}) -> {args.output}")
    print("Top 30:")

    for item in filtered[:30]:
        print(f"  {item['count']:>4}  {item['term']}")


if __name__ == "__main__":
    main()
