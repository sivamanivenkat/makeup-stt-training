"""
Post-processing correction: fix ASR output words that are close to a known
domain term (build_vocabulary.py output) but not exact, without touching
generation at all.

Replaces the generation-time sequence-bias approach (removed after it
proved unsafe -- unconditional single-token biasing caused a runaway
numeric-hallucination loop on an otherwise-clean example, see git history).
Post-processing can't cause that failure mode since it only ever edits
already-finished text.

Two signals, since ASR errors come in two different flavors:
  - spelling-level noise ("pencilt" for "pencil", "NARRS" for "NARS") --
    caught by plain edit-distance similarity.
  - genuine mishearing, where the wrong word is phonetically close but
    textually very different ("bruneat" for "brunette", "pajole" for
    "Kajal") -- plain edit distance misses these; metaphone/soundex codes
    catch them because they encode how the word sounds, not how it's
    spelled.

Operates on character spans of the original string rather than
tokenize-and-rejoin, so punctuation, digits, and hyphenated words are never
touched or mis-respaced by a replacement elsewhere in the sentence (an
earlier tokenize/rejoin version silently corrupted "10" -> "1 0" and
"double-ended" -> "double- ended" on every single call, regardless of
whether any vocabulary term matched -- pure WER damage with zero benefit).

Usage (as a library):
    from vocabulary_correct import load_vocabulary, correct_transcript
    vocab = load_vocabulary("./vocabulary.json")
    corrected = correct_transcript(raw_prediction_text, vocab)
"""

import json
import re

import jellyfish
from rapidfuzz import fuzz
from spellchecker import SpellChecker

WORD_RE = re.compile(r"[A-Za-z']+")
_SPELLCHECKER = SpellChecker()


def load_vocabulary(vocab_file: str) -> dict:
    """
    Returns {word_count: [(term_str, [term_words_lower]), ...]}, grouped by
    how many words each term has, so correction only ever compares
    same-length windows against same-length terms.
    """
    with open(vocab_file, "r", encoding="utf-8") as f:
        raw_terms = json.load(f)

    by_length: dict = {}

    for item in raw_terms:
        term = item["term"] if isinstance(item, dict) else item
        words = term.split()
        by_length.setdefault(len(words), []).append((term, [w.lower() for w in words]))

    return by_length


def _phonetic_code(words_lower: list) -> str:
    return " ".join(jellyfish.metaphone(w) for w in words_lower)


def _similarity(window_words_lower: list, term_words_lower: list) -> float:
    window_text = " ".join(window_words_lower)
    term_text = " ".join(term_words_lower)

    text_score = fuzz.ratio(window_text, term_text)
    phonetic_score = fuzz.ratio(_phonetic_code(window_words_lower), _phonetic_code(term_words_lower))

    return max(text_score, phonetic_score)


def correct_transcript(
    text: str,
    vocabulary: dict,
    threshold: float = 88.0,
    multi_word_threshold: float = 80.0,
    min_word_length: int = 4,
) -> str:
    """
    Slides windows (longest term length first, so multi-word product names
    take priority over single-word matches) over the words found in text,
    replacing a window's exact character span with a vocabulary term's
    canonical form when the combined text/phonetic similarity clears the
    threshold. Everything outside a replaced span -- spacing, punctuation,
    digits, hyphens -- is copied through unchanged.

    multi_word_threshold is lower than the single-word threshold: matching
    TWO (or more) words' combined text/phonetic signature by pure
    coincidence is far rarer than a single short word colliding with some
    vocabulary entry, so multi-word windows can safely use a looser bar.

    min_word_length guards against short common words (e.g. "so", "to")
    matching a vocabulary term by coincidence -- phonetic codes on very
    short words collide easily and the blast radius of a wrong short-word
    replacement is proportionally larger relative to its own length.
    """
    matches = list(WORD_RE.finditer(text))

    if not matches:
        return text

    consumed = [False] * len(matches)
    replacements = {}  # start_match_index -> (end_match_index, replacement_text)
    max_term_length = max(vocabulary.keys()) if vocabulary else 0

    for window_size in range(max_term_length, 0, -1):
        candidates = vocabulary.get(window_size)

        if not candidates:
            continue

        for start_idx in range(len(matches) - window_size + 1):
            window_matches = matches[start_idx : start_idx + window_size]
            indices = range(start_idx, start_idx + window_size)

            if any(consumed[i] for i in indices):
                continue

            window_words = [m.group() for m in window_matches]

            if window_size == 1 and len(window_words[0]) < min_word_length:
                continue

            # Hard gate: only ever consider correcting a window if every word
            # in it is NOT a recognized English word. Real ASR errors that
            # need vocabulary correction produce gibberish ("pajole",
            # "pencilt", "NARRS") -- common valid words ("we", "kind",
            # "super") must never be candidates, since fuzzy/phonetic
            # similarity to a large vocabulary is high-noise on short common
            # words and will always find a coincidental match otherwise.
            if any(_SPELLCHECKER.known([w.lower()]) for w in window_words):
                continue

            window_words_lower = [w.lower() for w in window_words]

            best_term = None
            best_score = 0.0

            for term, term_words_lower in candidates:
                if window_words_lower == term_words_lower:
                    break  # already correct, nothing to do

                score = _similarity(window_words_lower, term_words_lower)

                if score > best_score:
                    best_score = score
                    best_term = term
            else:
                effective_threshold = threshold if window_size == 1 else multi_word_threshold

                if best_term is not None and best_score >= effective_threshold:
                    replacements[start_idx] = (start_idx + window_size - 1, best_term)
                    for i in indices:
                        consumed[i] = True

    if not replacements:
        return text

    output_parts = []
    cursor = 0
    i = 0

    while i < len(matches):
        if i in replacements:
            end_idx, replacement_text = replacements[i]
            output_parts.append(text[cursor : matches[i].start()])
            output_parts.append(replacement_text)
            cursor = matches[end_idx].end()
            i = end_idx + 1
        else:
            i += 1

    output_parts.append(text[cursor:])

    return "".join(output_parts)
