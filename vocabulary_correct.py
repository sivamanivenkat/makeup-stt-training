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

WORD_RE = re.compile(r"[A-Za-z']+|[^A-Za-z'\s]")
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
    min_word_length: int = 4,
) -> str:
    """
    Slides windows (longest term length first, so multi-word product names
    take priority over single-word matches) over the tokenized text,
    replacing a window with a vocabulary term's canonical form when the
    combined text/phonetic similarity clears the threshold and the window
    doesn't already read as that term.

    min_word_length guards against short common words (e.g. "so", "to")
    matching a vocabulary term by coincidence -- phonetic codes on very
    short words collide easily and the blast radius of a wrong short-word
    replacement is proportionally larger relative to its own length.
    """
    tokens = WORD_RE.findall(text)
    word_positions = [i for i, tok in enumerate(tokens) if tok.isalpha() or "'" in tok]

    if not word_positions:
        return text

    consumed = [False] * len(tokens)
    max_term_length = max(vocabulary.keys()) if vocabulary else 0

    for window_size in range(max_term_length, 0, -1):
        candidates = vocabulary.get(window_size)

        if not candidates:
            continue

        for start_idx in range(len(word_positions) - window_size + 1):
            positions = word_positions[start_idx : start_idx + window_size]

            if any(consumed[p] for p in positions):
                continue

            window_words = [tokens[p] for p in positions]

            if any(len(w) < min_word_length for w in window_words) and window_size == 1:
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
                if best_term is not None and best_score >= threshold:
                    tokens[positions[0]] = best_term
                    for p in positions[1:]:
                        tokens[p] = ""
                    for p in positions:
                        consumed[p] = True

    corrected_parts = []

    for tok in tokens:
        if tok == "":
            continue

        is_word_like = any(c.isalnum() for c in tok)

        if corrected_parts and is_word_like:
            corrected_parts.append(" ")

        corrected_parts.append(tok)

    return "".join(corrected_parts)
