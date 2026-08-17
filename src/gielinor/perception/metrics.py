"""Accuracy metrics for OCR evaluation against synthetic ground truth.

Kept in the library rather than in `benchmarks/` because the tests assert on
the same numbers the benchmarks print. If the benchmark's definition of
"character accuracy" drifted from the test's, the headline figure would stop
meaning what the passing test claims it means.

Character accuracy is `1 - levenshtein(truth, pred) / len(truth)`, clamped at
zero: the edit-distance form, not "fraction of positions that agree", because
a single dropped character shifts every later position and would score a
one-character error as a total failure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


def levenshtein(a: str, b: str) -> int:
    """Edit distance, two-row DP. O(len(a) * len(b)) time, O(len(b)) space."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def char_accuracy(truth: str, pred: str) -> float:
    if not truth:
        return 1.0 if not pred else 0.0
    return max(0.0, 1.0 - levenshtein(truth, pred) / len(truth))


def word_accuracy(truth: str, pred: str) -> float:
    """Fraction of whitespace-separated truth tokens present in the prediction.

    Multiset containment rather than positional equality, so a missing leading
    word does not zero out the rest of an otherwise perfect line.
    """
    tw = truth.split()
    if not tw:
        return 1.0 if not pred.split() else 0.0
    pool = list(pred.split())
    hit = 0
    for w in tw:
        if w in pool:
            pool.remove(w)
            hit += 1
    return hit / len(tw)


def align_lines(truth: Sequence[str], pred: Sequence[str]) -> list[tuple[str, str]]:
    """Pair each truth line with its best unused prediction.

    Index-wise pairing is wrong for evaluation: one dropped line would shift
    every later pairing and report a catastrophic score for an OCR that missed
    exactly one line. Greedy best-match isolates the real error.
    """
    remaining = list(pred)
    out: list[tuple[str, str]] = []
    for t in truth:
        if not remaining:
            out.append((t, ""))
            continue
        best_i = max(range(len(remaining)), key=lambda i: char_accuracy(t, remaining[i]))
        out.append((t, remaining.pop(best_i)))
    return out


def summarise(truth: Sequence[str], pred: Sequence[str]) -> dict[str, float]:
    """Character/word accuracy and exact-line rate over one region's lines."""
    pairs = align_lines(truth, pred)
    if not pairs:
        return {"char_accuracy": 1.0, "word_accuracy": 1.0, "exact_lines": 1.0, "lines": 0.0}
    ca = sum(char_accuracy(t, p) for t, p in pairs) / len(pairs)
    wa = sum(word_accuracy(t, p) for t, p in pairs) / len(pairs)
    ex = sum(1.0 for t, p in pairs if t == p) / len(pairs)
    return {"char_accuracy": ca, "word_accuracy": wa, "exact_lines": ex, "lines": float(len(pairs))}


__all__ = ["align_lines", "char_accuracy", "levenshtein", "summarise", "word_accuracy"]
