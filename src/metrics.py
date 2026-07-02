import re
import string

ANLS_THRESHOLD = 0.5


def normalize(text: str) -> str:
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", text).strip()


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[-1]


def anls(prediction: str, references) -> float:
    """Average Normalized Levenshtein Similarity (standard DocVQA metric).

    For each reference: NLS = 1 - edit_distance(pred, ref) / max(len(pred), len(ref)).
    Take the best (max) NLS across references; below ANLS_THRESHOLD counts as 0."""
    pred = normalize(prediction)
    best = 0.0
    for ref in references:
        ref = normalize(ref)
        max_len = max(len(pred), len(ref))
        nls = 1.0 if max_len == 0 else 1.0 - levenshtein(pred, ref) / max_len
        best = max(best, nls)
    return best if best >= ANLS_THRESHOLD else 0.0
