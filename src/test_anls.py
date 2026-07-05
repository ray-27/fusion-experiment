"""Standalone sanity check + walkthrough for the ANLS metric (src/metrics.py).

ANLS (Average Normalized Levenshtein Similarity) is the standard DocVQA
scoring metric: normalize prediction + reference text, take the best
(highest) normalized edit-distance similarity across all reference answers,
then clip anything below ANLS_THRESHOLD to 0 (treats "way off" answers as
wrong rather than giving partial credit for coincidental character overlap).

Pure Python, no torch/model loading needed -- runs instantly.

Run directly:
    python src/test_anls.py
"""

from metrics import ANLS_THRESHOLD, anls, levenshtein, normalize

# (name, prediction, references, expected_anls)
CASES = [
    ("exact match", "Paris", ["Paris"], 1.0),
    ("case / punctuation / whitespace invariant", "  PARIS!!", ["paris"], 1.0),
    ("minor typo, above threshold (1 edit / len 5)", "Pari", ["Paris"], 0.8),
    ("best-of-multiple-references picks the closest one", "Bingo", ["bingo!", "Tangles"], 1.0),
    ("completely wrong, below threshold -> clipped to 0", "aaaa", ["bbbb"], 0.0),
    ("both empty strings -> treated as a perfect match", "", [""], 1.0),
    ("empty prediction vs a real answer -> 0", "", ["hello"], 0.0),
    ("exactly at threshold (0.5) -> inclusive, NOT clipped", "aabb", ["aacc"], 0.5),
]


def run_cases():
    print(f"ANLS_THRESHOLD = {ANLS_THRESHOLD}\n")
    n_passed = 0
    for name, pred, refs, expected in CASES:
        score = anls(pred, refs)
        ok = abs(score - expected) < 1e-9
        n_passed += ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        print(f"       prediction={pred!r}  references={refs!r}")
        print(f"       expected={expected:.4f}  got={score:.4f}")
    print(f"\n{n_passed}/{len(CASES)} cases passed")
    return n_passed == len(CASES)


def worked_example():
    """Manually reproduces exactly what anls() does internally, step by
    step, for one concrete DocVQA-style example -- so you can see the
    mechanism itself, not just the final score."""
    prediction = "10:00 - 11:30 AM"
    references = ["10:00 - 11:30 AM", "10:00 -  11:30 AM"]  # 2nd ref has a double space

    print("\n" + "=" * 70)
    print("WORKED EXAMPLE")
    print("=" * 70)
    print(f"prediction : {prediction!r}")
    print(f"references : {references!r}\n")

    pred_norm = normalize(prediction)
    print(f"1. normalize(prediction) -> {pred_norm!r}")
    print("   (lowercased, punctuation stripped, whitespace collapsed to single spaces)\n")

    best_nls, best_ref = 0.0, None
    for ref in references:
        ref_norm = normalize(ref)
        dist = levenshtein(pred_norm, ref_norm)
        max_len = max(len(pred_norm), len(ref_norm))
        nls = 1.0 if max_len == 0 else 1.0 - dist / max_len
        print(
            f"2. reference {ref!r} -> normalized {ref_norm!r}\n"
            f"     edit_distance={dist}, max_len={max_len}\n"
            f"     NLS = 1 - {dist}/{max_len} = {nls:.4f}"
        )
        if nls > best_nls:
            best_nls, best_ref = nls, ref

    final = best_nls if best_nls >= ANLS_THRESHOLD else 0.0
    print(f"\n3. best NLS across references = {best_nls:.4f}  (from reference {best_ref!r})")
    print(
        f"4. best NLS >= ANLS_THRESHOLD ({ANLS_THRESHOLD})? {best_nls >= ANLS_THRESHOLD} "
        f"-> final ANLS = {final:.4f}"
    )

    actual = anls(prediction, references)
    match = "matches" if abs(actual - final) < 1e-9 else "MISMATCH!"
    print(f"\nanls() actually returned: {actual:.4f}  -- {match} the manual walkthrough above")


def main():
    all_passed = run_cases()
    worked_example()
    print("\nANLS sanity check:", "ALL PASSED" if all_passed else "SOME FAILED")
    if not all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
