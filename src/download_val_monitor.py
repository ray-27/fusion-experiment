"""Downloads a per-epoch monitoring validation slice from DocVQA's official
`validation` split. Skips past the first VAL_MONITOR_SKIP examples (already
used as the held-out test set in download_data.py) so there is zero overlap
between this monitoring set and the final test set."""

import argparse
import json

from datasets import load_dataset

from config import (
    DATA_DIR,
    DOCVQA_CONFIG,
    DOCVQA_DATASET_ID,
    DOCVQA_SPLIT,
    VAL_MONITOR_SAMPLE_SIZE,
    VAL_MONITOR_SKIP,
)
from hf_auth import get_hf_token


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=VAL_MONITOR_SAMPLE_SIZE)
    args = parser.parse_args()

    token = get_hf_token()
    out_dir = DATA_DIR / "docvqa_val_monitor"
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    stream = load_dataset(
        DOCVQA_DATASET_ID,
        DOCVQA_CONFIG,
        split=DOCVQA_SPLIT,
        streaming=True,
        token=token,
    ).skip(VAL_MONITOR_SKIP)

    records = []
    for i, ex in enumerate(stream):
        if i >= args.limit:
            break
        img_path = img_dir / f"{i:04d}.png"
        ex["image"].convert("RGB").save(img_path)
        records.append(
            {
                "id": i,
                "image": str(img_path.relative_to(out_dir)),
                "question": ex.get("question"),
                "answers": ex.get("answers"),
            }
        )

    with open(out_dir / "samples.json", "w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"saved {len(records)} monitoring-val samples to {out_dir}")


if __name__ == "__main__":
    main()
