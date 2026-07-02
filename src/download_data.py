import json

from datasets import load_dataset

from config import (
    DATA_DIR,
    DOCVQA_CONFIG,
    DOCVQA_DATASET_ID,
    DOCVQA_SPLIT,
    SAMPLE_SIZE,
)
from hf_auth import get_hf_token


def main():
    token = get_hf_token()
    out_dir = DATA_DIR / "docvqa_sample"
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    stream = load_dataset(
        DOCVQA_DATASET_ID,
        DOCVQA_CONFIG,
        split=DOCVQA_SPLIT,
        streaming=True,
        token=token,
    )

    records = []
    for i, ex in enumerate(stream):
        if i >= SAMPLE_SIZE:
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
    print(f"saved {len(records)} samples to {out_dir}")


if __name__ == "__main__":
    main()
