"""Downloads DocVQA's own official `train` split for training -- in-domain
document reading, matching the brief's target task, instead of generic
captions. This split isn't exposed via the dataset's declared split names,
but its parquet shards are directly loadable. Separate from download_data.py
(the held-out test set: first SAMPLE_SIZE examples of `validation`)."""

import argparse
import json

from datasets import load_dataset
from huggingface_hub import HfApi
from tqdm import tqdm

from config import DATA_DIR, TRAIN_CONFIG, TRAIN_DATASET_ID, TRAIN_SAMPLE_SIZE
from hf_auth import get_hf_token

TRAIN_SHARD = "DocVQA/train-00000-of-00012.parquet"
TRAIN_PARQUET = f"hf://datasets/{TRAIN_DATASET_ID}/{TRAIN_SHARD}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit", type=int, default=TRAIN_SAMPLE_SIZE,
        help="number of training samples to download (default: config.TRAIN_SAMPLE_SIZE)",
    )
    args = parser.parse_args()

    token = get_hf_token()
    out_dir = DATA_DIR / "train_sample"
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    try:
        shard_size = HfApi().dataset_info(TRAIN_DATASET_ID, files_metadata=True)
        shard_size = next(
            s.size for s in shard_size.siblings if s.rfilename == TRAIN_SHARD
        )
        print(
            f"source shard: {TRAIN_SHARD} ({shard_size / 1e6:.0f} MB) -- "
            f"streamed lazily, not fully downloaded upfront"
        )
    except (StopIteration, Exception):
        pass
    print(f"target: {args.limit} samples -> {out_dir}")

    stream = load_dataset(
        "parquet",
        data_files={"train": TRAIN_PARQUET},
        split="train",
        streaming=True,
        token=token,
    )

    records = []
    for i, ex in enumerate(tqdm(stream, total=args.limit, unit="sample")):
        if i >= args.limit:
            break
        img_path = img_dir / f"{i:05d}.png"
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
    print(f"saved {len(records)} training samples to {out_dir}")


if __name__ == "__main__":
    main()
