"""Evaluates the natively-multimodal Qwen2-VL-2B-Instruct baseline on the same
held-out DocVQA test set used for the connector ablation, so its ANLS gives a
real-world upper-bound reference point next to the from-scratch fusion
mechanisms (which are trained with a tiny frozen-backbone connector, not
end-to-end on massive multimodal data like Qwen2-VL was).

Qwen2-VL's vision tower uses dynamic resolution -- large document scans can
otherwise blow attention memory past 20GB for a single image. We cap
min/max pixels (see config.py) and run in batches sized to fit whatever GPU
you have.

Prompts append a short-answer instruction by default so Qwen2-VL's chat-style
full-sentence answers don't tank ANLS purely on string-length/format grounds
against DocVQA's terse extractive references. Disable with
--no-short-answer-prompt to see raw, verbose answers.

Usage:
    python src/eval_baseline.py
    python src/eval_baseline.py --limit 20 --batch-size 4 --notify
"""

import argparse
import csv
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from qwen_vl_utils import process_vision_info

from config import BASELINE_MAX_PIXELS, BASELINE_MIN_PIXELS, RESULTS_DIR
from data import DocVQADataset
from device import get_device
from metrics import anls
from models import load_baseline_vlm
from notify import notify_discord

# ANLS is an edit-distance metric tuned for terse, extractive DocVQA-style
# answers ("ITC Limited"). Qwen2-VL-Instruct is chat-tuned and will otherwise
# answer in full sentences ("The name of the company is ITC Limited."), which
# tanks ANLS on string-length grounds alone even when the content is correct.
# This is the standard VQA-eval prompting trick (also used by LLaVA/Qwen-VL's
# own eval harnesses) to get answer style to match the reference format.
SHORT_ANSWER_SUFFIX = "\nAnswer the question using a single word or phrase."


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None, help="cap test set size, for a quick run")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument(
        "--no-short-answer-prompt", action="store_true",
        help="disable the 'answer in a word or phrase' suffix (verbose, full-sentence answers)",
    )
    p.add_argument(
        "--max-pixels", type=int, default=BASELINE_MAX_PIXELS,
        help="lower this if you hit OOM; raise it (GPU permitting) for sharper document text",
    )
    p.add_argument("--min-pixels", type=int, default=BASELINE_MIN_PIXELS)
    p.add_argument(
        "--notify", action="store_true",
        help="send a Discord webhook message when the benchmark finishes",
    )
    return p.parse_args()


def batched(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


@torch.no_grad()
def run(processor, model, device, records, max_new_tokens, batch_size, short_answer_prompt=True):
    suffix = SHORT_ANSWER_SUFFIX if short_answer_prompt else ""
    rows = []
    for batch in batched(records, batch_size):
        conversations = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": rec["image"]},
                        {"type": "text", "text": rec["question"] + suffix},
                    ],
                }
            ]
            for rec in batch
        ]
        texts = [
            processor.apply_chat_template(c, tokenize=False, add_generation_prompt=True)
            for c in conversations
        ]
        image_inputs, video_inputs = process_vision_info(conversations)
        inputs = processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(device)

        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        trimmed = output_ids[:, inputs.input_ids.shape[1] :]
        predictions = processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        for rec, prediction in zip(batch, predictions):
            prediction = prediction.strip()
            score = anls(prediction, rec["answers"])
            rows.append((rec["question"], prediction, rec["answers"], score))
            print(
                f"[{len(rows) - 1}] Q: {rec['question']!r} PRED: {prediction!r} "
                f"REF: {rec['answers']} ANLS: {score:.3f}"
            )
    return rows


def main():
    args = parse_args()
    device = get_device()
    print(f"device: {device}")

    dataset = DocVQADataset()
    records = [dataset[i] for i in range(min(args.limit, len(dataset)) if args.limit else len(dataset))]
    print(f"held-out DocVQA test set: {len(records)} samples, batch_size={args.batch_size}")

    print(f"loading Qwen2-VL-2B-Instruct baseline (min_pixels={args.min_pixels}, max_pixels={args.max_pixels})...")
    processor, model = load_baseline_vlm(min_pixels=args.min_pixels, max_pixels=args.max_pixels)

    try:
        rows = run(
            processor, model, device, records, args.max_new_tokens, args.batch_size,
            short_answer_prompt=not args.no_short_answer_prompt,
        )
    except RuntimeError as e:
        # Covers CUDA OOM and MPS "Invalid buffer size" -- both mean the
        # batch/resolution was too large for available device memory.
        msg = (
            f"Qwen2-VL baseline eval ran out of memory "
            f"(batch_size={args.batch_size}, max_pixels={args.max_pixels}): {e}\n"
            "Try a smaller --batch-size and/or --max-pixels."
        )
        print(f"[error] {msg}")
        if args.notify:
            notify_discord(msg)
        raise

    mean_anls = sum(r[3] for r in rows) / max(len(rows), 1)
    print(f"\nQwen2-VL-2B-Instruct baseline ANLS on held-out DocVQA test set: {mean_anls:.4f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "baseline_metrics.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["question", "prediction", "references", "anls"])
        for q, pred, refs, score in rows:
            writer.writerow([q, pred, "|".join(refs), f"{score:.6f}"])
        writer.writerow(["MEAN", "", "", f"{mean_anls:.6f}"])
    print(f"saved -> {out_path}")

    if args.notify:
        notify_discord(
            f"Qwen2-VL-2B-Instruct baseline eval complete ({device}).\n"
            f"n={len(rows)} samples, ANLS={mean_anls:.4f}\n"
            f"saved -> {out_path}"
        )


if __name__ == "__main__":
    main()
