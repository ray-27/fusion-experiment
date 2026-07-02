import argparse
import csv
import os
import random

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from torch.utils.data import DataLoader, Subset

from config import CHECKPOINTS_DIR, DATA_DIR, RESULTS_DIR
from connectors import CONNECTORS, build_connector
from data import Collator, DocVQADataset
from device import get_device
from metrics import anls
from models import load_llm, load_vision_encoder
from notify import notify_discord
from vlm import FusionVLM

# Train: DocVQA's own official `train` split (in-domain document reading).
# Val (per-epoch monitoring): a disjoint slice of the `validation` split.
# Test (final, held out): the original first-100 `validation` slice.
# All three are non-overlapping -- see download_train_data.py /
# download_val_monitor.py / download_data.py.
TRAIN_DATA_DIR = DATA_DIR / "train_sample"
VAL_MONITOR_DATA_DIR = DATA_DIR / "docvqa_val_monitor"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--connector", default=None, help="omit to run all mechanisms")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument(
        "--notify", action="store_true",
        help="send a Discord webhook message when the full run finishes",
    )
    p.add_argument(
        "--train-limit", type=int, default=None,
        help="cap the training pool size, for fast local smoke tests",
    )
    p.add_argument(
        "--test-limit", type=int, default=None,
        help="cap the DocVQA held-out test set size, for fast local smoke tests",
    )
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(vlm, loader, dataset, max_new_tokens):
    """Loss + ANLS over a held-out set. Used both for the training pool's
    internal val split (per-epoch monitoring) and the DocVQA test set
    (final, never-trained-on evaluation)."""
    vlm.eval()
    total_loss, n_batches = 0.0, 0
    for batch in loader:
        total_loss += vlm(**batch).loss.item()
        n_batches += 1
    mean_loss = total_loss / max(n_batches, 1)

    scores = []
    for rec in dataset:
        pred = vlm.generate(rec["image"], rec["question"], max_new_tokens=max_new_tokens)
        scores.append(anls(pred, rec["answers"]))
    vlm.train()
    return mean_loss, sum(scores) / max(len(scores), 1)


def train_one(name, args, device, writers):
    set_seed(args.seed)
    print(f"\n=== connector: {name} ===")

    image_processor, vision_model = load_vision_encoder()
    tokenizer, llm = load_llm()
    vision_dim = vision_model.config.vision_config.hidden_size
    llm_dim = llm.config.hidden_size

    connector = build_connector(
        name, vision_dim, llm_dim, num_llm_layers=llm.config.num_hidden_layers
    ).to(device)
    vlm = FusionVLM(vision_model, connector, llm, tokenizer, image_processor).to(device)
    vlm.train()

    trainable = [p for p in vlm.connector.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    print(f"trainable connector params: {n_trainable:,}")

    # Train: DocVQA's own official `train` split.
    train_pool = DocVQADataset(split_dir=TRAIN_DATA_DIR)
    if args.train_limit:
        train_pool = Subset(train_pool, list(range(min(args.train_limit, len(train_pool)))))
    collate = Collator(tokenizer, image_processor)
    train_loader = DataLoader(
        train_pool, batch_size=args.batch_size, shuffle=True, collate_fn=collate
    )

    # Val (per-epoch monitoring): disjoint slice of the `validation` split.
    val_dataset = DocVQADataset(split_dir=VAL_MONITOR_DATA_DIR)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate
    )

    # Test: original held-out slice, touched once at the very end.
    test_dataset = DocVQADataset()
    if args.test_limit:
        test_dataset = Subset(test_dataset, list(range(min(args.test_limit, len(test_dataset)))))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    optimizer = torch.optim.AdamW(trainable, lr=args.lr)

    step = 0
    for epoch in range(args.epochs):
        epoch_losses = []
        for batch in train_loader:
            optimizer.zero_grad()
            loss = vlm(**batch).loss
            loss.backward()
            optimizer.step()
            step += 1
            epoch_losses.append(loss.item())
            writers["train_step"].writerow([name, epoch, step, f"{loss.item():.6f}"])
            print(f"[{name}] epoch {epoch} step {step} train_loss {loss.item():.4f}")

        epoch_mean_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
        writers["train_epoch"].writerow([name, epoch, f"{epoch_mean_loss:.6f}"])
        print(f"[{name}] epoch {epoch} MEAN train_loss {epoch_mean_loss:.4f}")

        val_loss, val_anls_score = evaluate(vlm, val_loader, val_dataset, args.max_new_tokens)
        writers["val_loss"].writerow([name, epoch, f"{val_loss:.6f}"])
        writers["val_anls"].writerow([name, epoch, f"{val_anls_score:.6f}"])
        print(f"[{name}] epoch {epoch} val_loss {val_loss:.4f} val_anls {val_anls_score:.4f}")

    test_loss, test_anls_score = evaluate(vlm, test_loader, test_dataset, args.max_new_tokens)
    writers["test"].writerow([name, f"{test_loss:.6f}", f"{test_anls_score:.6f}"])
    print(f"[{name}] FINAL DocVQA test (held out): loss {test_loss:.4f} anls {test_anls_score:.4f}")

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = CHECKPOINTS_DIR / f"connector_{name}.pt"
    torch.save(vlm.connector.state_dict(), ckpt)
    print(f"[{name}] saved connector -> {ckpt}")

    return n_trainable, test_anls_score


def main():
    args = parse_args()
    device = get_device()
    names = [args.connector] if args.connector else list(CONNECTORS)
    print(f"device: {device} | connectors: {names}")

    if not (TRAIN_DATA_DIR / "samples.json").exists():
        raise SystemExit(
            f"no training data at {TRAIN_DATA_DIR} -- run `python src/download_train_data.py` first"
        )
    if not (VAL_MONITOR_DATA_DIR / "samples.json").exists():
        raise SystemExit(
            f"no monitoring-val data at {VAL_MONITOR_DATA_DIR} -- "
            "run `python src/download_val_monitor.py` first"
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    paths = {
        "train_step": RESULTS_DIR / "train_loss.csv",
        "train_epoch": RESULTS_DIR / "train_loss_epoch.csv",
        "val_loss": RESULTS_DIR / "val_loss.csv",
        "val_anls": RESULTS_DIR / "val_anls.csv",
        "test": RESULTS_DIR / "test_metrics.csv",
    }
    headers = {
        "train_step": ["connector", "epoch", "step", "loss"],
        "train_epoch": ["connector", "epoch", "mean_loss"],
        "val_loss": ["connector", "epoch", "loss"],
        "val_anls": ["connector", "epoch", "anls"],
        "test": ["connector", "test_loss", "test_anls"],
    }

    files = {k: open(v, "w", newline="") for k, v in paths.items()}
    writers = {k: csv.writer(f) for k, f in files.items()}
    for k, w in writers.items():
        w.writerow(headers[k])

    efficiency_rows = []
    summary_lines = [f"VLM fusion ablation run complete ({device})."]
    try:
        for name in names:
            n_trainable, final_anls = train_one(name, args, device, writers)
            for f in files.values():
                f.flush()
            anls_per_million_params = final_anls / (n_trainable / 1e6)
            efficiency_rows.append((name, n_trainable, final_anls, anls_per_million_params))
            summary_lines.append(
                f"- {name}: test_anls={final_anls:.4f}, params={n_trainable:,}"
            )
    finally:
        for f in files.values():
            f.close()

    efficiency_path = RESULTS_DIR / "efficiency.csv"
    with open(efficiency_path, "w", newline="") as f_eff:
        writer = csv.writer(f_eff)
        writer.writerow(["connector", "trainable_params", "final_anls", "anls_per_million_params"])
        for row in efficiency_rows:
            writer.writerow([row[0], row[1], f"{row[2]:.6f}", f"{row[3]:.6f}"])

    print(f"\ntrain loss (step)  -> {paths['train_step']}")
    print(f"train loss (epoch) -> {paths['train_epoch']}")
    print(f"val loss           -> {paths['val_loss']}")
    print(f"val anls           -> {paths['val_anls']}")
    print(f"test metrics       -> {paths['test']}")
    print(f"efficiency         -> {efficiency_path}")

    if args.notify:
        notify_discord("\n".join(summary_lines))


if __name__ == "__main__":
    main()
