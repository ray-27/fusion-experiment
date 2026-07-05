import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from config import DATA_DIR, IMAGE_TOKEN

PROMPT_TEMPLATE = f"{IMAGE_TOKEN}\n{{question}}\nAnswer:"


class DocVQADataset(Dataset):
    def __init__(self, split_dir=None):
        self.root = Path(split_dir or (DATA_DIR / "docvqa_sample"))
        self.records = json.loads((self.root / "samples.json").read_text())

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        rec = self.records[i]
        image = Image.open(self.root / rec["image"]).convert("RGB")
        answers = rec["answers"] or [""]
        return {
            "image": image,
            "question": rec["question"],
            "answer": answers[0],
            "answers": answers,
        }


class WithVisionFeatures(Dataset):
    """Wraps a DocVQADataset (or a Subset of one) to attach a precomputed
    per-sample vision-encoder feature tensor (see vision_cache.py). Once
    wrapped, the Collator stacks these cached features straight into a batch
    instead of running the image processor + frozen vision tower again."""

    def __init__(self, dataset, vision_features):
        assert len(dataset) == vision_features.size(0), (
            f"vision_features has {vision_features.size(0)} entries but "
            f"dataset has {len(dataset)} -- did the split change size?"
        )
        self.dataset = dataset
        self.vision_features = vision_features

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        item = dict(self.dataset[i])
        item["vision_features"] = self.vision_features[i]
        return item


class Collator:
    """Builds one <image> placeholder per sample and masks the prompt so loss
    is only taken on the answer tokens (plus EOS).

    If samples carry a precomputed "vision_features" tensor (see
    WithVisionFeatures), those are stacked directly and the image
    processor/vision tower are skipped entirely; otherwise falls back to
    processing raw PIL images into pixel_values as before."""

    def __init__(self, tokenizer, image_processor):
        self.tokenizer = tokenizer
        self.image_processor = image_processor

    def __call__(self, batch):
        if "vision_features" in batch[0]:
            pixel_values = None
            vision_features = torch.stack([b["vision_features"] for b in batch])
        else:
            images = [b["image"] for b in batch]
            pixel_values = self.image_processor(images=images, return_tensors="pt")[
                "pixel_values"
            ]
            vision_features = None

        input_ids_list, labels_list = [], []
        for b in batch:
            prompt = PROMPT_TEMPLATE.format(question=b["question"])
            prompt_ids = self.tokenizer(prompt, add_special_tokens=True).input_ids
            answer_ids = self.tokenizer(
                f" {b['answer']}", add_special_tokens=False
            ).input_ids + [self.tokenizer.eos_token_id]

            input_ids = prompt_ids + answer_ids
            labels = [-100] * len(prompt_ids) + answer_ids
            input_ids_list.append(torch.tensor(input_ids, dtype=torch.long))
            labels_list.append(torch.tensor(labels, dtype=torch.long))

        return {
            "pixel_values": pixel_values,
            "vision_features": vision_features,
            "input_ids_list": input_ids_list,
            "labels_list": labels_list,
        }
