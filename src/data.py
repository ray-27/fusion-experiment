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


class Collator:
    """Builds one <image> placeholder per sample and masks the prompt so loss
    is only taken on the answer tokens (plus EOS)."""

    def __init__(self, tokenizer, image_processor):
        self.tokenizer = tokenizer
        self.image_processor = image_processor

    def __call__(self, batch):
        images = [b["image"] for b in batch]
        pixel_values = self.image_processor(images=images, return_tensors="pt")[
            "pixel_values"
        ]

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
            "input_ids_list": input_ids_list,
            "labels_list": labels_list,
        }
