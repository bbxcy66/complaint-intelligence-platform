from __future__ import annotations
from typing import Dict, List

import torch
from transformers import AutoTokenizer


class TransformerCollator:
    def __init__(self, model_name: str, max_length: int = 256):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.max_length = max_length

    def __call__(self, batch: List[Dict[str, object]]) -> Dict[str, torch.Tensor]:
        texts = [x["text"] for x in batch]
        labels = torch.tensor([int(x["label"]) for x in batch], dtype=torch.long)

        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        enc["labels"] = labels
        return enc