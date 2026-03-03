from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class LabelEncoder:
    label2id: Dict[str, int]
    id2label: Dict[int, str]

    @classmethod
    def from_labels(cls, labels: List[str]) -> "LabelEncoder":
        uniq = sorted(set(labels))
        label2id = {l: i for i, l in enumerate(uniq)}
        id2label = {i: l for l, i in label2id.items()}
        return cls(label2id=label2id, id2label=id2label)

    def encode(self, label: str) -> int:
        return self.label2id[label]

    def decode(self, idx: int) -> str:
        return self.id2label[idx]


class ComplaintsDataset(Dataset):
    """
    Minimal dataset for transformer fine-tuning later.
    Returns raw text + integer label. Tokenization will be done in the collate_fn later.
    """

    def __init__(self, df: pd.DataFrame, label_encoder: LabelEncoder):
        self.df = df.reset_index(drop=True)
        self.le = label_encoder

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        row = self.df.iloc[idx]
        text = str(row["text"])
        label = str(row["product"])
        y = self.le.encode(label)
        return {"text": text, "label": y}


def load_parquet_splits(parquet_path: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_parquet(parquet_path)
    needed = {"text", "product", "split"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in parquet: {missing}. Found: {list(df.columns)}")

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()
    return train_df, val_df, test_df