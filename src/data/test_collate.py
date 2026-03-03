from pathlib import Path
from torch.utils.data import DataLoader

from src.data.torch_dataset import ComplaintsDataset, LabelEncoder, load_parquet_splits
from src.data.collate import TransformerCollator


def main():
    parquet_path = Path("src/data/processed/complaints_2024_3products.parquet")
    train_df, _, _ = load_parquet_splits(parquet_path)

    le = LabelEncoder.from_labels(train_df["product"].tolist())
    train_ds = ComplaintsDataset(train_df, le)

    model_name = "distilbert-base-uncased"  # fast + good baseline
    collate_fn = TransformerCollator(model_name=model_name, max_length=256)

    loader = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=collate_fn)

    batch = next(iter(loader))
    print("keys:", batch.keys())
    print("input_ids shape:", batch["input_ids"].shape)
    print("attention_mask shape:", batch["attention_mask"].shape)
    print("labels:", batch["labels"])
    print("labels shape:", batch["labels"].shape)


if __name__ == "__main__":
    main()