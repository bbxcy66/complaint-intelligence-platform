from pathlib import Path

from torch.utils.data import DataLoader

from src.data.torch_dataset import ComplaintsDataset, LabelEncoder, load_parquet_splits


def main():
    parquet_path = Path("src/data/processed/complaints_2024_3products.parquet")

    train_df, val_df, test_df = load_parquet_splits(parquet_path)

    le = LabelEncoder.from_labels(train_df["product"].tolist())

    train_ds = ComplaintsDataset(train_df, le)
    val_ds = ComplaintsDataset(val_df, le)

    print("labels:", le.label2id)
    print("train size:", len(train_ds), "val size:", len(val_ds), "test size:", len(test_df))

    loader = DataLoader(train_ds, batch_size=4, shuffle=True)

    batch = next(iter(loader))
    print("batch keys:", batch.keys())
    print("batch label ids:", batch["label"])
    print("first text snippet:", batch["text"][0][:200])


if __name__ == "__main__":
    main()