from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup

from src.data.torch_dataset import ComplaintsDataset, LabelEncoder, load_parquet_splits
from src.data.collate import TransformerCollator


def get_device() -> torch.device:
    # Apple Silicon acceleration if available
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def move_to_device(batch: dict, device: torch.device) -> dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def main():
    # Paths
    parquet_path = Path("src/data/processed/complaints_2024_3products.parquet")
    out_dir = Path("outputs/models/distilbert_v1")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Hyperparams
    model_name = "distilbert-base-uncased"
    max_length = 128
    batch_size = 4
    lr = 2e-5
    epochs = 1
    warmup_ratio = 0.1

    # Subset for faster training
    train_n = 40000
    val_n = 8000

    device = get_device()
    print("device:", device)

    # Load data
    train_df, val_df, _ = load_parquet_splits(parquet_path)

    # Sample smaller subset for speed
    if train_n is not None and train_n < len(train_df):
        train_df = train_df.sample(n=train_n, random_state=42).reset_index(drop=True)
    if val_n is not None and val_n < len(val_df):
        val_df = val_df.sample(n=val_n, random_state=42).reset_index(drop=True)

    le = LabelEncoder.from_labels(train_df["product"].tolist())
    num_labels = len(le.label2id)
    print("labels:", le.label2id)

    train_ds = ComplaintsDataset(train_df, le)
    val_ds = ComplaintsDataset(val_df, le)

    collate_fn = TransformerCollator(model_name=model_name, max_length=max_length)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    # Model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=le.id2label,
        label2id=le.label2id,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    total_steps = epochs * len(train_loader)
    warmup_steps = int(warmup_ratio * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    global_step = 0

    try:
        # Train
        model.train()
        for epoch in range(1, epochs + 1):
            running_loss = 0.0

            for batch in train_loader:
                batch = move_to_device(batch, device)
                # DistilBERT doesn't need token_type_ids
                batch.pop("token_type_ids", None)

                out = model(**batch)
                loss = out.loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                if device.type == "mps":
                    torch.mps.empty_cache()

                running_loss += loss.item()
                global_step += 1

                if global_step % 200 == 0:
                    avg = running_loss / 200
                    print(f"step {global_step}/{total_steps} - loss {avg:.4f}")
                    running_loss = 0.0

            # Validate (simple accuracy)
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = move_to_device(batch, device)
                    batch.pop("token_type_ids", None)

                    logits = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                    ).logits
                    preds = torch.argmax(logits, dim=1)
                    correct += (preds == batch["labels"]).sum().item()
                    total += batch["labels"].numel()

            acc = correct / max(total, 1)
            print(f"epoch {epoch} val_acc={acc:.4f}")
            model.train()

    except KeyboardInterrupt:
        print("Training interrupted (Ctrl+C) — saving checkpoint...")

    finally:
        print("Saving checkpoint...")
        model.save_pretrained(out_dir)
        collate_fn.tokenizer.save_pretrained(out_dir)
        print("Model saved to:", out_dir)


if __name__ == "__main__":
    main()
