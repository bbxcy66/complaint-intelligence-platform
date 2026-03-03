import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import yaml


def pick_column(cols: list[str], candidates: list[str]) -> str:
    low = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in low:
            return low[cand.lower()]
    raise ValueError(
        f"Could not find any of these columns in CSV: {candidates}\n"
        f"Available columns: {cols}"
    )


def main():
    cfg = yaml.safe_load(Path("configs/data.yaml").read_text())

    raw_path = Path(cfg["raw_csv"])
    out_parquet = Path(cfg["processed_parquet"])
    out_splits = Path(cfg["splits_json"])
    out_parquet.parent.mkdir(parents=True, exist_ok=True)

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw CSV not found at: {raw_path.resolve()}")

    chunksize = int(cfg.get("chunksize", 2000000))
    max_rows = int(cfg.get("max_rows", 1200000))
    rs = int(cfg["random_state"])

    start_date = pd.to_datetime(cfg["start_date"])
    end_date = cfg.get("end_date")
    end_date = pd.to_datetime(end_date) if end_date else None

    products_keep = set(cfg.get("products_keep", []))

    # Peek header
    header_df = pd.read_csv(raw_path, nrows=5)
    date_col = pick_column(list(header_df.columns), cfg["date_col_candidates"])
    text_col = pick_column(list(header_df.columns), cfg["text_col_candidates"])
    label_col = pick_column(list(header_df.columns), cfg["label_col_candidates"])

    usecols = list({date_col, text_col, label_col})

    rng = np.random.default_rng(rs)
    kept = []
    total_seen = 0
    total_kept = 0

    for chunk in pd.read_csv(raw_path, usecols=usecols, chunksize=chunksize):
        total_seen += len(chunk)

        # parse dates + filter
        chunk[date_col] = pd.to_datetime(chunk[date_col], errors="coerce")
        chunk = chunk[chunk[date_col].notna()]
        chunk = chunk[chunk[date_col] >= start_date]
        if end_date is not None:
            chunk = chunk[chunk[date_col] <= end_date]
        if len(chunk) == 0:
            continue

        # product filter (3 products)
        chunk[label_col] = chunk[label_col].astype(str).str.strip()
        if products_keep:
            chunk = chunk[chunk[label_col].isin(products_keep)]
        if len(chunk) == 0:
            continue

        # text cleaning
        chunk[text_col] = (
            chunk[text_col]
            .astype(str)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

        min_len = int(cfg["min_text_len"])
        chunk = chunk[chunk[text_col].str.len() >= min_len]
        chunk = chunk[chunk[label_col].notna() & (chunk[label_col] != "")]
        if len(chunk) == 0:
            continue

        # cap total rows
        space_left = max_rows - total_kept
        if space_left <= 0:
            break

        if len(chunk) <= space_left:
            kept.append(chunk)
            total_kept += len(chunk)
        else:
            idx = rng.choice(chunk.index.to_numpy(), size=space_left, replace=False)
            kept.append(chunk.loc[idx])
            total_kept += space_left
            break

        if total_seen % (chunksize * 5) == 0:
            print(f"seen={total_seen:,} kept={total_kept:,}")

    if not kept:
        raise RuntimeError("No rows kept. Check date + product filters and column names.")

    df = pd.concat(kept, ignore_index=True)
    df = df.rename(columns={text_col: "text", label_col: "product", date_col: "date_received"})

    # optional: keep only reasonably common labels (here products are only 3 anyway)
    min_label_count = int(cfg.get("min_label_count", 50))
    vc = df["product"].value_counts()
    df = df[df["product"].isin(vc[vc >= min_label_count].index)].copy()

    # Split stratified by product
    test_size = float(cfg["test_size"])
    val_size = float(cfg["val_size"])

    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=rs, stratify=df["product"]
    )
    val_frac_of_train = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_df, test_size=val_frac_of_train, random_state=rs, stratify=train_df["product"]
    )

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"
    out = pd.concat([train_df, val_df, test_df], ignore_index=True)

    out.to_parquet(out_parquet, index=False)

    splits = {
        "raw_rows_seen": int(total_seen),
        "rows_sampled": int(len(df)),
        "n_total": int(len(out)),
        "n_train": int((out["split"] == "train").sum()),
        "n_val": int((out["split"] == "val").sum()),
        "n_test": int((out["split"] == "test").sum()),
        "date_min": str(out["date_received"].min()),
        "date_max": str(out["date_received"].max()),
        "product_counts": out["product"].value_counts().to_dict(),
        "date_col_used": date_col,
        "text_col_used": text_col,
        "product_col_used": label_col,
    }
    out_splits.write_text(json.dumps(splits, indent=2))

    print("Saved:", out_parquet)
    print("Saved:", out_splits)
    print("Summary:", json.dumps(splits, indent=2))


if __name__ == "__main__":
    main()