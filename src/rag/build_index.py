from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss


def main():
    parquet_path = Path("src/data/processed/complaints_2024_3products.parquet")
    out_dir = Path("outputs/rag")
    out_dir.mkdir(parents=True, exist_ok=True)

    index_path = out_dir / "complaints.faiss"
    meta_path = out_dir / "meta.jsonl"

    # Load processed dataset (use train split to avoid leakage if you care; for demo, you can use all)
    df = pd.read_parquet(parquet_path)

    # For a first build, keep it manageable
    # Retrieval works great even with 30k–80k items.
    df = df.sample(n=min(80000, len(df)), random_state=42).reset_index(drop=True)

    texts = df["text"].astype(str).tolist()

    # A solid, lightweight embedding model
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embedder = SentenceTransformer(model_name)

    # Encode
    # normalize_embeddings=True makes cosine similarity easy via dot product
    embeddings = embedder.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    dim = embeddings.shape[1]
    print("embeddings:", embeddings.shape)

    # Build FAISS index for cosine similarity (dot product on normalized vectors)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, str(index_path))

    # Save metadata aligned with FAISS row order
    # Keep whatever you want to display in Streamlit
    with meta_path.open("w", encoding="utf-8") as f:
        for i in range(len(df)):
            row = {
                "i": i,
                "product": df.loc[i, "product"],
                "date_received": str(df.loc[i, "date_received"]) if "date_received" in df.columns else None,
                "text": df.loc[i, "text"][:1200],  # truncate to keep meta smaller
                "split": df.loc[i, "split"] if "split" in df.columns else None,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("Saved index:", index_path)
    print("Saved meta:", meta_path)


if __name__ == "__main__":
    main()