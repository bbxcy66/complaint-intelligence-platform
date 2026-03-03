import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # project root (new_complaint_ai/)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json

import numpy as np
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import faiss
from sentence_transformers import SentenceTransformer


st.set_page_config(page_title="Complaint Intelligence", page_icon="🧠", layout="wide")


# -------------------------
# Helpers
# -------------------------
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@st.cache_resource
def load_classifier(model_dir: str):
    model_dir = Path(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    device = get_device()
    model.to(device)
    model.eval()
    return tokenizer, model, device


def predict_product(text: str, tokenizer, model, device, max_length=128):
    enc = tokenizer(
        [text],
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )
    # DistilBERT doesn't need token_type_ids
    enc.pop("token_type_ids", None)
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred_id = int(np.argmax(probs))
    pred_label = model.config.id2label[str(pred_id)] if isinstance(model.config.id2label, dict) and str(pred_id) in model.config.id2label else model.config.id2label[pred_id]
    confidence = float(probs[pred_id])

    return pred_label, confidence, probs


@st.cache_resource
def load_rag(index_path: str, meta_path: str):
    index = faiss.read_index(index_path)

    meta = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            meta.append(json.loads(line))

    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return index, meta, embedder


def rag_search(query: str, index, meta, embedder, top_k: int = 5):
    q = embedder.encode([query], normalize_embeddings=True).astype(np.float32)
    scores, idxs = index.search(q, top_k)
    scores = scores[0].tolist()
    idxs = idxs[0].tolist()

    results = []
    for score, i in zip(scores, idxs):
        if i < 0 or i >= len(meta):
            continue
        row = meta[i]
        row_out = {
            "score": float(score),
            "product": row.get("product"),
            "date_received": row.get("date_received"),
            "split": row.get("split"),
            "text": row.get("text"),
        }
        results.append(row_out)
    return results


# -------------------------
# UI
# -------------------------
st.title("Complaint Intelligence Platform")
st.caption("V1: product triage + semantic retrieval (RAG-style) for CFPB complaints")

model_dir = "outputs/models/distilbert_v1"
rag_index_path = "outputs/rag/complaints.faiss"
rag_meta_path = "outputs/rag/meta.jsonl"

# Sidebar
st.sidebar.header("System Status")

model_ok = Path(model_dir).exists()
rag_ok = Path(rag_index_path).exists() and Path(rag_meta_path).exists()

st.sidebar.write(f"Classifier model: {'✅' if model_ok else '❌'} ({model_dir})")
st.sidebar.write(f"RAG index: {'✅' if rag_ok else '❌'} ({rag_index_path})")

st.sidebar.divider()
st.sidebar.write("Scope products:")
st.sidebar.write("- Money transfer / virtual currency / money service")
st.sidebar.write("- Checking or savings account")
st.sidebar.write("- Credit card")

tab1, tab2, tab3 = st.tabs(["Single Complaint Triage", "Similar Complaints", "Analytics Dashboard"])

with tab1:
    st.subheader("Single Complaint Triage")
    text = st.text_area("Paste a complaint narrative", height=220)

    if not model_ok:
        st.error("Model not found. Make sure outputs/models/distilbert_v1 exists.")
    else:
        tokenizer, model, device = load_classifier(model_dir)

        if st.button("Predict", type="primary"):
            if text.strip() == "":
                st.warning("Paste some text first.")
            else:
                pred_label, conf, probs = predict_product(text, tokenizer, model, device)

                c1, c2 = st.columns(2)
                with c1:
                    st.metric("Predicted product", pred_label)
                with c2:
                    st.metric("Confidence", f"{conf:.2%}")

                st.write("Class probabilities")
                # display as readable list
                # id2label keys might be str/int depending on HF version
                id2label = model.config.id2label
                rows = []
                for i, p in enumerate(probs):
                    label = id2label[str(i)] if isinstance(id2label, dict) and str(i) in id2label else id2label[i]
                    rows.append((label, float(p)))
                rows = sorted(rows, key=lambda x: x[1], reverse=True)
                st.write(rows)

with tab2:
    st.subheader("Similar Complaints Search (RAG-style retrieval)")
    query = st.text_area("Enter a narrative to find similar past complaints", height=160)
    top_k = st.slider("Top K", 3, 15, 5)

    if not rag_ok:
        st.error("RAG files not found. Make sure outputs/rag/complaints.faiss and outputs/rag/meta.jsonl exist.")
    else:
        index, meta, embedder = load_rag(rag_index_path, rag_meta_path)

        if st.button("Search", type="primary"):
            if query.strip() == "":
                st.warning("Enter some text first.")
            else:
                results = rag_search(query, index, meta, embedder, top_k=top_k)

                for r in results:
                    with st.expander(f"{r['product']} | score={r['score']:.3f} | date={r['date_received']}"):
                        st.write(r["text"])

with tab3:
    st.subheader("Analytics Dashboard")

    parquet_path = "src/data/processed/complaints_2024_3products.parquet"
    if not Path(parquet_path).exists():
        st.error(f"Processed parquet not found: {parquet_path}")
    else:
        import altair as alt
        from src.analytics.dashboard_data import load_dashboard_df, monthly_counts

        df = load_dashboard_df(parquet_path)

        # KPIs
        c1, c2, c3 = st.columns(3)
        c1.metric("Total complaints", f"{len(df):,}")
        c2.metric("Date min", str(df["date_received"].min().date()))
        c3.metric("Date max", str(df["date_received"].max().date()))

        st.write("Complaints by product")
        prod_counts = df["product"].value_counts().reset_index()
        prod_counts.columns = ["product", "count"]
        st.dataframe(prod_counts, use_container_width=True)

        st.divider()
        st.write("Monthly complaint volume (2024–present)")

        mdf = monthly_counts(df)

        chart = (
            alt.Chart(mdf)
            .mark_line()
            .encode(
                x=alt.X("month:T", title="Month"),
                y=alt.Y("count:Q", title="Complaints"),
                color=alt.Color("product:N", title="Product"),
                tooltip=["month:T", "product:N", "count:Q"],
            )
            .properties(height=360)
        )
        st.altair_chart(chart, use_container_width=True)

        st.divider()
        st.write("Quick keyword peek (top tokens by product)")
        st.caption("This is a lightweight sanity check, not a full NLP topic model.")

        # Lightweight token counts
        import re
        from collections import Counter

        stop = set([
            "the","and","to","of","i","a","in","for","it","is","was","on","that","with","my","this","you","they",
            "are","be","have","has","had","as","at","but","or","not","we","me","so","an","if","by","from",
            "their","them","your","our"
        ])

        def top_words(texts, k=15):
            cnt = Counter()
            for t in texts:
                words = re.findall(r"[a-zA-Z']{3,}", str(t).lower())
                words = [w for w in words if w not in stop]
                cnt.update(words)
            return cnt.most_common(k)

        colA, colB, colC = st.columns(3)
        products = df["product"].unique().tolist()

        # Show per product
        for col, prod in zip([colA, colB, colC], products):
            with col:
                st.write(prod)
                topk = top_words(df.loc[df["product"] == prod, "text"].sample(n=min(5000, (df["product"] == prod).sum()), random_state=42))
                st.write(topk) 