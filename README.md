**Complaint Intelligence Platform**

An NLP-powered complaint triage and retrieval system built on CFPB complaint data (2024–present).

Features:

Fine-tuned DistilBERT classifier for product prediction

RAG-style semantic search using FAISS + sentence-transformers

Interactive Streamlit dashboard

Time-series complaint analytics

Products Covered:

Checking or savings account

Credit card

Money transfer / virtual currency / money service

Tech Stack:

Python

PyTorch

HuggingFace Transformers

Sentence-Transformers

FAISS

Streamlit

Pandas + Altair

Run Locally:
pip install -r requirements.txt
streamlit run app/app.py
