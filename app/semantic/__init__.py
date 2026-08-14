"""Semantic retrieval: embeddings and vector similarity.

An independent retrieval path, additive to the lexical one. Nothing here imports
FastAPI, and BM25 does not depend on any of it — the two paths are usable
separately, which is what lets a later phase reason about combining them.
"""
