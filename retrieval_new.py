# ============================================================
# RETRIEVAL MODULE — 6 SKENARIO
# ============================================================
# 1. BM25
# 2. BM25 + QE
# 3. IndoBERT Baseline
# 4. IndoBERT Baseline + QE
# 5. IndoBERT Finetuned
# 6. IndoBERT Finetuned + QE  ← Model Final
# ============================================================

import os
import re
import numpy as np
import pandas as pd
from collections import Counter
import pickle

from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

import faiss
import torch

from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# ===============================
# DEVICE
# ===============================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===============================
# LOAD DATA
# ===============================
DATA_PATH = "data repo final.csv"
docs = pd.read_csv(DATA_PATH)
docs = docs.reset_index(drop=True)
docs["doc_id"] = docs.index

# ===============================
# TEXT COLUMN
# ===============================
TEXT_W2V_COL = "text_word2vec"
docs[TEXT_W2V_COL] = docs.get(TEXT_W2V_COL, "").fillna("")

# ===============================
# TOKENIZER
# ===============================
def simple_tokenize(text):
    text = str(text).lower()
    tokens = re.findall(r"[a-z0-9_]+", text)
    return [t for t in tokens if len(t) > 2]

# ===============================
# DOC FREQ
# ===============================
docs_tokens_w2v = [simple_tokenize(t) for t in docs[TEXT_W2V_COL]]

doc_freq = Counter()
for tokens in docs_tokens_w2v:
    for tok in set(tokens):
        doc_freq[tok] += 1

N_DOCS = len(docs_tokens_w2v)

# ===============================
# BM25
# ===============================
bm25 = BM25Okapi(docs_tokens_w2v)

# ===============================
# LOAD FASTTEXT SUBSET
# ===============================
with open("fasttext_subset.pkl", "rb") as f:
    ft = pickle.load(f)

ft_norm = {k: np.linalg.norm(v) for k, v in ft.items()}

# ===============================
# TRAIN WORD2VEC
# ===============================
w2v_model = Word2Vec(
    sentences=docs_tokens_w2v,
    vector_size=300,
    window=8,
    min_count=5,
    workers=4,
    sg=1
)
w2v_model.train(docs_tokens_w2v, total_examples=len(docs_tokens_w2v), epochs=10)
w2v = w2v_model.wv

# ===============================
# STOPWORDS + STEMMER
# ===============================
factory   = StopWordRemoverFactory()
STOPWORDS = set(factory.get_stop_words())
stemmer   = StemmerFactory().create_stemmer()

def normalize(word):
    return stemmer.stem(word.lower())

# ===============================
# QE VALIDATION
# ===============================
def is_valid_expansion(word, base_tokens, min_df=3, max_df_ratio=0.2):
    w = normalize(word)
    if w in [normalize(bt) for bt in base_tokens]:
        return False
    if w in STOPWORDS:
        return False
    if word.isdigit() or len(word) < 3:
        return False
    if doc_freq.get(word, 0) < min_df:
        return False
    if doc_freq.get(word, 0) / N_DOCS > max_df_ratio:
        return False
    return True

# ===============================
# HYBRID QUERY EXPANSION
# ===============================
def expand_query_hybrid(query, topn_per_term=15, min_sim=0.5, max_expansion=7):
    base_tokens = simple_tokenize(query)
    expansions  = []

    for token in base_tokens:
        candidates = []

        if token in w2v:
            candidates += w2v.most_similar(token, topn=topn_per_term)

        if token in ft:
            vec  = ft[token]
            sims = []
            for wrd, v in ft.items():
                sim = np.dot(vec, v) / (np.linalg.norm(vec) * ft_norm[wrd] + 1e-9)
                sims.append((wrd, sim))
            sims = sorted(sims, key=lambda x: -x[1])[:topn_per_term]
            candidates += sims

        for cand, sim in candidates:
            if sim < min_sim:
                continue
            if not is_valid_expansion(cand, base_tokens):
                continue
            if cand in expansions:
                continue
            expansions.append(cand)
            if len(expansions) >= max_expansion:
                break

        if len(expansions) >= max_expansion:
            break

    return {"token_query": base_tokens, "hasil_expansi": expansions}

# ===============================
# LOAD SEMANTIC MODELS + INDEXES
# ===============================

# Baseline
MODEL_NAME_BASE = "firqaaa/indo-sentence-bert-base"
INDEX_DIR_BASE  = "indexes_base"
model_base      = SentenceTransformer(MODEL_NAME_BASE, device=DEVICE)
index_base      = faiss.read_index(os.path.join(INDEX_DIR_BASE, "semantic_index.faiss"))
doc_ids_base    = np.load(os.path.join(INDEX_DIR_BASE, "doc_ids.npy")).tolist()
vectors_base    = index_base.reconstruct_n(0, index_base.ntotal)

# Finetuned
MODEL_DIR_FT = "indosbert_finetuned_title_abs"
INDEX_DIR_FT = "indexes_finetuned"
model_ft     = SentenceTransformer(MODEL_DIR_FT, device=DEVICE)
index_ft     = faiss.read_index(os.path.join(INDEX_DIR_FT, "semantic_index.faiss"))
doc_ids_ft   = np.load(os.path.join(INDEX_DIR_FT, "doc_ids.npy")).tolist()
vectors_ft   = index_ft.reconstruct_n(0, index_ft.ntotal)

# ===============================
# HELPERS
# ===============================
def _filter_docs(fakultas=None, prodi=None):
    filtered = docs.copy()
    if fakultas and fakultas != "Semua":
        filtered = filtered[filtered["faculty"] == fakultas]
    if prodi and prodi != "Semua":
        filtered = filtered[filtered["department"] == prodi]
    return filtered

def _build_results(ranked_doc_ids, score_map):
    results = []
    for rank, doc_id in enumerate(ranked_doc_ids, start=1):
        row = docs.iloc[doc_id]
        results.append({
            "rank":   rank,
            "title":  row.get("title", "-"),
            "url":    row.get("url", "#"),
            "score":  float(score_map.get(doc_id, 0.0)),
            "doc_id": doc_id,
        })
    return results

def _semantic_search(q_emb, filtered_ids, vectors_all, doc_ids_all, top_k=500):
    pos_map    = {d: i for i, d in enumerate(doc_ids_all)}
    valid_ids  = [d for d in filtered_ids if d in pos_map]
    if not valid_ids:
        return [], {}

    filt_vecs = np.array([vectors_all[pos_map[d]] for d in valid_ids])
    sims = np.dot(filt_vecs, q_emb.T).squeeze()

    if sims.ndim == 0:
        sims = np.array([float(sims)])

    # ---- STEP 1: pasangan semua
    pairs = [(valid_ids[i], float(sims[i])) for i in range(len(sims))]

    # ---- STEP 2: sort descending
    pairs = sorted(pairs, key=lambda x: -x[1])

    # ---- STEP 3: ambil top_k dulu (biar cepat & stabil)
    pairs = pairs[:top_k]

    scores = np.array([p[1] for p in pairs])

    # ---- STEP 4: adaptive threshold (mean + sedikit bias)
    thresh = scores.mean()  # bisa juga: mean + 0.1 * std

    pairs = [p for p in pairs if p[1] >= thresh]

    ranked_ids = [d for d, _ in pairs]
    score_map  = {d: s for d, s in pairs}

    return ranked_ids, score_map


# # ============================================================
# # 1. BM25
# # ============================================================
# def search_bm25(query, fakultas=None, prodi=None):
#     tokens   = simple_tokenize(query)
#     filtered = _filter_docs(fakultas, prodi)
#     fids     = set(filtered["doc_id"].tolist())

#     all_scores = bm25.get_scores(tokens)
#     pairs = [(doc_id, all_scores[doc_id]) for doc_id in fids if all_scores[doc_id] >= 0.1]
#     pairs = sorted(pairs, key=lambda x: -x[1])

#     ranked_ids = [d for d, _ in pairs]
#     score_map  = {d: s for d, s in pairs}
#     return _build_results(ranked_ids, score_map), tokens, []


# ============================================================
# 2. BM25 + QE
# ============================================================
def search_bm25_qe(query, selected_expansions=None, fakultas=None, prodi=None):
    qe         = expand_query_hybrid(query)
    tokens     = qe["token_query"]
    expansions = qe["hasil_expansi"] if selected_expansions is None else selected_expansions

    terms    = tokens + expansions if expansions else tokens
    filtered = _filter_docs(fakultas, prodi)
    fids     = set(filtered["doc_id"].tolist())

    all_scores = bm25.get_scores(terms)
    pairs = [(doc_id, all_scores[doc_id]) for doc_id in fids if all_scores[doc_id] >= 0.1]
    pairs = sorted(pairs, key=lambda x: -x[1])

    ranked_ids = [d for d, _ in pairs]
    score_map  = {d: s for d, s in pairs}
    return _build_results(ranked_ids, score_map), tokens, expansions


# ============================================================
# 3. IndoBERT Baseline
# ============================================================
def search_base(query, fakultas=None, prodi=None):
    tokens   = simple_tokenize(query)
    filtered = _filter_docs(fakultas, prodi)
    fids     = filtered["doc_id"].tolist()

    q_emb = model_base.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    ranked_ids, score_map = _semantic_search(q_emb, fids, vectors_base, doc_ids_base)
    return _build_results(ranked_ids, score_map), tokens, []


# ============================================================
# 4. IndoBERT Baseline + QE
# ============================================================
def search_base_qe(query, selected_expansions=None, fakultas=None, prodi=None):
    qe         = expand_query_hybrid(query)
    tokens     = qe["token_query"]
    expansions = qe["hasil_expansi"] if selected_expansions is None else selected_expansions

    expanded = " ".join(tokens + expansions) if expansions else query
    filtered = _filter_docs(fakultas, prodi)
    fids     = filtered["doc_id"].tolist()

    q_emb = model_base.encode([expanded], convert_to_numpy=True, normalize_embeddings=True)
    ranked_ids, score_map = _semantic_search(q_emb, fids, vectors_base, doc_ids_base)
    return _build_results(ranked_ids, score_map), tokens, expansions


# ============================================================
# 5. IndoBERT Finetuned
# ============================================================
def search_ft(query, fakultas=None, prodi=None):
    tokens   = simple_tokenize(query)
    filtered = _filter_docs(fakultas, prodi)
    fids     = filtered["doc_id"].tolist()

    q_emb = model_ft.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    ranked_ids, score_map = _semantic_search(q_emb, fids, vectors_ft, doc_ids_ft)
    return _build_results(ranked_ids, score_map), tokens, []


# ============================================================
# 6. IndoBERT Finetuned + QE  ← MODEL FINAL
# ============================================================
def search(query, selected_expansions=None, fakultas=None, prodi=None):
    qe         = expand_query_hybrid(query)
    tokens     = qe["token_query"]
    expansions = qe["hasil_expansi"] if selected_expansions is None else selected_expansions

    expanded = " ".join(tokens + expansions) if expansions else query
    filtered = _filter_docs(fakultas, prodi)
    fids     = filtered["doc_id"].tolist()

    q_emb = model_ft.encode([expanded], convert_to_numpy=True, normalize_embeddings=True)
    ranked_ids, score_map = _semantic_search(q_emb, fids, vectors_ft, doc_ids_ft)
    return _build_results(ranked_ids, score_map), tokens, expansions
