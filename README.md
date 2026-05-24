# 📚 Repositori Skripsi — Sistem Pencarian Semantik Skripsi

Sistem pencarian skripsi berbasis **Semantic Information Retrieval** untuk membantu pengguna menemukan judul skripsi yang relevan meskipun menggunakan variasi kata, sinonim, atau query natural language.

Penelitian ini membandingkan beberapa pendekatan retrieval dan query expansion pada repository skripsi universitas.

---

## ✨ Fitur Utama

- 🔎 Pencarian judul skripsi berbasis semantic search
- 🏫 Filter berdasarkan Fakultas
- 🎓 Filter berdasarkan Program Studi
- 🧠 Query Expansion otomatis
- 📊 Perbandingan model retrieval
- 📝 Highlight keyword dan hasil ekspansi
- 📈 Menampilkan skor relevansi dokumen

---

## Model yang Dibandingkan

Sistem mendukung evaluasi beberapa konfigurasi retrieval:

| Model | Deskripsi |
|-------|------------|
| BM25 | Baseline lexical retrieval |
| BM25 + QE | BM25 dengan Query Expansion |
| Dense Retrieval Baseline | Embedding semantic search |
| Dense Retrieval + QE | Semantic retrieval + expansion |
| Finetuned | Dense retrieval model hasil fine-tuning |
| Finetuned + QE | Fine-tuned semantic model + query expansion |

---

## Arsitektur Sistem

```text
User Query
    ↓
Preprocessing Query
    ↓
Query Expansion
    ↓
Retriever Model
(BM25 / Semantic Retrieval / Finetuned)
    ↓
Ranking
    ↓
Top-K Relevant Documents
