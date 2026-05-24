# Sistem Pencarian Skripsi dengan Semantic Search dan Query Expansion pada Repository UPN Jatim

<img width="1916" height="904" alt="Screenshot 2026-04-28 213518" src="https://github.com/user-attachments/assets/cf40ef9b-3045-4fce-b1f1-858723397d71" />

Sistem pencarian skripsi berbasis **Semantic Information Retrieval** untuk membantu pengguna menemukan judul skripsi yang relevan meskipun menggunakan variasi kata, sinonim, maupun query natural language.

Penelitian ini bertujuan untuk membangun sistem pencarian skripsi berbasis semantic search yang mampu memahami kesamaan makna antara query pengguna dan judul atau abstrak skripsi meskipun tidak terdapat kecocokan kata secara eksplisit pada repository UPN “Veteran” Jawa Timur.

---

## Fitur Utama

- Pencarian judul skripsi berbasis semantic search
- Filter berdasarkan Fakultas
- Filter berdasarkan Program Studi
- Query Expansion interaktif
- Perbandingan model retrieval
- Highlight keyword dan hasil ekspansi
- Menampilkan skor relevansi dokumen
- Pagination di setiap halaman

---

## Model yang Dibandingkan

| Model | Deskripsi |
|-------|------------|
| BM25 + QE | Baseline lexical retrieval |
| Baseline | Semantic Retrieval baseline dengan IndoSBERT |
| Baseline + QE | Semantic Retrieval baseline IndoSBERT + Query Expansion |
| Finetuned | Model semantic retrieval IndoSBERT hasil fine-tuning |
| Finetuned + QE | Fine-tuned IndoSBERT + Query Expansion |

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
(BM25 / Baseline / Baseline+QE / Finetuned / Finetuned+QE)
    ↓
Ranking
    ↓
Top-K Relevant Documents
```
---

## Dataset

Dataset menggunakan metadata repository skripsi universitas yang mencakup:

- Judul Skripsi
- Abstrak 
- URL Repository
- Fakultas
- Program Studi

---

## Struktur Folder

```text
repository-skripsi/
│
├── README.md
│
├── app/
│   ├── app_streamlit.py
│   ├── retrieval.py
│
├── data/
│   ├── raw/
│   │   └── dataset_repo_upn.xlsx
│   │
│   └── qrels/
│       ├── pool_for_judging_longquery.csv
│       ├── pool_for_judging_new.csv
│       └── pool_qrels_pengaruh15.csv
│
├── notebooks/
│   │
│   ├── preprocessing/
│   │   └── Preprocess_Data_Repo_UPN.ipynb
│   │
│   ├── model_training/
│   │   └── create_model_semantic_search_indosbert.ipynb
│   │
│   ├── experiments/
│   │   ├── semantic_qe_final.ipynb
│   │   └── pengaruh_qe.ipynb
│
└── requirements.txt
```

---

## Tampilan Sistem

Fitur utama antarmuka sistem:

- Input Query
- Filter Fakultas & Program Studi
- Checkbox Query Expansion
- Pemilihan Model Retrieval
- Ranking Hasil Pencarian
- Highlight Kata Kunci
- Score Relevansi

---

## Author

**Dela Puspita Lasminingrum**  
Program Studi Informatika  
UPN "Veteran" Jawa Timur

---
