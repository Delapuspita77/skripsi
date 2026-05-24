from gensim.models.fasttext import load_facebook_vectors
import pandas as pd
import re

# load full fasttext (sekali aja, jangan di streamlit)
ft = load_facebook_vectors("cc.id.300.bin")

# load dataset
docs = pd.read_csv("data repo final.csv")

def tokenize(text):
    return re.findall(r"[a-z0-9_]+", str(text).lower())

vocab = set()

for text in docs["text_word2vec"].fillna(""):
    vocab.update(tokenize(text))

# ambil hanya kata yang ada di fasttext
filtered = {w: ft[w] for w in vocab if w in ft}

# simpan
import pickle
with open("fasttext_subset.pkl", "wb") as f:
    pickle.dump(filtered, f)

print("Done. vocab size:", len(filtered))