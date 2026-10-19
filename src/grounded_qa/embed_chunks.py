import time
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

chunks = pd.read_parquet("data/chunks.parquet")
model = SentenceTransformer("intfloat/multilingual-e5-small")

texts = ["passage: " + t for t in chunks["text"]]

chunks["n_tokens"] = [len(model.tokenizer(t)["input_ids"]) for t in texts]
print(chunks["n_tokens"].describe())
print(f"512 tokennél hosszabb: {(chunks['n_tokens'] > 512).sum()}")

# az 542 tokenes chunk vizsgálata
long = chunks[chunks["n_tokens"] > 512]
print(long[["chunk_id", "n_tokens"]])
print(long.iloc[0]["text"])

q = pd.read_parquet("data/questions.parquet")
for r in q[q["passage_id"] == "p00526"].itertuples():
    print(r.question, "->", list(r.answers["text"])[:1])

start = time.perf_counter()
vecs = model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=True)
print(f"Idő: {time.perf_counter() - start:.1f} mp, alak: {vecs.shape}")

np.save("data/chunk_embeddings.npy", vecs)
print("Mentve!")