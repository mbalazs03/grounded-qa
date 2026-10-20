import os
import time
import pandas as pd
import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer
import mlflow

load_dotenv()
K_VALUES = [1, 3, 5, 10]
MAX_K = max(K_VALUES)
QUERY_PREFIX = "query: "

questions = pd.read_parquet("data/questions.parquet")
model = SentenceTransformer("intfloat/multilingual-e5-small")

q_vecs = model.encode(
    [QUERY_PREFIX + q for q in questions["question"]],
    batch_size=64, normalize_embeddings=True, show_progress_bar=True,
)

ranks = []
start = time.perf_counter()
with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
    register_vector(conn)
    for vec, gold in zip(q_vecs, questions["passage_id"]):
        rows = conn.execute(
            """
            SELECT passage_id
            FROM chunks
            ORDER BY embedding <=> %s
            LIMIT %s;
            """,
            (vec, MAX_K),
        ).fetchall()
        found = [r[0] for r in rows]
        ranks.append(found.index(gold) + 1 if gold in found else None)
# print(f"KERESÉS: {time.perf_counter() - start:.1f} mp")

search_seconds = time.perf_counter() - start
print(f"Keresés: {search_seconds:.1f} mp")

questions["rank"] = ranks

groups = [
    ("answerable", questions[~questions["unanswerable"]]),
    ("unanswerable", questions[questions["unanswerable"]]),
]

# for name, subset in groups:
#     print(f"\n {name} ({len(subset)} kérdés)")
#     for k in K_VALUES:
#         recall = (subset["rank"] <= k).mean()
#         print(f" Recall@{k}: {recall:.1%}")

# KERESÉS: 22.7 mp
# 
#  answerable (5928 kérdés)
#  Recall@1: 78.9%
#  Recall@3: 91.8%
#  Recall@5: 95.1% --> jo
#  Recall@10: 97.5%
# 
#  unanswerable (5945 kérdés)
#  Recall@1: 72.2%
#  Recall@3: 86.9%
#  Recall@5: 91.3%
#  Recall@10: 95.4%

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("retrieval")

with mlflow.start_run(run_name="e5-small baseline"):
    mlflow.log_params({
        "embedding_model": "intfloat/multilingual-e5-small",
        "max_words": 380,
        "window": 300,
        "overlap": 50,
        "n_questions": len(questions),
        "query_prefix": QUERY_PREFIX,
    })
    mlflow.log_metric("search_seconds", search_seconds)

    for name, subset in groups:
        print(f"\n {name} ({len(subset)} kérdés)")
        for k in K_VALUES:
            recall = (subset["rank"] <= k).mean()
            print(f" Recall@{k}: {recall:.1%}")
            mlflow.log_metric(f"recall_at_{k}_{name}", recall)

    