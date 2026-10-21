import os
import mlflow
import pandas as pd
import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from tqdm import tqdm

from rag import ask, K
from metrics import exact_match, f1

load_dotenv()
N_PER_GROUP = 150
SEED = 42

questions = pd.read_parquet("data/questions.parquet")
sample = (  
    questions.groupby("unanswerable")
    .sample(n=N_PER_GROUP, random_state=SEED)
    .reset_index(drop=True)
)


rows = []
with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
    register_vector(conn)
    ask(conn, "warm-up")
    for q in tqdm(sample.itertuples(), total=len(sample)):
        r = ask(conn, q.question)
        golds = list(q.answers["text"])
        answered = r["answer"] is not None and len(golds) > 0
        rows.append({
            "id": q.id,
            "question": q.question,
            "unanswerable": q.unanswerable,
            "gold": golds,
            "answer": r["answer"],
            "abstained": r["abstained"],
            "em": exact_match(r["answer"], golds) if answered else False,
            "f1": f1(r["answer"], golds) if answered else 0.0,
            "top_score": r["top_score"],
            "seconds": r["seconds"],
            "gold_retrieved": q.passage_id in [c.split("_")[0] for c in r["retrieved"]],
            "raw": r["raw"],
        })

results = pd.DataFrame(rows)
results.to_parquet("data/rag_results.parquet", index=False)


ans = results[~results["unanswerable"]]
una = results[results["unanswerable"]]
correct = (ans["em"].sum() + una["abstained"].sum()) / len(results)

metrics = {
    "answerable_em": ans["em"].mean(),
    "answerable_f1": ans["f1"].mean(),
    "answerable_false_abstain": ans["abstained"].mean(),
    "unanswerable_abstain": una["abstained"].mean(),
    "overall_score": correct,
    "avg_seconds": results["seconds"].mean(),
}
for name, value in metrics.items():
    print(f"{name:28s} {value:.3f}")


mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("rag")
with mlflow.start_run(run_name="baseline"):
    mlflow.log_params({
        "llm_model": os.environ["LLM_MODEL"],
        "k": K,
        "n_per_group": N_PER_GROUP,
        "seed": SEED,
        "abstention": "llm only",
    })
    mlflow.log_metrics(metrics)
    mlflow.log_artifact("data/rag_results.parquet")