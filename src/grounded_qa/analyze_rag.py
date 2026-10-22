import numpy as np
import pandas as pd
import mlflow

results = pd.read_parquet("data/rag_results.parquet")
ans_mask = ~results["unanswerable"]


print("Legjobb találat pontszáma (False = megválaszolható, True = megválaszolhatatlan):")
print(results.groupby("unanswerable")["top_score"].describe()[["mean", "25%", "50%", "75%"]].round(3))


def evaluate(threshold: float) -> dict:
    abstain = results["abstained"] | (results["top_score"] < threshold)
    em = results["em"] & ~abstain
    return{
        "threshold": threshold,
        "answerable_em": em[ans_mask].mean(),
        "answerable_false_abstain": abstain[ans_mask].mean(),
        "unanswerable_abstain": abstain[~ans_mask].mean(),
        "overall_score": (em[ans_mask].sum() + abstain[~ans_mask].sum()) / len(results),
    }


sweep = pd.DataFrame([evaluate(t) for t in np.arange(0.78, 0.93, 0.01)])
print("\n küszöb végigpróbálása:")
print(sweep.round(3).to_string(index=False))

best = sweep.loc[sweep["overall_score"].idxmax()]
print(f"\nLegjobb küszöb: {best['threshold']:.2f}, overall_score: {best['overall_score']:.3f}")


mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("rag")
with mlflow.start_run(run_name=f"threshold {best['threshold']:.2f}"):
    mlflow.log_params({
        "abstention": "threshold + llm",
        "threshold": round(best["threshold"], 2),
        "tuned_on": "eval sample (optimistic)",
    })
    mlflow.log_metrics({k: float(best[k]) for k in ["answerable_em", "answerable_false_abstain", "unanswerable_abstain", "overall_score"]})


pd.set_option("display.max_colwidth", 70)

print("\nMegválaszolhatatlan, mégis válaszolt (5 példa):")
wrong = results[results["unanswerable"] & ~results["abstained"]]
print(wrong[["question", "answer", "top_score"]].head(5).to_string())

print("\nMegválaszolható, részben jó: EM = 0, de F1 > 0 (5 példa):")
partial = results[ans_mask & ~results["em"] & (results["f1"] > 0)]
print(partial[["question", "answer", "gold", "f1"]].head(5).to_string())

print("\nMegválaszolható, rossz válasz: benne volt-e a helyes bekezdés az 5 chunkban?")
miss = results[ans_mask & ~results["em"] & ~results["abstained"]]
print(miss["gold_retrieved"].value_counts())
