from datasets import load_dataset
import pandas as pd

ds = load_dataset("rajpurkar/squad_v2", split="validation")
print(ds)
print(ds[0])

df = ds.to_pandas()
print(len(df))
print(df["title"].nunique())
print(f"Egyedi bekezdések: {df['context'].nunique()}")

ratio = df["answers"].apply(lambda a: len(a["text"]) == 0).mean()
print(f"Megválaszolhatatlan: {ratio:.1%}")


df["unanswerable"] = df["answers"].apply(lambda a: len(a["text"]) == 0)

ctx = df.loc[0, "context"]
same = df[df["context"] == ctx]

print("\n=== BEKEZDÉS ===")
print(ctx)

print("\n=== KÉRDÉSEK ===")
for _, row in same.iterrows():
    label = "NINCS VÁLASZ" if row["unanswerable"] else f"VÁLASZ: {row['answers']['text'][0]}"
    print(f"- {row['question']}\n -> {label}")