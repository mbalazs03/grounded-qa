from datasets import load_dataset
from pathlib import Path
import pandas as pd


df = load_dataset("rajpurkar/squad_v2", split="validation").to_pandas()
print(len(df))


passages = df[["title", "context"]].drop_duplicates(subset="context")
print(f"Egyedi bekezdések: {len(passages)}")
# print(passages.head())


passages = passages.reset_index(drop=True)
passages["passage_id"] = [f"p{i:05d}" for i in range(len(passages))]
#print(passages.head())


df["unanswerable"] = df["answers"].apply(lambda a: len(a["text"]) == 0)

questions = df.merge(passages[["context", "passage_id"]], on="context", how="left")
questions = questions[["id", "question", "passage_id", "unanswerable", "answers"]]


print(f"Kérdések merge után: {len(questions)}")
print(f"Azonositó nélküli kérdések: {questions['passage_id'].isna().sum()}")
print(questions.head())


passages["n_words"] = (passages["context"].str.split()).str.len()
print(f"Bekezdések hossza: {passages["n_words"].describe()}")
print(f"380 szónál hosszabb bekezdések: {(passages["n_words"] > 380).sum()}")


WINDOW = 300
OVERLAP = 50
MAX_WORDS = 380

def split_words(text: str) -> list[str]:
    words = text.split()
    if len(words) <= MAX_WORDS:
        return [text]
    step = WINDOW - OVERLAP
    pieces = []
    for start in range(0, len(words), step):
        pieces.append(" ".join(words[start:start + WINDOW]))
        if start + WINDOW >= len(words):
            break
    return pieces


rows = []
for p in passages.itertuples():
    for i, piece in enumerate(split_words(p.context)):
        rows.append({
            "chunk_id": f"{p.passage_id}_c{i}",
            "passage_id": p.passage_id,
            "title": p.title,
            "text": piece,
        })

chunks = pd.DataFrame(rows)
print(f"Chunkok: {len(chunks)}, bekezdések: {len(passages)}")


longest = passages.loc[passages["n_words"].idxmax(), "passage_id"]
for c in chunks[chunks["passage_id"] == longest].itertuples():
    print(f"{c.chunk_id}: {len(c.text.split())} szó")


DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
passages.to_parquet(DATA_DIR / "passages.parquet", index=False)
chunks.to_parquet(DATA_DIR / "chunks.parquet", index=False)
questions.to_parquet(DATA_DIR / "questions.parquet", index=False)
print("Mentve a data/ mappába!")