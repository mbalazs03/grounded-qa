import os
import time
import psycopg
from dotenv import load_dotenv
from openai import OpenAI
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

load_dotenv()
K = 5

embedder = SentenceTransformer("intfloat/multilingual-e5-small")
llm = OpenAI(base_url=os.environ["LLM_BASE_URL"], api_key=os.environ["LLM_API_KEY"])

SYSTEM_PROMPT = """You answer questions using ONLY the sources provided.
Rules:
1. If the sources do not contain the answer, reply with exactly: NO_ANSWER
2. Otherwise reply in exactly this format:
ANSWER: <the shortest possible answer, copied from the sources>
SOURCE: <the id of the source you used>
3. Never use outside knowledge."""


def retrieve(conn, question: str, k: int = K):
    q_vec = embedder.encode("query: " + question, normalize_embeddings=True)
    return conn.execute(
        """
        SELECT chunk_id, title, text, 1 - (embedding <=> %(q)s) AS score
        FROM chunks
        ORDER BY embedding <=> %(q)s
        LIMIT %(k)s
        """,
        {"q": q_vec,"k": k},
    ).fetchall()


def build_user_message(question: str, chunks) -> str:
    sources = "\n\n".join(f"[{cid}] ({title})\n{text}" for cid, title, text, _ in chunks)
    return f"Sources: \n{sources}\n\nQuestuion: {question}"


def parse(reply: str):
    reply = reply.strip()
    if reply.startswith("NO_ANSWER"):
        return None, None
    answer = source = None
    for line in reply.splitlines():
        if line.startswith("ANSWER:"):
            answer = line.removeprefix("ANSWER:").strip()
        elif line.startswith("SOURCE:"):
            source = line.removeprefix("SOURCE:").strip()
    return answer, source


def ask(conn, question: str) -> dict:
    chunks = retrieve(conn, question)
    start = time.perf_counter()
    resp = llm.chat.completions.create(
        model=os.environ["LLM_MODEL"],
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_message(question, chunks)},
        ],
        temperature=0,
        seed=42,
        reasoning_effort="none",
    )
    reply = resp.choices[0].message.content
    answer, source = parse(reply)
    return {
        "answer": answer,
        "source": source,
        "abstained": answer is None,
        "raw": reply,
        "seconds": time.perf_counter() - start,
        "retrieved": [c[0] for c in chunks],
        "top_score": chunks[0][3]
    }


if __name__ == "__main__":
    tests = [
        "In what country is Normandy located?",
        "Who gave their name to Normandy in the 1000s and 1100s?",
        "Ki adta a nevét Normandiának?",
    ]
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        register_vector(conn)
        for q in tests:
            r = ask(conn, q)
            print(f"\nKÉRDÉS: {q}")
            print(f" válasz: {r['answer']} forrés: {r['source']} tartózkodott: {r['abstained']}")
            print(f" idő: {r['seconds']:.1f} mp")