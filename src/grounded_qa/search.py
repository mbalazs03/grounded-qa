import os
import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

load_dotenv()
model = SentenceTransformer("intfloat/multilingual-e5-small")

def search(conn, question: str, k: int = 5):
    q_vec = model.encode("query: " + question, normalize_embeddings=True)
    return conn.execute(
        """
        SELECT chunk_id, passage_id, title,
            1 - (embedding <=> %(q)s) AS score
        FROM chunks
        ORDER BY embedding <=> %(q)s
        LIMIT %(k)s
        """,
        {"q": q_vec, "k": k},
    ).fetchall()

tests = [
    ("In what country is Normandy located?", "p00000"),
    ("Who gave their name to Normandy in the 1000s and 1100s?", "p00000"),
    ("Ki adta a nevét Normandiának?", "p00000"),
]

with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
    register_vector(conn)
    for question, gold in tests:
        print(f"\n KÉRDÉS: {question} (helyes: {gold})")
        for rank, (chunk_id, passsage_id, title, score) in enumerate(search(conn, question), start=1):
            mark = " <-- HELYES" if passsage_id == gold else ""
            print(f" {rank}. {score:.3f} {chunk_id} {title}{mark}")

    # keresés viszgálata indexeléshez: 

    # q_vec = model.encode("query: In what country is Normandy located?", normalize_embeddings=True)
    # plan = conn.execute(
    #     """
    #     EXPLAIN ANALYZE 
    #     SELECT chunk_id 
    #     FROM chunks 
    #     ORDER BY embedding <=> %s 
    #     LIMIT 5;""",
    #     (q_vec,),
    # ).fetchall()
    # for line in plan:
    #     print(line[0])

    # nem kell indexelsé: 
    #     Sort Method: top-N heapsort  Memory: 25kB
    # ->  Seq Scan on chunks  (cost=0.00..153.17 rows=1214 width=18) (actual time=0.039..2.465 rows=1214 loops=1)
    # Planning Time: 0.054 ms
    # Execution Time: 2.670 ms