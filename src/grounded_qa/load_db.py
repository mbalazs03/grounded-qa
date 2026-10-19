import os 
import psycopg
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from pgvector.psycopg import register_vector

load_dotenv()

chunks = pd.read_parquet("data/chunks.parquet")
vecs = np.load("data/chunk_embeddings.npy")
assert len(chunks) == len(vecs), "A chunkok és vektorok száma nem egyezik!"

with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id TEXT PRIMARY KEY,
            passage_id TEXT NOT NULL,
            title TEXT,
            text TEXT NOT NULL,
            embedding vector(384) NOT NULL
        );
    """)
    print("Tábla kész")

    # cols = conn.execute("""
    #     SELECT column_name, data_type, udt_name FROM information_schema.columns
    #     WHERE table_name = 'chunks'
    #     ORDER BY ordinal_position;
    # """).fetchall()
    # for c in cols:
    #     print(c)

    register_vector(conn)
    conn.execute("TRUNCATE chunks;")

    rows = [
        (r.chunk_id, r.passage_id, r.title, r.text, v)
        for r, v in zip(chunks.itertuples(), vecs)
    ]

    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO chunks (chunk_id, passage_id, title, text, embedding) "
            "VALUES (%s, %s, %s, %s, %s)",
            rows,
        )

    n = conn.execute("SELECT count(*) FROM chunks;").fetchone()[0]
    print(f"Betöltve: {n} chunk")