import os 
import psycopg
from dotenv import load_dotenv

load_dotenv()

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

    cols = conn.execute("""
        SELECT column_name, data_type, udt_name FROM information_schema.columns
        WHERE table_name = 'chunks'
        ORDER BY ordinal_position;
    """).fetchall()
    for c in cols:
        print(c)