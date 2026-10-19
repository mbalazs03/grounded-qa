import os
import psycopg
from dotenv import load_dotenv

load_dotenv()

url = os.environ["DATABASE_URL"]
print("Kapcsolódás:", url)

with psycopg.connect(url) as conn:
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    row = conn.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector';").fetchone()
    print("pgvector verzió:", row[0])