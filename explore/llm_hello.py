import os
import time
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url=os.environ["LLM_BASE_URL"],
    api_key=os.environ["LLM_API_KEY"],
)

start = time.perf_counter()
resp = client.chat.completions.create(
    model=os.environ["LLM_MODEL"],
    messages=[
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": "n one sentence: what is retrieval-augmented generation?"}
    ],
    temperature=0,
    seed=42,
    reasoning_effort="none"
)
elapsed = time.perf_counter() - start

print(resp.choices[0].message.content)
print(resp.choices[0].message) #alapbol van thinking
print(f"\nIdő: {elapsed:.1f} mp")
print(resp.usage)