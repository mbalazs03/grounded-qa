from sentence_transformers import SentenceTransformer

model = SentenceTransformer("intfloat/multilingual-e5-small")

print(f"Max input token: {model.max_seq_length}")
print(f"Vektor hossza: {model.get_embedding_dimension()}")

text = "The Normans (Norman: Nourmands; French: Normands) gave their name to Normandy."
tokens = model.tokenizer.tokenize(text)
print(f"\nSzavak: {len(text.split())}, tokenek: {len(tokens)}")
print(tokens)