from sentence_transformers import SentenceTransformer

model = SentenceTransformer("intfloat/multilingual-e5-small")

passage = ("passage: The Normans gave their name to Normandy, "
           "a region in France, in the 10th and 11th centuries.")

queries = [
    "query: Who gave their name to Normandy?",
    "query: Ki adta a nevét Normandiának?",
    "query: Who gave their name to Normandy in the 1000s and 1100s?",
    "query: What is the best pizza topping?",
]

p_vec = model.encode(passage, normalize_embeddings=True)
q_vecs = model.encode(queries, normalize_embeddings=True)

scores = q_vecs @ p_vec
for q, s in zip(queries, scores):
    print(f"{s:.3f} {q}")