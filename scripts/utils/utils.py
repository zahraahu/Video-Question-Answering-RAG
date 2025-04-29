import faiss
import json
import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

# === JSON UTILS ===
def save_json(obj, path):
    """
    Save a Python object as a JSON file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"Saved JSON to {path}")

def load_json(path):
    """
    Load a JSON file into a Python object.
    """
    with open(path) as f:
        return json.load(f)

# === FAISS UTILS ===
def build_faiss_index(embeddings, dim, index_path):
    """
    Build and save a FAISS index from embeddings.
    """
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, str(index_path))
    print(f"Saved FAISS index to {index_path}")

def faiss_retrieve(query_embedding, index_path, top_k=1):
    """
    Retrieve the top-k nearest neighbors using FAISS flat index.
    Returns the top result only.
    """
    index = faiss.read_index(str(index_path))
    D, I = index.search(query_embedding.reshape(1, -1), top_k)
    
    # Return the top results
    return 1 - D[0][0], I[0][0] 

# === BM25 UTILS ===
def bm25_retrieve(query, documents):
    """
    Retrieve the top-k most similar documents using BM25 ranking.
    Returns only the top result.
    """
    tokenized_documents = [doc.split() for doc in documents]
    bm25 = BM25Okapi(tokenized_documents)

    query_tokens = query.split()
    scores = bm25.get_scores(query_tokens)

    # Get the top result
    top_index = np.argmax(scores)  # Index of the top document
    return top_index, scores  # Return the top document and its score

# === TF-IDF Utils ===
def tfidf_retrieve(query, documents):
    """
    Retrieve the top-k most similar documents based on TF-IDF cosine similarity.
    Returns only the top result.
    """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    query_vector = vectorizer.transform([query])

    cosine_similarities = np.dot(query_vector, tfidf_matrix.T).toarray().flatten()

    # Get the top result
    top_index = np.argmax(cosine_similarities)  # Index of the top document
    return top_index, cosine_similarities[top_index]  # Return the top document and its similarity score
