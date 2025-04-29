import argparse
import numpy as np
from utils.utils import faiss_retrieve, bm25_retrieve, tfidf_retrieve, load_json
from utils.postgresql_utils import postgres_text_retrieve, postgres_image_retrieve
from utils.paths import TRANSCRIPT_PATH, TEXT_FAISS_INDEX_PATH, IMAGE_FAISS_INDEX_PATH, IMAGE_EMBEDDINGS_JSON
from embed import embed_texts, embed_texts_clip

SIMILARITY_THRESHOLD_TEXT = 0.5
SIMILARITY_THRESHOLD_IMAGE = 0.2

def text_retrieve(query, query_embedding, retrieval_method="faiss", index_type="ivfflat"):
    """Perform text-only retrieval and reject unanswerable queries."""
    
    # Load text embeddings from the saved JSON file
    text_data = load_json(TRANSCRIPT_PATH)
    texts = [entry["text"] for entry in text_data]
    timestamps = [entry["start"] for entry in text_data]

    # Normalize query embedding
    if isinstance(query_embedding, np.ndarray):
        query_embedding = query_embedding / np.linalg.norm(query_embedding)

    # Retrieve text-based results based on chosen retrieval method
    if retrieval_method == "faiss":
        similarity, index = faiss_retrieve(query_embedding, TEXT_FAISS_INDEX_PATH)
        result_text = texts[index]
        result_timestamp = timestamps[index]
    elif retrieval_method == "postgres":
        result_text, result_timestamp, similarity = postgres_text_retrieve(query_embedding, index_type)
    elif retrieval_method == "tfidf":
        index, similarity = tfidf_retrieve(query, texts)
        result_text = texts[index]
        result_timestamp = timestamps[index]
    elif retrieval_method == "bm25":
        index, scores = bm25_retrieve(query, texts)
        result_text = texts[index]
        result_timestamp = timestamps[index]
        similarity = scores[index]
    else:
        raise ValueError("Invalid retrieval method specified.")
    
    # Check similarity threshold for rejection
    if retrieval_method != "bm25" and similarity < SIMILARITY_THRESHOLD_TEXT:
        return None, None, None
    
    return result_text, result_timestamp, similarity

def image_retrieve(query_embedding, retrieval_method="faiss", index_type="ivfflat"):
    """Retrieve the most relevant image based on the query embedding."""

    # Load image metadata (filenames and timestamps)
    image_data = load_json(IMAGE_EMBEDDINGS_JSON)
    filenames = [entry["filename"] for entry in image_data]
    timestamps = [float(entry["filename"].split('_')[-1].split('.')[0]) for entry in image_data]

    # Normalize query embedding
    if isinstance(query_embedding, np.ndarray):
        query_embedding = query_embedding / np.linalg.norm(query_embedding)

    if retrieval_method == "faiss":
        similarity, index = faiss_retrieve(query_embedding, IMAGE_FAISS_INDEX_PATH)
        result_filename = filenames[index]
        result_timestamp = timestamps[index]
    elif retrieval_method == "postgres":
        result_filename, similarity = postgres_image_retrieve(query_embedding, index_type)
        result_timestamp = float(result_filename.split('_')[-1].split('.')[0])
    else:
        raise ValueError("Invalid retrieval method specified for images.")

    # Similarity check (reject bad matches)
    if similarity < SIMILARITY_THRESHOLD_IMAGE:
        return None, None, None

    return result_filename, result_timestamp, similarity


def main():
    parser = argparse.ArgumentParser(description="Retrieval Script")
    parser.add_argument("query", help="The text query for retrieval (can be dummy if image only).")
    parser.add_argument("retrieval_method", choices=["faiss", "postgres", "tfidf", "bm25"], help="Retrieval method to use.")
    parser.add_argument("--modality", choices=["text", "image"], default="text", help="Specify if retrieval is text-based or image-based.")
    parser.add_argument("--index_type", choices=["ivfflat", "hnsw"], default="ivfflat", help="Index type for PostgreSQL retrieval.")

    args = parser.parse_args()

    if args.modality == "text":
        query_embedding = embed_texts(args.query)[0]
    else:
        query_embedding = embed_texts_clip(args.query)[0]

    # Choose retrieval based on modality
    if args.modality == "text":
        result_text, result_timestamp, similarity = text_retrieve(
            args.query,
            query_embedding,
            retrieval_method=args.retrieval_method,
            index_type=args.index_type
        )
        if result_text is not None:
            print("\nTop Text Result:")
            print(f"Result: {result_text}, Timestamp: {result_timestamp} (Similarity: {similarity:.4f})")
        else:
            print("\nNo text result returned.")

    elif args.modality == "image":
        result_filename, result_timestamp, similarity = image_retrieve(
            query_embedding,
            retrieval_method=args.retrieval_method,
            index_type=args.index_type
        )
        if result_filename is not None:
            print("\nTop Image Result:")
            print(f"Image Filename: {result_filename}, Timestamp: {result_timestamp} (Similarity: {similarity:.4f})")
        else:
            print("\nNo image result returned.")

if __name__ == "__main__":
    main()
