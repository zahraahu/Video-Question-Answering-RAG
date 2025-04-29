from utils.paths import TEXT_EMBEDDINGS_JSON, IMAGE_EMBEDDINGS_JSON, TEXT_FAISS_INDEX_PATH, IMAGE_FAISS_INDEX_PATH
from utils.utils import load_json, build_faiss_index
import numpy as np

def main():
    # Load text and image embeddings from the intermediate JSON files
    print("Loading text embeddings from JSON...")
    text_data = load_json(TEXT_EMBEDDINGS_JSON)
    texts = [entry["text"] for entry in text_data]
    timestamps = [entry["timestamp"] for entry in text_data]
    text_embeddings = np.array([entry["embedding"] for entry in text_data], dtype=np.float32)

    print("Loading image embeddings from JSON...")
    image_data = load_json(IMAGE_EMBEDDINGS_JSON)
    filenames = [entry["filename"] for entry in image_data]
    image_embeddings = np.array([entry["embedding"] for entry in image_data], dtype=np.float32)

    # Create FAISS indexes
    print("Building FAISS index for text embeddings...")
    build_faiss_index(text_embeddings, text_embeddings.shape[1], TEXT_FAISS_INDEX_PATH)

    print("Building FAISS index for image embeddings...")
    build_faiss_index(image_embeddings, image_embeddings.shape[1], IMAGE_FAISS_INDEX_PATH)

    print("Finished faiss indexing process!")


if __name__ == "__main__":
    main()
