from utils.paths import TEXT_EMBEDDINGS_JSON, IMAGE_EMBEDDINGS_JSON, TEXT_FAISS_INDEX_PATH, IMAGE_FAISS_INDEX_PATH
from utils.utils import load_json, build_faiss_index
from utils.postgresql_utils import save_text_embeddings_to_postgres, save_image_embeddings_to_postgres, create_index_on_embeddings
from utils.postgresql_utils import create_text_embeddings_table, create_image_embeddings_table
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

    # Save text and image embeddings to Postgres
    print("Saving text embeddings to Postgres...")
    create_text_embeddings_table()
    save_text_embeddings_to_postgres(texts, timestamps, text_embeddings)

    print("Saving image embeddings to Postgres...")
    create_image_embeddings_table()
    save_image_embeddings_to_postgres(filenames, image_embeddings)

    # Create Postgres indexes for vector search
    print("Creating vector indexes on Postgres...")
    create_index_on_embeddings()
    print("Postgres indexes created.")

    print("Finished indexing process!")


if __name__ == "__main__":
    main()
