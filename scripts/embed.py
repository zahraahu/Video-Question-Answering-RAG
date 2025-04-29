from utils.paths import TRANSCRIPT_PATH, KEYFRAMES_DIR, TEXT_EMBEDDINGS_JSON, IMAGE_EMBEDDINGS_JSON
from utils.utils import load_json, save_json
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import numpy as np
from sentence_transformers import SentenceTransformer

# Load text model
TEXT_MODEL_NAME = "Snowflake/snowflake-arctic-embed-m-v1.5"
TEXT_MODEL = SentenceTransformer(TEXT_MODEL_NAME)

# Load CLIP model
MODEL_NAME = "openai/clip-vit-base-patch32"
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
model = CLIPModel.from_pretrained(MODEL_NAME)
model.eval()

def embed_texts(texts):
    """Embed a list of texts into float32 numpy arrays."""
    embeddings = TEXT_MODEL.encode(texts, convert_to_numpy=True)
    return embeddings.astype(np.float32)


def embed_texts_clip(texts):
    """Embed a list of texts into float32 numpy arrays using CLIP."""
    inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model.get_text_features(**inputs)
    embeddings = outputs.cpu().numpy().astype(np.float32)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings


def embed_images(image_dir):
    """Embed images from a directory into float32 numpy arrays using CLIP."""
    embeddings = []
    filenames = []
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory {image_dir} not found.")

    for fname in sorted(image_dir.glob("*.jpg")):
        try:
            image = Image.open(fname).convert("RGB")
        except Exception as e:
            print(f"Error loading image {fname}: {e}")
            continue

        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model.get_image_features(**inputs)
            emb = outputs.cpu().numpy().flatten().astype(np.float32)
            emb = emb / np.linalg.norm(emb)
        embeddings.append(emb)
        filenames.append(fname.name)

    return np.array(embeddings), filenames


def main():
    # Load transcript
    chunks = load_json(TRANSCRIPT_PATH)
    texts = [chunk["text"] for chunk in chunks]
    timestamps = [chunk["start"] for chunk in chunks]

    # Text embedding
    print("Embedding text...")
    text_embs = embed_texts(texts)
    text_records = [{"text": text, "embedding": emb.tolist(), "timestamp": timestamp} for text, emb, timestamp in zip(texts, text_embs, timestamps)]
    save_json(text_records, TEXT_EMBEDDINGS_JSON)

    # Image embedding
    print("Embedding images...")
    image_embs, filenames = embed_images(KEYFRAMES_DIR)
    image_records = [{"filename": fname, "embedding": emb.tolist()} for fname, emb in zip(filenames, image_embs)]
    save_json(image_records, IMAGE_EMBEDDINGS_JSON)

    print("Finished embedding and saving to intermediate JSON files.")


if __name__ == "__main__":
    main()
