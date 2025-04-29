# utils/path.py
from pathlib import Path

# Base data directory
DATA_DIR = Path("data")

# Processing paths
VIDEO_PATH = DATA_DIR / "video.mp4"
AUDIO_PATH = DATA_DIR / "audio.mp3"
KEYFRAMES_DIR = DATA_DIR / "keyframes"
TRANSCRIPT_PATH = DATA_DIR / "transcript.json"
PROCESSED_DATA_PATH = DATA_DIR / "frame_to_text.json"

# Embedding paths
TEXT_EMBEDDINGS_JSON = DATA_DIR / "text_embeddings.json"
IMAGE_EMBEDDINGS_JSON = DATA_DIR / "image_embeddings.json"

# FAISS indexes
TEXT_FAISS_INDEX_PATH = DATA_DIR / "faiss_text.index"
IMAGE_FAISS_INDEX_PATH = DATA_DIR / "faiss_image.index"

# EVAL paths
GOLD_TEST_PATH = DATA_DIR / "gold_test.json"
EVALUATION_RESULTS_PATH = DATA_DIR / "evaluation_results.json"
EVALUATION_SUMMARY_CSV = DATA_DIR / "evaluation_summary.csv"


