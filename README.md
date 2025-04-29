# 🎥 Video Question Answering with Multimodal RAG

This project allows you to ask natural language questions about a video and returns:
- Relevant **timestamps** where the answer appears
- An **embedded video player** showing that segment
- A conversational **chat interface** (built with Streamlit)
- A rejection message if no relevant answer is found



## 🚀 Features

- Retrieval using text and image embeddings
- Powered by Open AI's CLIP for image embedding, and Snowflake Arctic for text embedding.
- Supports multiple retrieval backends:
  - FAISS (in-memory flat index)
  - PostgreSQL with pgvector (ivfflat and hnsw indexes)
  - BM25 and TF-IDF (text-only methods)
- Video visualization right in the browser



## 🛠️ Setup

### 1. Clone the repo

```bash
git clone https://github.com/zahraahu/Video-Question-Answering-RAG.git
cd Video-Question-Answering-RAG
```

### 2. Create a virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install ffmpeg

To enable video clipping (used to extract relevant video segments), make sure **`ffmpeg`** is installed and available in your system's PATH.

- On macOS: `brew install ffmpeg`
- On Ubuntu/Debian: `sudo apt install ffmpeg`
- On Windows:
  1. Download from [ffmpeg.org](https://ffmpeg.org/download.html)
  2. Extract it and add the `bin/` folder to your system's PATH
     
### 4. (Optional) Set up PostgreSQL + pgvector

If you want to use the PostgreSQL retrieval option:

- Ensure PostgreSQL is installed
- Create a database and enable the `pgvector` extension: [pgvector](https://github.com/pgvector/pgvector)
- Add `.env` file in the root:

```
DB_NAME=qa_rag
DB_USER=your_user
DB_PASSWORD=your_password
```
- run  `postgres_index.py`:

```
python postgres_index.py
```

**You can skip this step** if you’re only using FAISS or text-based retrieval.

### 5. (Optional) Adjust Retrieval Method

The retrieval method, index type, and modality can be adjusted through the global variables in `app.py`.
By default, the code is:

```
# Choose modality / retrieval method

RETRIEVAL_METHOD = "tfidf"    # or "faiss", "postgres", "bm25"
INDEX_TYPE = ""    # if using postgres, you may need this ("ivfflat", "hnsw")
MODALITY = "text"  # or "image"
```



## ▶️ Run the App

```bash
streamlit run app.py
```



## 🧠 To Test Out Different Videos

The following steps have already been completed and the results are included in the repository, but if you'd like to test out a different video, be sure to run:

```bash
process.py                 # Extracts video frames and audio
python embed.py            # Generates CLIP embeddings for frames and text
python faiss_index.py      # Builds FAISS indexes
python postgres_index.py   # Build Postgres indexes (ivfflat and hnsw)
```

**You do NOT need to run these scripts** unless you're using a different video.



## 💬 How It Works

1. You ask a question.
2. The app retrieves relevant frames/text using the chosen retrieval method.
3. If a match is found, it returns:
   - A timestamp
   - A video player starting at that time
4. If no relevant match is found, it lets you know and keeps the chat going.



## 📊 Evaluation

Run `scripts/evaluate.py` to benchmark each retrieval method on accuracy, rejection quality, and latency.
The queries to be tested can be found in `gold_test.json`, such that 10 questions are directly answerable from the video (include a corresponding timestamp), and 5 are unanswerable.

The result of each run is saved in `evaluation_results.json`, and then the accuracy and rejection quality of each method are assessed and saved in `evaluation_summary.csv`.



## 📽️ Notes

- Currently supports a single preprocessed video.
- Timestamp precision is within ±20 seconds tolerance by default.
