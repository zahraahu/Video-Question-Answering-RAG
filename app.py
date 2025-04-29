import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "scripts")))
import streamlit as st
from scripts.retrieve import text_retrieve, image_retrieve
from scripts.embed import embed_texts
from scripts.utils.paths import VIDEO_PATH

# Choose modality / retrieval method
RETRIEVAL_METHOD = "tfidf"    # or "faiss", "postgres", "bm25"
INDEX_TYPE = ""    # if using postgres, you may need this ("ivfflat", "hnsw")
MODALITY = "text"  # or "image"

# Setup
st.set_page_config(page_title="Video QA RAG", page_icon="🎥", layout="wide")
st.title("🎬 Video Question Answering")

# Chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Function to retrieve video segment
def retrieve_video_segment(start_time, video_path=VIDEO_PATH):
    video_file = open(video_path, "rb")
    video_bytes = video_file.read()
    return video_bytes

# User input
user_query = st.chat_input("Ask a question about the video...")

if user_query:
    # Display user query
    st.session_state.chat_history.append(("user", user_query))

    # Embed the query
    query_embedding = embed_texts([user_query])[0]

    # Retrieve
    if MODALITY == "text":
        _, result_timestamp, _ = text_retrieve(user_query, query_embedding, RETRIEVAL_METHOD, INDEX_TYPE)
    elif MODALITY == "image":
        _, result_timestamp, _ = image_retrieve(query_embedding, RETRIEVAL_METHOD, INDEX_TYPE)

    if result_timestamp is not None:
        minutes = int(result_timestamp) // 60
        seconds = int(result_timestamp) % 60
        formatted_timestamp = f"{minutes:02d}:{seconds:02d}"
        response_text = f"✅ I found the answer at timestamp: {formatted_timestamp}."


        # Load the video segment
        video_bytes = retrieve_video_segment(start_time=float(result_timestamp))

        st.session_state.chat_history.append(("assistant", response_text))

        # Display history
        for speaker, message in st.session_state.chat_history:
            with st.chat_message(speaker):
                st.write(message)

        # Show video below
        st.video(video_bytes, start_time=float(result_timestamp))

    else:
        # No result
        response_text = "❌ Sorry, I couldn't find an answer in the video for that question."

        st.session_state.chat_history.append(("assistant", response_text))

        for speaker, message in st.session_state.chat_history:
            with st.chat_message(speaker):
                st.write(message)
