import psycopg2
from psycopg2.extras import Json
import numpy as np
import os
from dotenv import load_dotenv

def connect_to_postgres():
    """
    Connect to PostgreSQL database and return the connection object.
    """

    load_dotenv()

    conn = psycopg2.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD")
    )

    return conn


def create_text_embeddings_table():
    """
    Create the PostgreSQL table for storing text embeddings along with timestamp.
    """
    conn = connect_to_postgres()
    cur = conn.cursor()
    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS text_embeddings (
            id SERIAL PRIMARY KEY,
            text TEXT,
            timestamp TEXT, 
            text_embedding_hnsw vector(768),
            text_embedding_ivfflat vector(768)
        );
    """)
    conn.commit()
    cur.close()
    conn.close()
    print("Text embeddings table created.")

def create_image_embeddings_table():
    """
    Create the PostgreSQL table for storing image embeddings along with timestamp.
    """
    conn = connect_to_postgres()
    cur = conn.cursor()
    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS image_embeddings (
            id SERIAL PRIMARY KEY,
            filename TEXT,
            timestamp TEXT, 
            image_embedding_hnsw vector(512),
            image_embedding_ivfflat vector(512)
        );
    """)
    conn.commit()
    cur.close()
    conn.close()
    print("Image embeddings table created.")

def save_text_embeddings_to_postgres(texts, timestamps, text_embeddings):
    """
    Save text embeddings to PostgreSQL for different indexing methods, including timestamps.
    """
    conn = connect_to_postgres()
    cur = conn.cursor()

    text_emb_list = text_embeddings.tolist()
    
    for text, timestamp, text_emb in zip(texts, timestamps, text_emb_list):
        cur.execute("""
            INSERT INTO text_embeddings (text, timestamp, text_embedding_hnsw, text_embedding_ivfflat)
            VALUES (%s, %s, %s, %s);
        """, (
            text,
            timestamp,  
            text_emb,  # Placeholder for IVFFLAT 
            text_emb   # Placeholder for HNSW 
        ))

    conn.commit()
    cur.close()
    conn.close()

def save_image_embeddings_to_postgres(image_filenames, image_embeddings):
    """
    Save image embeddings to PostgreSQL for different indexing methods.
    """
    conn = connect_to_postgres()
    cur = conn.cursor()

    image_emb_list = image_embeddings.tolist()
    for img_filename, img_emb in zip(image_filenames, image_emb_list):
        cur.execute("""
            INSERT INTO image_embeddings (filename, image_embedding_hnsw, image_embedding_ivfflat)
            VALUES (%s, %s, %s);
        """, (
            img_filename,
            img_emb,  # Placeholder for HNSW 
            img_emb   # Placeholder for IVFFLAT 
        ))

    conn.commit()
    cur.close()
    conn.close()

def create_index_on_embeddings():
    """
    Create indexes for embeddings to optimize similarity queries (IVFFLAT and HNSW).
    """
    conn = connect_to_postgres()
    cur = conn.cursor()

    # IVFFLAT Index
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_text_ivfflat 
        ON text_embeddings USING ivfflat (text_embedding_ivfflat vector_cosine_ops)
        WITH (lists = 100);
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_image_ivfflat 
        ON image_embeddings USING ivfflat (image_embedding_ivfflat vector_cosine_ops)
        WITH (lists = 100);
    """)

    # HNSW Index
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_text_hnsw 
        ON text_embeddings USING hnsw (text_embedding_hnsw vector_cosine_ops)
        WITH (m = 16, ef_construction = 64);
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_image_hnsw 
        ON image_embeddings USING hnsw (image_embedding_hnsw vector_cosine_ops)
        WITH (m = 16, ef_construction = 64);
    """)
    
    conn.commit()
    cur.close()
    conn.close()

def postgres_text_retrieve(query_embedding, index_type="ivfflat", top_k=1):
    """
    Retrieve top-k documents from PostgreSQL using pgvector (IVFFLAT or HNSW), including timestamps.
    """
    conn = connect_to_postgres()
    cur = conn.cursor()
    
    query_vector = np.array(query_embedding, dtype=np.float32)
    query_vector = query_vector.tolist()  # Convert to list for JSON compatibility

    if index_type == "ivfflat":
        query_column = "text_embedding_ivfflat"
    elif index_type == "hnsw":
        query_column = "text_embedding_hnsw"
    else:
        raise ValueError("Invalid index type specified. Choose 'ivfflat' or 'hnsw'.")

    # Perform search in PostgreSQL using pgvector cosine similarity
    cur.execute(f"""
        SELECT id, text, timestamp, {query_column} <=> %s AS distance
        FROM text_embeddings
        ORDER BY distance
        LIMIT %s;
    """, (Json(query_vector), top_k))

    results = cur.fetchall()
    cur.close()
    conn.close()

    return results[0][1], results[0][2], 1 - results[0][3]  # Return text, timestamp, and similarity

def postgres_image_retrieve(query_embedding, index_type="ivfflat", top_k=1):
    conn = connect_to_postgres()
    cur = conn.cursor()

    query_vector = np.array(query_embedding, dtype=np.float32).tolist()

    if index_type == "ivfflat":
        query_column = "image_embedding_ivfflat"
    elif index_type == "hnsw":
        query_column = "image_embedding_hnsw"
    else:
        raise ValueError("Invalid index type specified. Choose 'ivfflat' or 'hnsw'.")

    cur.execute(f"""
        SELECT id, filename, {query_column} <=> %s AS distance
        FROM image_embeddings
        ORDER BY distance
        LIMIT %s;
    """, (Json(query_vector), top_k))

    results = cur.fetchall()
    cur.close()
    conn.close()

    return results[0][1], 1 - results[0][2]  # Return filename and similarity 

