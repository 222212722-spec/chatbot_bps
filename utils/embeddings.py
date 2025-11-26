import os
import streamlit as st
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_core.embeddings import Embeddings
from typing import List

# Atur token (kunci rahasia) dari Streamlit ke environment
if "HF_TOKEN" in st.secrets:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HF_TOKEN"]

@st.cache_resource
def get_query_embedding_model() -> Embeddings:
    """
    Menginisialisasi dan menyimpan (cache) model embeddinggemma-300m
    melalui Titik Akhir (Endpoint) Inferensi Hugging Face.
    Fungsi ini hanya akan dijalankan sekali.
    """
    hf_token = st.secrets.get("HF_TOKEN")
    if not hf_token:
	# Tampilkan error jika kunci rahasia (token) tidak ditemukan
        raise ValueError("HF_TOKEN is not set in secrets.toml.")

    # Gunakan kelas HuggingFaceEndpointEmbeddings untuk embedding berbasis API
    embeddings = HuggingFaceEndpointEmbeddings(
        repo_id="google/embeddinggemma-300m",
        huggingfacehub_api_token=hf_token,
    )
    return embeddings

def embed_query_text(query: str) -> List[float]:
    model = get_query_embedding_model()
    sanitized = " ".join(query.strip().split())  # hapus spasi berlebih, newline, dsb.
    prefixed_query = f"task: search result | query: {sanitized}"
    return model.embed_query(prefixed_query)
