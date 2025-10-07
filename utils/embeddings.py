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
    """
    Mengubah teks pertanyaan (query) mentah menjadi vektor embedding
    menggunakan awalan (prefix), yang merupakan best practice
    untuk model embedding Gemma.
    """
    model = get_query_embedding_model()
    # Model Gemma bekerja lebih baik jika tugasnya ditentukan (misalnya, 'query' vs 'document')
    prefixed_query = f"query: {query}"
    
    # Lakukan konversi teks (dengan awalan) menjadi vektor angka
    return model.embed_query(prefixed_query)
