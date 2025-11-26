import streamlit as st
from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI
from langchain_google_genai import ChatGoogleGenerativeAI

def get_llm(provider="mistral"):
    """
    Mengambil instance model bahasa dari provider yang ditentukan.
    Default diubah ke 'mistral' untuk kualitas jawaban yang lebih tinggi.
    """
    if provider == "mistral":
        # Mistral Large adalah model yang sangat kuat, bagus untuk generasi jawaban yang akurat.
        return ChatMistralAI(
            model="mistral-large-2411", 
            api_key=st.secrets["MISTRAL_API_KEY"],
            temperature=0.1 # Sedikit lebih rendah untuk jawaban yang lebih faktual
        )
    elif provider == "gemini":
        # Opsi alternatif yang juga sangat baik
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            api_key=st.secrets["GEMINI_API_KEY"],
            temperature=0.1,
            convert_system_message_to_human=True
        )
    elif provider == "groq":
        # Groq tetap digunakan untuk tugas cepat seperti 'condense_chat_history'
        return ChatGroq(
            model="openai/gpt-oss-120b", 
            api_key=st.secrets["GROQ_API_KEY"]
        )
    else:
        # Fallback ke Mistral jika provider tidak dikenali
        return ChatMistralAI(
            model="mistral-large-2411", 
            api_key=st.secrets["MISTRAL_API_KEY"],
            temperature=0.1
        )
