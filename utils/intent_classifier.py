import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

def classify_intent(user_input: str, gemini_api_key: str) -> str:
    """
    Mengklasifikasikan maksud (intent) pertanyaan pengguna.
    Hanya mengembalikan 'tables' atau 'other'.
    """
    user_input_lower = user_input.lower()

    # 🔍 Periksa kata kunci eksplisit dulu
    table_keywords = [
        "tabel", "data", "jumlah", "berapa", "statistik", "angka", 
        "persentase", "rasio", "total", "banyak", "rata-rata"
    ]
    
    # Jika ada kata kunci tabel yang jelas, langsung return tables
    if any(keyword in user_input_lower for keyword in table_keywords):
        return "tables"
    
    # Jika pertanyaan tentang publikasi, BRS, berita, atau hal umum
    non_table_keywords = [
        "publikasi", "brs", "berita", "acara", "seminar", "event",
        "alamat", "kantor", "profil", "struktur", "organisasi", "tugas"
    ]
    
    if any(keyword in user_input_lower for keyword in non_table_keywords):
        return "other"
        
    # Gunakan LLM untuk kasus yang ambigu
    gemini_api_key = st.secrets["GEMINI_API_KEY"]
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=gemini_api_key,
        temperature=0
    )

    system_prompt = """
    Kamu adalah intent classifier untuk chatbot data BPS Kota Bandung.

    Kategorikan pertanyaan pengguna ke dalam salah satu kategori berikut:

    1. tables → jika pengguna meminta data statistik, jumlah, rasio, persentase,
       atau angka yang berasal dari tabel data BPS.
       Contoh: "berapa jumlah penduduk?", "sekolah", "kemiskinan di bandung"

    2. other → jika pengguna meminta publikasi, BRS, berita, profil BPS,
       atau hal-hal yang tidak terkait dengan data tabel statistik.
       Contoh: "publikasi terbaru", "alamat kantor BPS", "halo apa kabar"

    Output HARUS berupa SATU KATA: 'tables' atau 'other'.
    Jangan sertakan penjelasan tambahan.
    Jika ragu, pilih "tables" karena fokus chatbot adalah data tabel.
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt.strip()),
        ("user", "{question}")
    ])

    chain = prompt | llm
    response = chain.invoke({"question": user_input})
    result = response.content.strip().lower()

    # Validasi hasil
    if result not in {"tables", "other"}:
        return "tables"  # Default ke tables
    return result
