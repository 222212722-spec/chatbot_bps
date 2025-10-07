import logging
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from .embeddings import embed_query_text
from utils.intent_classifier import classify_intent
from utils.llm_handler import get_llm
from utils.retriever import retrieve_from_zilliz
from utils.security_filter import is_malicious

from utils.staticparsing import parse_table_static
from utils.dynamicparsing import parse_table_dynamic
from utils.simdasiparsing import parse_table_simdasi

# Konfigurasi logging
logging.basicConfig(level=logging.INFO)

def condense_chat_history(user_input: str, chat_history: list):
    """
    Mengambil pesan terakhir pengguna dan riwayat obrolan, dan mengembalikan
    pertanyaan mandiri yang dapat dipahami tanpa konteks penuh.
    """
    if not chat_history:
        return user_input

    # Gunakan model cepat seperti Groq untuk tugas internal ini
    llm = get_llm("groq") 

    condenser_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{question}"),
        ("system", 
         """
         Anda adalah AI yang bertugas memformat ulang pertanyaan. JANGAN menjawab pertanyaan.
         TUGAS UTAMA ANDA: Berdasarkan riwayat obrolan dan pertanyaan lanjutan, ubah pertanyaan lanjutan menjadi pertanyaan mandiri yang lengkap.
         
         Contoh 1:
         Riwayat: User bertanya "Berapa inflasi di Bandung tahun 2023?", Anda menjawab "3.5%".
         Pertanyaan Lanjutan: "bagaimana dengan tahun sebelumnya?"
         Output Anda: "Berapa inflasi di Bandung tahun 2022?"

         Contoh 2:
         Riwayat: Tidak ada.
         Pertanyaan Lanjutan: "siapa kepala bps kota bandung?"
         Output Anda: "siapa kepala bps kota bandung?"

         JANGAN menjawab pertanyaan. HANYA format ulang pertanyaan tersebut.
         """)
    ])
    
    chain = condenser_prompt | llm
    
    # Format riwayat untuk prompt
    formatted_history = []
    for message in chat_history:
        if message["role"] == "user":
            formatted_history.append(HumanMessage(content=message["content"]))
        elif message["role"] == "assistant":
            formatted_history.append(AIMessage(content=message["content"]))

    response = chain.invoke({
        "chat_history": formatted_history,
        "question": user_input
    })

    logging.info(f"Pertanyaan asli: '{user_input}'")
    logging.info(f"Pertanyaan yang dipadatkan: '{response.content}'")
    return response.content

def rag_answer(user_input: str, chat_history: list, provider="mistral"):
    """
    Menangani seluruh proses RAG, sekarang dengan riwayat percakapan.
    Provider default adalah mistral.
    """
    if is_malicious(user_input):
        logging.warning(f"Input berbahaya terdeteksi: {user_input}")
        return None, "⚠️ Pertanyaan terdeteksi berbahaya atau tidak pantas.", "blocked"

    logging.info("Langkah 0.5: Memadatkan riwayat obrolan...")
    standalone_question = condense_chat_history(user_input, chat_history)

    logging.info("Langkah 1: Mengklasifikasikan intent...")
    intent = classify_intent(standalone_question, st.secrets["FIREWORKS_API_KEY"])
    logging.info(f"Intent diklasifikasikan sebagai: {intent}")

    if intent == "other":
        logging.info("Intent adalah 'other', kembali lebih awal.")
        return None, "ℹ️ Maaf, informasi tersebut belum tersedia di chatbot.", "other"
    
    logging.info("Langkah 2: Membuat embedding untuk query...")
    try:
        query_vector = embed_query_text(standalone_question)
        logging.info("Berhasil membuat vektor query via API.")
    except Exception as e:
        logging.error(f"Error selama embedding: {e}", exc_info=True)
        return None, f"❌ Terjadi kesalahan saat membuat embedding: {e}.", "error"
    
    logging.info(f"Langkah 3: Mengambil dokumen dari koleksi Zilliz: '{intent}'")
    docs = retrieve_from_zilliz(query_vector, intent)
    
    logging.info("Langkah 4: Merakit konteks dan sitasi...")
    context = ""
    citations = [] # Buat list kosong
    if docs and docs[0]:
        if intent == "tables":
            # pilih hanya dokumen teratas
            hit = docs[0][0]
            entity = hit.entity
            table_attrs = {
                "unique": entity.get("unique"),
                "id": entity.get("id"),
                "id_table": entity.get("id_table"),
                "title": entity.get("title"),
                "subcat": entity.get("subcat"),
                "tablesource": entity.get("tablesource"),
                "years": entity.get("years"),
                "last_update": entity.get("last_update"),
                "link": entity.get("link"),
                "page_content": entity.get("page_content", "")
            }

            ts = str(table_attrs.get("tablesource", "")).strip()
            if ts == "1":
                parsed_context = parse_table_static(table_attrs)
            elif ts == "2":
                parsed_context = parse_table_dynamic(table_attrs)
            elif ts == "3":
                parsed_context = parse_table_simdasi(table_attrs)
            else:
                parsed_context = table_attrs.get("page_content", "")

            context = f"[SUMBER 1]: {parsed_context}\n\n"
            citations.append({"id": 1, "title": table_attrs.get("title"), "link": table_attrs.get("link")})
        else:
        # selain tables
            for i, r in enumerate(docs[0]):
                entity = r.entity
                doc_content = entity.get("page_content", "")
                context += f"[SUMBER {i+1}]: {doc_content}\n\n"
                
                link = entity.get("link")
                title = entity.get("title", f"Sumber {i+1}")

                # Tambahkan setiap sumber yang ditemukan ke dalam daftar. TANPA FILTER.
                citations.append({"id": i + 1, "title": title, "link": link})
        
        logging.info(f"Menemukan {len(docs[0])} dokumen relevan.")
    else:
        logging.warning("Tidak ada dokumen relevan yang ditemukan di database.")
        return None, "ℹ️ Tidak ada hasil relevan yang ditemukan di database BPS Kota Bandung.", "retrieval_fail"


    logging.info(f"Langkah 5: Menghasilkan jawaban dengan LLM (Provider: {provider})...")
    llm = get_llm(provider)

    # --- PROMPT ---
    prompt = f"""
    Anda adalah chatbot AI dari BPS Kota Bandung. Gunakan konteks yang diberikan untuk menjawab pertanyaan.
    Jawablah dengan informatif dan singkat. Jawab selalu dalam Bahasa Indonesia atau Inggris.

    ATURAN SANGAT PENTING:
    1. Berikan jawaban Anda HANYA berdasarkan informasi dari [SUMBER 1], [SUMBER 2], dst.
    2. Hubungkan setiap fakta dengan sumbernya secara akurat. JANGAN menggabungkan informasi dari sumber yang berbeda menjadi satu kalimat kecuali jika keduanya mendukung fakta yang sama.
    3. Saat Anda menggunakan informasi dari sebuah sumber, Anda WAJIB menyisipkan sitasi di akhir kalimat atau klausa dengan format [nomor sumber]. Contoh: "Inflasi di Kota Bandung pada bulan Mei adalah 2.5% [1]."
    4. JANGAN membuat informasi sendiri. Jika jawaban tidak ada di konteks, katakan dengan sopan bahwa Anda tidak dapat menemukan informasinya.
    5. JANGAN sertakan daftar sumber di akhir jawaban Anda. Itu akan ditambahkan secara otomatis oleh sistem.

    Konteks:
    ---
    {context}
    ---

    Pertanyaan: {standalone_question}

    Jawaban (ingat untuk menyisipkan sitasi seperti [1], [2], dst. dan mengikuti semua aturan):
    """

    try:
        response_stream = llm.stream(prompt)
        logging.info("Berhasil memulai streaming respons.")
        return response_stream, citations, intent
    except Exception as e:
        logging.error(f"Error selama generasi LLM: {e}", exc_info=True)
        return None, f"❌ Terjadi kesalahan saat menghasilkan jawaban: {e}", "error"

