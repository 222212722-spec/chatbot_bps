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

# Import untuk error handling
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Konfigurasi logging
logging.basicConfig(level=logging.INFO)

def condense_chat_history(user_input: str, chat_history: list):
    """
    Memadatkan riwayat obrolan dan memperlengkapi pertanyaan menjadi query yang lebih baik.
    Jika user hanya tulis kata kunci singkat, lengkapi dengan konteks BPS Kota Bandung.
    """
    if not chat_history:
        # Tidak ada riwayat, tetap perlengkapi query
        try:
            llm = get_llm("groq")
            
            simple_prompt = ChatPromptTemplate.from_messages([
                ("user", "{question}"),
                ("system",
                """
                Anda adalah AI yang bertugas memperlengkapi pertanyaan pengguna menjadi query pencarian yang baik untuk data BPS Kota Bandung.
                
                Tugas Anda:
                1. Jika pertanyaan sudah lengkap (ada subjek, predikat, keterangan), biarkan apa adanya.
                2. Jika pertanyaan hanya kata kunci singkat (misal: "sekolah", "kemiskinan", "inflasi"), 
                   lengkapi menjadi query yang eksplisit: "jumlah/data [topik] di Kota Bandung tahun terbaru"
                3. Tambahkan konteks "Kota Bandung" jika belum ada.
                4. Tambahkan kata "data" atau "jumlah" jika belum ada untuk memperjelas maksud pencarian data statistik.
                5. Jangan menjawab pertanyaan, hanya format ulang menjadi query pencarian yang baik.
                
                Contoh:
                Input: "sekolah"
                Output: "jumlah sekolah di Kota Bandung"
                
                Input: "kemiskinan 2023"
                Output: "data kemiskinan di Kota Bandung tahun 2023"
                
                Input: "berapa inflasi bulan ini?"
                Output: "inflasi bulan terbaru di Kota Bandung"
                
                Input: "jumlah penduduk menurut jenis kelamin"
                Output: "jumlah penduduk menurut jenis kelamin di Kota Bandung"
                
                Pastikan hasil Anda jelas, spesifik, dan cocok untuk pencarian data BPS.
                """)
            ])
            
            chain = simple_prompt | llm
            response = chain.invoke({"question": user_input})
            result = response.content.strip()
            
            logging.info(f"Pertanyaan asli: '{user_input}'")
            logging.info(f"Query yang diperlengkapi: '{result}'")
            return result
        
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                logging.warning("Rate limit tercapai saat condense query, gunakan query asli")
                return user_input
            else:
                logging.error(f"HTTP error saat condense: {e}")
                return user_input
        except Exception as e:
            logging.error(f"Error saat condense query: {e}")
            # Fallback: gunakan query asli
            return user_input

    # Ada riwayat chat
    try:
        llm = get_llm("groq")

        condenser_prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{question}"),
            ("system",
            """
            Anda adalah AI yang bertugas untuk:
            1. Memformat ulang pertanyaan lanjutan menjadi pertanyaan mandiri yang lengkap.
            2. Memperlengkapi pertanyaan singkat dengan konteks yang relevan.
            3. Menambahkan "Kota Bandung" jika belum ada dalam konteks.
            4. Menambahkan kata kunci seperti "data" atau "jumlah" untuk memperjelas pencarian statistik.
            5. Jangan menjawab pertanyaan. Hanya ubah dan kembalikan versi yang sudah dipadatkan dan diperlengkapi.

            Contoh 1:
            Riwayat:
            - User: "Berapa inflasi di Bandung tahun 2023?"
            - Assistant: "3.5%"
            Pertanyaan lanjutan: "Bagaimana dengan tahun sebelumnya?"
            Output: "data inflasi di Kota Bandung tahun 2022"

            Contoh 2:
            Riwayat: Tidak ada.
            Pertanyaan: "sekolah"
            Output: "jumlah sekolah di Kota Bandung"

            Contoh 3:
            Riwayat:
            - User: "Ada data kemiskinan?"
            - Assistant: "Kemiskinan yang mana ya?"
            Pertanyaan: "Yang terbaru"
            Output: "data kemiskinan terbaru di Kota Bandung"

            Pastikan hasil Anda jelas, lengkap, dan cocok untuk diproses embedding.
            Jangan menjawab pertanyaannya.
            """)
        ])

        chain = condenser_prompt | llm

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
        logging.info(f"Pertanyaan yang dipadatkan & diperlengkapi: '{response.content}'")
        return response.content
    
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 429:
            logging.warning("Rate limit tercapai saat condense query dengan riwayat, gunakan query asli")
            return user_input
        else:
            logging.error(f"HTTP error saat condense dengan riwayat: {e}")
            return user_input
    except Exception as e:
        logging.error(f"Error saat condense query dengan riwayat: {e}")
        return user_input

def rag_answer(user_input: str, chat_history: list, provider="mistral"):
    """
    Menangani seluruh proses RAG, fokus ke tabel saja.
    """
    if is_malicious(user_input):
        logging.warning(f"Input berbahaya terdeteksi: {user_input}")
        return None, "⚠️ Pertanyaan terdeteksi berbahaya atau tidak pantas.", "blocked"

    # Langkah 1: Condense dan perlengkapi query
    logging.info("Langkah 1: Memproses dan memperlengkapi query...")
    standalone_question = condense_chat_history(user_input, chat_history)
    logging.info(f"Processed question: {standalone_question}")

    # Langkah 2: Intent classification
    logging.info("Langkah 2: Mengklasifikasikan intent...")
    intent = classify_intent(standalone_question, st.secrets["GEMINI_API_KEY"])
    logging.info(f"Intent: {intent}")

    if intent == "other":
        logging.info("Intent adalah 'other', chatbot hanya fokus ke data tabel.")
        return None, "ℹ️ Maaf, saat ini chatbot hanya dapat menjawab pertanyaan tentang data tabel statistik BPS Kota Bandung. Untuk informasi lain seperti publikasi, BRS, atau berita, silakan kunjungi website resmi BPS Kota Bandung.", "other"
    
    # Langkah 3: Embedding
    logging.info("Langkah 3: Membuat embedding untuk query...")
    try:
        query_vector = embed_query_text(standalone_question)
        logging.info("Berhasil membuat vektor query via API.")
    except Exception as e:
        logging.error(f"Error selama embedding: {e}", exc_info=True)
        return None, f"❌ Terjadi kesalahan saat membuat embedding: {e}.", "error"
    
    # Langkah 4: Retrieve top 3 dokumen
    logging.info(f"Langkah 4: Mengambil 3 dokumen teratas dari koleksi 'tables'")
    docs = retrieve_from_zilliz(query_vector, "tables", top_k=3)
    
    # Langkah 5: Parse semua 3 tabel
    logging.info("Langkah 5: Merakit konteks dari 3 tabel teratas...")
    context = ""
    citations = []
    
    if docs and docs[0]:
        for i, hit in enumerate(docs[0][:3], start=1):  # Ambil max 3 tabel
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
            
            # Parse berdasarkan jenis tabel
            if ts == "1":
                parsed_content = parse_table_static(table_attrs)
            elif ts == "2":
                parsed_content = parse_table_dynamic(table_attrs)
            elif ts == "3":
                parsed_content = parse_table_simdasi(table_attrs)
            else:
                parsed_content = table_attrs.get("page_content", "")

            # Tambahkan judul tabel di awal
            table_title = table_attrs.get("title", f"Tabel {i}")
            context += f"[SUMBER {i}]\nJudul Tabel: {table_title}\nData:\n{parsed_content}\n\n"
            
            citations.append({
                "id": i, 
                "title": table_title, 
                "link": table_attrs.get("link")
            })
        
        logging.info(f"Berhasil memproses {len(citations)} tabel.")
    else:
        logging.warning("Tidak ada dokumen relevan yang ditemukan di database.")
        return None, "ℹ️ Tidak ada data tabel yang relevan ditemukan di database BPS Kota Bandung untuk pertanyaan Anda. Coba ubah kata kunci pencarian atau tanyakan topik data lainnya.", "retrieval_fail"

    # Langkah 6: Generate jawaban dengan LLM
    logging.info(f"Langkah 6: Menghasilkan jawaban dengan LLM (Provider: {provider})...")
    llm = get_llm(provider)

    prompt = f"""
    Anda adalah chatbot AI dari BPS Kota Bandung yang HANYA menjawab pertanyaan tentang data tabel statistik.
    
    KETENTUAN PENTING:
    1. Anda diberikan maksimal 3 tabel yang relevan dengan pertanyaan.
    2. Baca dengan teliti JUDUL dan ISI setiap tabel sebelum menjawab.
    3. Pilih tabel yang PALING relevan dengan pertanyaan user, jika tahun tidak sama, beri data tahun yang tersedia saja tidak masalah.
    4. Jika ada beberapa tabel yang relevan, gabungkan informasinya dengan sitasi yang tepat.
    5. Jika TIDAK ADA tabel yang relevan dengan pertanyaan, katakan dengan jelas bahwa data tidak tersedia.

    CARA MENJAWAB:
    1. Jawab langsung dan spesifik sesuai pertanyaan atau yang relevan dan mendekati.
    2. Sebutkan angka, nilai, atau informasi yang diminta dengan jelas.
    3. Hubungkan setiap fakta dengan sumbernya menggunakan format [nomor sumber].
    4. Jika data yang ditanya tidak ada di tabel manapun, katakan: "Maaf, data tentang [topik] tidak tersedia dalam tabel yang ditemukan."

    LARANGAN:
    1. JANGAN mencampur informasi dari tabel yang berbeda tanpa sitasi yang jelas
    2. JANGAN sertakan daftar sumber di akhir (akan ditambahkan otomatis)

    CONTOH JAWABAN YANG BAIK:
    "Jumlah sekolah SD di Kota Bandung tahun 2023 adalah 850 sekolah [1]. Sementara untuk SMP terdapat 320 sekolah [2]."

    CONTOH JAWABAN JIKA DATA TIDAK ADA:
    "Maaf, data tentang jumlah universitas swasta tidak tersedia dalam tabel yang ditemukan. Tabel yang tersedia berisi data tentang jumlah sekolah SD dan SMP [1]."

    ===== DATA TABEL =====
    {context}
    ======================

    Pertanyaan: {standalone_question}

    Jawaban (ikuti SEMUA ketentuan di atas):
    """

    try:
        response_stream = llm.stream(prompt)
        logging.info("Berhasil memulai streaming respons.")
        return response_stream, citations, intent
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 429:
            # Rate limit error
            logging.error("Rate limit tercapai pada API LLM")
            error_msg = "⏳ Kapasitas AI sedang penuh. Silakan coba lagi dalam beberapa saat. Mohon maaf atas ketidaknyamanannya."
            return None, error_msg, "rate_limit"
        elif e.response.status_code == 401:
            logging.error("Authentication error pada API LLM")
            error_msg = "❌ Terjadi kesalahan autentikasi. Silakan hubungi administrator."
            return None, error_msg, "auth_error"
        elif e.response.status_code >= 500:
            logging.error(f"Server error pada API LLM: {e.response.status_code}")
            error_msg = "❌ Server AI sedang mengalami gangguan. Silakan coba lagi nanti."
            return None, error_msg, "server_error"
        else:
            logging.error(f"HTTP error pada API LLM: {e.response.status_code}")
            error_msg = f"❌ Terjadi kesalahan (kode: {e.response.status_code}). Silakan coba lagi."
            return None, error_msg, "http_error"
    
    except Exception as e:
        logging.error(f"Error selama generasi LLM: {e}", exc_info=True)
        error_msg = "❌ Terjadi kesalahan saat menghasilkan jawaban. Silakan coba lagi atau hubungi administrator jika masalah berlanjut."
        return None, error_msg, "error"
