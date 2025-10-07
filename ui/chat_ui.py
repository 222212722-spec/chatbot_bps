import streamlit as st
from utils.rag_chain import rag_answer
import re

def render_chat_ui():
    st.set_page_config(page_title="BPS Chatbot", page_icon="📊", layout="wide")

    st.markdown(
        """
        <style>
        .stChatMessage {border-radius: 12px; padding: 16px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);}
        .stChatMessage[data-testid="user"] {background-color: #e3f2fd;}
        .stChatMessage[data-testid="assistant"] {background-color: #f1f8ff;}
        .stSpinner > div {text-align: left; align-items: flex-start;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("📊 Chatbot BPS Kota Bandung")
    st.caption("Asisten AI untuk mencari tabel statistik, publikasi, berita, BRS, dan informasi umum BPS Kota Bandung.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Tampilkan riwayat pesan. Ini adalah kunci untuk memperbaiki superscript.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            # --- PERBAIKAN UTAMA UNTUK SUPERSCRIPT ---
            st.markdown(message["content"], unsafe_allow_html=True)

    if user_input := st.chat_input("Tanyakan sesuatu..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Mencari jawaban..."):
                # Gunakan Mistral sebagai default untuk jawaban berkualitas tinggi
                response_stream, sources, intent = rag_answer(user_input, st.session_state.messages, provider="mistral")
            
            placeholder = st.empty()
            full_response = ""
            
            if response_stream:
                for chunk in response_stream:
                    if hasattr(chunk, 'content') and chunk.content is not None:
                        full_response += chunk.content
                        placeholder.markdown(full_response + "▌", unsafe_allow_html=True)
                placeholder.markdown(full_response, unsafe_allow_html=True)
            else:
                full_response = sources # Ini menampung pesan error atau info
                st.markdown(full_response)

        # Pemrosesan akhir untuk sitasi setelah streaming selesai
        final_content = full_response
        if sources and isinstance(sources, list) and intent not in ["other", "blocked", "error", "retrieval_fail"]:

            # 1. Temukan semua ID sumber asli yang dikutip dalam teks.
            # Contoh: jika teksnya "... [3] ... [1] ...", hasilnya adalah [3, 1]
            found_ids = sorted(list(set([int(i) for i in re.findall(r'\[(\d+)\]', full_response)])))

            # 2. Filter daftar 'sources' untuk hanya menyimpan yang benar-benar dikutip.
            cited_sources = [src for src in sources if src['id'] in found_ids]

            # Jika ada sumber yang valid untuk dikutip, lanjutkan.
            if cited_sources:
                # 3. Buat pemetaan dari ID lama ke ID baru (mulai dari 1).
                # Contoh: jika found_ids adalah [3, 5], map akan menjadi {3: 1, 5: 2}
                id_map = {old_id: new_id + 1 for new_id, old_id in enumerate(found_ids)}

                # 4. Ganti nomor sitasi di dalam teks jawaban menggunakan map.
                # Ini akan mengubah "... [3] ..." menjadi "... [1] ..."
                temp_content = full_response
                for old_id, new_id in id_map.items():
                    temp_content = re.sub(r'\[\s*' + str(old_id) + r'\s*\]', f" <sup><a href='#' style='text-decoration: none;'>[{new_id}]</a></sup>", temp_content)
                
                final_content = temp_content
                
                # 5. Buat daftar sumber rujukan akhir dengan nomor baru.
                final_content += "\n\n---\n**Sumber Rujukan:**\n"
                for source in cited_sources:
                    old_id = source['id']
                    new_id = id_map[old_id]
                    title = source.get("title", "Sumber Tanpa Judul")
                    link = source.get("link", "#")
                    final_content += f'{new_id}. <a href="{link}" target="_blank">{title}</a>\n'
            # Jika tidak ada sumber yang dikutip, cukup tampilkan jawaban aslinya.
            else:
                final_content = full_response


        st.session_state.messages.append({"role": "assistant", "content": final_content})
        st.rerun()


