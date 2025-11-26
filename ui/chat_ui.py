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

    st.title("📊 Chatbot BPS Kota Bandung, Melesat👆")
    st.caption("Asisten AI untuk mencari tabel statistik (publikasi, berita, BRS, dll sedang development)")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Tampilkan riwayat pesan
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"], unsafe_allow_html=True)

    if user_input := st.chat_input("Tanyakan sesuatu..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Mencari jawaban..."):
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
                full_response = sources  # Menangani error atau info
                st.markdown(full_response)

        # Pemrosesan akhir untuk sitasi
        final_content = full_response
        if sources and isinstance(sources, list) and intent not in ["other", "blocked", "error", "retrieval_fail"]:
            # Temukan semua ID sumber
            found_ids = sorted(list(set([int(i) for i in re.findall(r'\[(\d+)\]', full_response)])))

            cited_sources = [src for src in sources if src['id'] in found_ids]

            if cited_sources:
                # Pemetaan ID lama ke ID baru
                id_map = {old_id: new_id + 1 for new_id, old_id in enumerate(found_ids)}

                # Ganti nomor sitasi menggunakan map
                temp_content = full_response
                for old_id, new_id in id_map.items():
                    temp_content = re.sub(r'\[\s*' + str(old_id) + r'\s*\]', f" <sup><a href='#source-{new_id}' style='text-decoration: none;'>[{new_id}]</a></sup>", temp_content)
                
                final_content = temp_content
                
                # Daftar sumber dengan ID yang ditambahkan
                final_content += "\n\n---\n**Sumber Rujukan:**\n"
                for idx, source in enumerate(cited_sources):
                    old_id = source['id']
                    new_id = id_map[old_id]
                    title = source.get("title", "Sumber Tanpa Judul")
                    link = source.get("link", "#")
                    final_content += f'<div id="source-{new_id}">{new_id}. <a href="{link}" target="_blank">{title}</a></div>\n'
            else:
                final_content = full_response

        st.session_state.messages.append({"role": "assistant", "content": final_content})
        st.rerun()


