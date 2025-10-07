import json
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken  # Library untuk menghitung jumlah token
import os

# --- 1. PENGATURAN FUNGSI PERHITUNGAN PANJANG TOKENS DENGAN TIKTOKEN ---
# Ini memastikan bahwa chunk_size diukur dalam satuan token (cl100k_base),
# yang merupakan standar untuk banyak model LLM modern dan LangChain.
tiktoken_encoder = tiktoken.get_encoding("cl100k_base")

def tiktoken_len(text):
    """Menghitung jumlah token dalam sebuah string."""
    return len(tiktoken_encoder.encode(text))

def chunk_general_data(input_file_path: Path, output_file_path: Path):
    """
    Membaca file JSON, memecah konten 'full_text' menjadi potongan (chunk)
    menggunakan pembagi berbasis token, dan menambahkan konteks lengkap pada setiap potongan.
    Dioptimalkan untuk data informasi umum.
    """
    try:
        with open(input_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"Data berhasil dimuat dari: {input_file_path}")
    except FileNotFoundError:
        print(f"Kesalahan: File '{input_file_path}' tidak ditemukan.")
        return
    except json.JSONDecodeError:
        print(f"Kesalahan: File '{input_file_path}' bukan file JSON yang valid.")
        return

    # --- 2. KONFIGURASI PEMBAGI TEKS UNTUK CHUNK LEBIH PANJANG ---
    # Kita menggunakan RecursiveCharacterTextSplitter standar dengan fungsi panjang berbasis token,
    # serta pemisah khusus untuk menurunkan prioritas '\n\n' dibanding tanda baca lainnya.
    text_splitter = RecursiveCharacterTextSplitter(
        # Ukuran chunk diperbesar untuk mempertahankan kepadatan makna yang lebih tinggi
        chunk_size=120,
        chunk_overlap=30,  # Overlap standar agar alur antar chunk tetap mulus
        length_function=tiktoken_len,  # Menggunakan fungsi panjang berbasis token
        # Pemisah kustom agar chunk_size tercapai sebelum jeda paragraf:
        separators=[
            ".",   # Akhir kalimat
            "!",   # Akhir kalimat
            "?",   # Akhir kalimat
            "\n\n",  # Jeda dua baris (paragraf) – prioritas lebih rendah
            "\n",   # Jeda satu baris – prioritas sedang
            ":",
            " ",   # Spasi
            "",    # Karakter tunggal
        ],
    )

    chunked_records = []

    # Memproses setiap data informasi umum
    for record in data:
        full_text_content = record.get("full_text")
        if not isinstance(full_text_content, str) or not full_text_content.strip():
            print(f"Melewati record dengan ID {record.get('service_id')}: kolom 'full_text' kosong atau tidak valid.")
            continue

        # 1. Membagi teks mentah terlebih dahulu.
        # Dengan konfigurasi di atas, hasil chunk akan sedikit lebih panjang.
        chunks = text_splitter.split_text(full_text_content)

        # 2. Menambahkan konteks ke setiap potongan teks.
        for i, chunk in enumerate(chunks):
            # Membersihkan karakter titik di awal (jika ada)
            cleaned_chunk = chunk.lstrip('.')
            # Membuat string dengan konteks yang lengkap
            page_content_with_context = (
                f"Menu: {record.get('menu', '')}. "
                f"Title: {record.get('title', '')}. "
                f"Part {i + 1}/{len(chunks)}: {cleaned_chunk}"
            )

            new_record = {
                "chunk_id": f"{record.get('service_id')}_{i}",
                "page_content": page_content_with_context,  # Menggunakan string dengan konteks baru
                "metadata": {
                    "service_id": record.get("service_id"),
                    "title": record.get("title"),
                    "menu": record.get("menu"),
                    "link": record.get("link"),
                }
            }
            chunked_records.append(new_record)

    # --- 3. MENYIMPAN HASIL KE FILE OUTPUT ---
    if chunked_records:
        with open(output_file_path, "w", encoding="utf-8") as f:
            json.dump(chunked_records, f, indent=4, ensure_ascii=False)

        print(f"\nBerhasil memecah {len(data)} record menjadi {len(chunked_records)} potongan (chunk).")
        print(f"Rata-rata panjang token (tanpa konteks): {sum(tiktoken_len(r['page_content'].split(': ')[-1]) for r in chunked_records) / len(chunked_records):.2f}")
        print(f"Hasil disimpan di: {output_file_path}")
    else:
        print("\nTidak ada data yang berhasil diproses. Periksa kembali struktur file input Anda.")

if __name__ == "__main__":
    # CATATAN: Ubah path di bawah ini sesuai dengan lokasi file Anda
    input_file_path = Path(r"C:\qwe\database\general.json")
    output_file_path = Path(r"C:\qwe\database\general_chunked.json")
    chunk_general_data(input_file_path, output_file_path)
