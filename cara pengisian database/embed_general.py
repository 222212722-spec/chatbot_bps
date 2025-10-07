import json
import torch
from sentence_transformers import SentenceTransformer
from google.colab import drive
import os

# --- KONFIGURASI ---
device = "cuda" if torch.cuda.is_available() else "cpu"  # Gunakan GPU jika tersedia
model_id = "google/embeddinggemma-300m"
batch_size = 64  # Jumlah data per batch (semakin besar = semakin cepat, tapi butuh RAM/GPU lebih besar)

# Tentukan lokasi file input dan output
input_filepath = "/content/drive/MyDrive/qwe/general_chunked.json"
output_filepath = "/content/drive/MyDrive/qwe/general_embed.json"

# --- 1. MEMUAT MODEL ---
try:
    # Menempatkan model ke perangkat (GPU/CPU)
    model = SentenceTransformer(model_id, device=device)
    print(f"Model '{model_id}' berhasil dimuat pada perangkat: {device}")
except Exception as e:
    print(f"Terjadi kesalahan saat memuat model: {e}")
    exit()

# --- 2. MEMUAT DATA JSON ---
try:
    with open(input_filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"File JSON berhasil dimuat dari {input_filepath}")
except Exception as e:
    print(f"Terjadi kesalahan saat memuat file JSON: {e}")
    data = []

# --- 3. MENYIAPKAN TEKS UNTUK PROSES EMBEDDING ---
# Teks yang akan dikonversi menjadi vektor (tanpa prefix 'query:')
texts_to_embed = [item.get('page_content', '') for item in data if item.get('page_content')]
# Hanya ambil item yang memiliki konten valid
data_with_content = [item for item in data if item.get('page_content')]

print(f"Berhasil menyiapkan {len(texts_to_embed)} item untuk proses embedding.")

# --- 4. MELAKUKAN PROSES EMBEDDING SECARA BATCH 🚀 ---
embedded_data = []

if texts_to_embed:
    try:
        print(f"Memulai proses embedding dengan batch size {batch_size} pada perangkat {device}...")

        # Fungsi encode menghasilkan array NumPy (convert_to_tensor=False)
        embeddings = model.encode(
            texts_to_embed,
            convert_to_tensor=False,
            show_progress_bar=True,
            batch_size=batch_size,
            normalize_embeddings=True  # Praktik terbaik untuk menjaga konsistensi jarak vektor
        )

        # --- 5. MENGGABUNGKAN EMBEDDING KE DATA ASLI ---
        for item, embedding in zip(data_with_content, embeddings):
            # Konversi array NumPy ke list agar bisa disimpan dalam format JSON
            item['embedding'] = embedding.tolist()
            embedded_data.append(item)

        print(f"\nBerhasil membuat embedding untuk {len(embedded_data)} item.")

    except Exception as e:
        print(f"Terjadi kesalahan selama proses embedding: {e}")
        embedded_data = []

# --- 6. MENYIMPAN HASIL KE FILE JSON BARU ---
if embedded_data:
    with open(output_filepath, "w", encoding="utf-8") as f:
        # Gunakan indent=4 agar hasil JSON lebih mudah dibaca
        json.dump(embedded_data, f, indent=4)
    print(f"Embedding berhasil disimpan ke {output_filepath}")
else:
    print("Tidak ada data yang disimpan. Periksa kemungkinan kesalahan pada langkah sebelumnya.")
