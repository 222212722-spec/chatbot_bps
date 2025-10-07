import os
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    Collection,
    DataType,
)

# --- 1. Detail Koneksi ke Zilliz Cloud (ISI DENGAN KREDENSIAL ANDA SENDIRI) ---
ZILLIZ_ENDPOINT = "XXX"  # Ganti dengan endpoint dari Zilliz Cloud
ZILLIZ_TOKEN = "XXX"      # Ganti dengan API Token kamu

# Dimensi vektor hasil embedding (harus sama dengan model yang digunakan)
DIMENSION = 768

# Batas panjang maksimum untuk tipe teks
MAX_VARCHAR_LENGTH = 512
MAX_ID_LENGTH = 200

# --- 2. Menghubungkan ke Zilliz Cloud ---
try:
    connections.connect(
        alias="default",
        uri=ZILLIZ_ENDPOINT,
        token=ZILLIZ_TOKEN
    )
    print("✅ Berhasil terhubung ke Zilliz Cloud.")
except Exception as e:
    print(f"❌ Gagal terhubung ke Zilliz Cloud: {e}")
    # exit()  # Jika ingin langsung keluar saat gagal koneksi, hilangkan tanda komentar ini

# --- 3. Skema Koleksi untuk Data Informasi Umum (general) ---
COLLECTION_NAME = "general"

# Daftar field dalam koleksi
FIELDS = [
    # Primary Key: chunk_id (unik untuk setiap potongan teks)
    FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=MAX_ID_LENGTH),

    # Field vektor untuk embedding
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=DIMENSION),

    # Konten teks utama
    FieldSchema(name="page_content", dtype=DataType.VARCHAR, max_length=65535),

    # Metadata
    FieldSchema(name="service_id", dtype=DataType.VARCHAR, max_length=MAX_ID_LENGTH),
    FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=MAX_VARCHAR_LENGTH),
    FieldSchema(name="menu", dtype=DataType.VARCHAR, max_length=MAX_VARCHAR_LENGTH),
    FieldSchema(name="link", dtype=DataType.VARCHAR, max_length=MAX_VARCHAR_LENGTH * 2),
]

# --- 4. Membuat Koleksi ---
print(f"\n--- Proses pembuatan koleksi '{COLLECTION_NAME}' dimulai ---")

if utility.has_collection(COLLECTION_NAME):
    print(f"⚠️ Koleksi '{COLLECTION_NAME}' sudah ada. Melewati proses pembuatan.")
else:
    # Membuat skema
    schema = CollectionSchema(
        fields=FIELDS,
        description="Koleksi vektor untuk data informasi umum BPS (general information).",
        enable_dynamic_field=True
    )

    # Membuat koleksi di Zilliz
    collection = Collection(
        name=COLLECTION_NAME,
        schema=schema,
        using="default"
    )

    # Membuat indeks vektor (otomatis, berbasis cosine similarity)
    index_params = {
        "index_type": "AUTOINDEX",
        "metric_type": "COSINE"
    }

    collection.create_index(
        field_name="embedding",
        index_params=index_params
    )

    # Memuat ke memori (diperlukan agar bisa dicari)
    collection.load()

    print(f"✅ Koleksi '{COLLECTION_NAME}' berhasil dibuat dan dimuat ke memori.")

print("\n✔️ Verifikasi pembuatan koleksi selesai.")
