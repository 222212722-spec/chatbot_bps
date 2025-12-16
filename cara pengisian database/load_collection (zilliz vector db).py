import json
import os
from pathlib import Path
from pymilvus import connections, utility, Collection, FieldSchema, CollectionSchema, DataType

# --- 1. Detail Koneksi Zilliz Cloud (Gunakan detail Anda sendiri) ---
ZILLIZ_ENDPOINT = "https://in03-xxx.serverless.gcp-us-west1.cloud.zilliz.com"
ZILLIZ_TOKEN = "xxx"
COLLECTION_NAME = "tables"

# Pastikan dimensi ini sama dengan dimensi embedding pada script embedding Anda
DIMENSION = 768

# --- 2. Definisi Path File Input dan Ukuran Batch ---
input_filepath = Path(r"C:\qwe\database\alltables_embedded_gemma.json")
BATCH_SIZE = 1000  # Sesuaikan dengan ukuran data dan kondisi jaringan Anda

# --- 3. Koneksi ke Zilliz Cloud ---
try:
    connections.connect(
        alias="default",
        uri=ZILLIZ_ENDPOINT,
        token=ZILLIZ_TOKEN
    )
    print("Berhasil terhubung ke Zilliz Cloud.")
except Exception as e:
    print(f"Gagal terhubung ke Zilliz Cloud: {e}")
    exit()

# --- 4a. Pengecekan dan Penghapusan Koleksi Lama ---
print(f"Memeriksa keberadaan koleksi '{COLLECTION_NAME}'...")
if utility.has_collection(COLLECTION_NAME):
    # Menghapus koleksi yang sudah ada
    utility.drop_collection(COLLECTION_NAME)
    print(f"Koleksi lama '{COLLECTION_NAME}' berhasil dihapus.")

# --- 4b. Definisi Skema Koleksi (Wajib untuk koleksi baru) ---
# Skema harus sesuai dengan urutan dan tipe data pada batch_data
fields = [
    FieldSchema(
        name="unique",
        dtype=DataType.VARCHAR,
        max_length=128,
        is_primary=True,
        auto_id=False,
        description="ID utama untuk menjamin keunikan data"
    ),
    FieldSchema(
        name="embedding",
        dtype=DataType.FLOAT_VECTOR,
        dim=DIMENSION,
        description="Vektor embedding Gemma"
    ),
    # Perhatian: Milvus/Zilliz memiliki batasan tipe VARCHAR/JSON untuk metadata
    FieldSchema(
        name="page_content",
        dtype=DataType.VARCHAR,
        max_length=2048,
        description="Konten teks utama (Subkategori + Judul)"
    ),
    FieldSchema(
        name="title",
        dtype=DataType.VARCHAR,
        max_length=512,
        description="Judul asli"
    ),
    FieldSchema(
        name="link",
        dtype=DataType.VARCHAR,
        max_length=512,
        description="Tautan asli"
    ),
    FieldSchema(
        name="id",
        dtype=DataType.VARCHAR,
        max_length=128,
        description="ID asli (bukan primary key)"
    ),
    FieldSchema(
        name="id_table",
        dtype=DataType.VARCHAR,
        max_length=128,
        description="ID tabel"
    ),
    FieldSchema(
        name="id_subcat",
        dtype=DataType.INT64,
        description="ID subkategori"
    ),
    FieldSchema(
        name="subcat",
        dtype=DataType.VARCHAR,
        max_length=256,
        description="Nama subkategori"
    ),
    FieldSchema(
        name="tablesource",
        dtype=DataType.VARCHAR,
        max_length=128,
        description="Sumber tabel"
    ),
    FieldSchema(
        name="last_update",
        dtype=DataType.VARCHAR,
        max_length=128,
        description="Tanggal pembaruan terakhir"
    ),
    FieldSchema(
        name="years",
        dtype=DataType.VARCHAR,
        max_length=256,
        description="Daftar tahun dalam format JSON string"
    ),
]

schema = CollectionSchema(
    fields,
    description=f"Metadata tabel RAG untuk koleksi {COLLECTION_NAME}"
)

# --- 4c. Membuat Koleksi Baru ---
collection = Collection(
    name=COLLECTION_NAME,
    schema=schema,
    using="default",  # Menggunakan koneksi default
    shards_num=1      # Sesuaikan jika jumlah data sangat besar
)

# --- 4d. Membuat Index pada Field Vector (Wajib) ---
index_params = {
    "index_type": "IVF_FLAT",  # Jenis index yang umum dan cepat
    "metric_type": "COSINE",   # COSINE digunakan karena embedding dinormalisasi
    "params": {"nlist": 6823}  # Sesuaikan dengan jumlah data
}

collection.create_index(
    field_name="embedding",
    index_params=index_params
)
print("Koleksi baru berhasil dibuat dan index telah dibangun.")

# --- 5. Memuat Koleksi Zilliz ---
if not utility.has_collection(COLLECTION_NAME):
    print(f"Koleksi '{COLLECTION_NAME}' tidak ditemukan. Jalankan script pembuatan koleksi terlebih dahulu.")
    exit()

collection = Collection(COLLECTION_NAME)
print(f"Koleksi '{COLLECTION_NAME}' berhasil dimuat.")

# --- 6. Memuat Data Embedding dari File JSON ---
try:
    with open(input_filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Data JSON berhasil dimuat dari {input_filepath}")
except FileNotFoundError:
    print(f"Error: File '{input_filepath}' tidak ditemukan.")
    exit()
except json.JSONDecodeError:
    print(f"Error: File '{input_filepath}' bukan file JSON yang valid.")
    exit()

# --- 7. Menyiapkan dan Memasukkan Data Secara Bertahap (Batch) ---
print(f"Memulai proses insert batch ke koleksi '{COLLECTION_NAME}'...")

total_records = len(data)
inserted_count = 0

for i in range(0, total_records, BATCH_SIZE):
    batch = data[i:i + BATCH_SIZE]

    # Ekstraksi field untuk proses insert batch
    batch_data = [
        [record["unique"] for record in batch],
        [record["embedding"] for record in batch],
        [record["page_content"] for record in batch],
        [record.get("title", "") for record in batch],
        [record.get("link", "") for record in batch],
        [record.get("id", "") for record in batch],
        [str(record.get("id_table", "")) for record in batch],
        [record.get("id_subcat", 0) for record in batch],
        [record.get("subcat", "") for record in batch],
        [record.get("tablesource", "") for record in batch],
        [record.get("last_update", "") for record in batch],
        [json.dumps(record.get("years", [])) for record in batch]
    ]

    try:
        # Insert data batch ke koleksi
        collection.insert(batch_data)
        inserted_count += len(batch)
        print(f"Batch {i // BATCH_SIZE + 1} berhasil diinsert. Total: {inserted_count}/{total_records}")
    except Exception as e:
        print(f"Terjadi error saat insert batch mulai indeks {i}: {e}")
        # Lanjut ke batch berikutnya agar proses tidak berhenti total

# Memastikan data dikirim ke server dan index diperbarui
collection.flush()
print("\nProses insert batch selesai. Data berhasil disimpan di Zilliz Cloud.")
