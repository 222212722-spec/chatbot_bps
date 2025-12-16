import os
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    Collection,
    DataType,
)

# --- 1. Detail Koneksi Zilliz Cloud (Gunakan detail Anda sendiri) ---
ZILLIZ_ENDPOINT = "https://in03-xxx.serverless.gcp-us-west1.cloud.zilliz.com"
ZILLIZ_TOKEN = "xxx"
DIMENSION = 768  # Dimensi dari model embedding yang digunakan
MAX_VARCHAR_LENGTH = 512
MAX_ID_LENGTH = 200 

# --- 2. Koneksi ke Zilliz Cloud ---
try:
    connections.connect(
        alias="default",
        uri=ZILLIZ_ENDPOINT,
        token=ZILLIZ_TOKEN
    )
    print("Berhasil terhubung ke Zilliz Cloud.")
except Exception as e:
    print(f"Gagal terhubung ke Zilliz Cloud: {e}")
    # exit() 

# --- 3. Field Dasar (Digunakan oleh SEMUA koleksi) ---
BASE_FIELDS = [
    # Primary Key: unik (menjamin keunikan global di semua tabel)
    FieldSchema(
        name="unique",
        dtype=DataType.VARCHAR,
        is_primary=True,
        auto_id=False,
        max_length=MAX_ID_LENGTH
    ),
    # Field Vektor: embedding
    FieldSchema(
        name="embedding",
        dtype=DataType.FLOAT_VECTOR,
        dim=DIMENSION
    ),
    # Konten teks untuk konteks/pencarian
    # (Teks gabungan yang digunakan untuk menghasilkan embedding)
    FieldSchema(
        name="page_content",
        dtype=DataType.VARCHAR,
        max_length=65535
    ),
    # Metadata judul dan tautan
    FieldSchema(
        name="title",
        dtype=DataType.VARCHAR,
        max_length=MAX_VARCHAR_LENGTH * 2
    ),
    FieldSchema(
        name="link",
        dtype=DataType.VARCHAR,
        max_length=MAX_VARCHAR_LENGTH
    ),
]

# --- 4. Definisi Field Skema Khusus untuk Koleksi 'tables' ---
COLLECTION_NAME = "tables"
CONFIG = {
    "description": "Koleksi vektor untuk seluruh tabel statistik BPS (Statik, Dinamik, dan Simdasi).",
    "fields": [
        # Metadata dari alltables.json:
        FieldSchema(
            name="id",
            dtype=DataType.VARCHAR,
            max_length=MAX_ID_LENGTH
        ), 
        FieldSchema(
            name="id_table",
            dtype=DataType.VARCHAR,
            max_length=MAX_ID_LENGTH
        ), 
        FieldSchema(
            name="id_subcat",
            dtype=DataType.INT64
        ),  # ID numerik
        FieldSchema(
            name="subcat",
            dtype=DataType.VARCHAR,
            max_length=MAX_VARCHAR_LENGTH
        ),
        FieldSchema(
            name="tablesource",
            dtype=DataType.VARCHAR,
            max_length=10
        ),  # 1, 2, atau 3
        FieldSchema(
            name="last_update",
            dtype=DataType.VARCHAR,
            max_length=50
        ), 
        # Field 'years' disimpan sebagai VARCHAR / string JSON
        FieldSchema(
            name="years",
            dtype=DataType.VARCHAR,
            max_length=MAX_VARCHAR_LENGTH * 2
        ), 
    ]
}

# --- 5. Logika Utama Script: Membuat Koleksi ---
print(f"\n--- Memproses koleksi '{COLLECTION_NAME}' ---")

# Mengecek apakah koleksi sudah ada
if utility.has_collection(COLLECTION_NAME):
    print(f"Koleksi '{COLLECTION_NAME}' sudah ada. Pembuatan dilewati.")
else:
    # Menggabungkan field dasar dan field khusus
    all_fields = BASE_FIELDS + CONFIG["fields"]

    # Membuat skema koleksi
    schema = CollectionSchema(
        fields=all_fields,
        description=CONFIG["description"],
        enable_dynamic_field=True 
    )

    # Membuat koleksi
    collection = Collection(
        name=COLLECTION_NAME,
        schema=schema,
        using="default"
    )

    # Mendefinisikan dan membuat index vektor
    index_params = {
        "index_type": "AUTOINDEX", 
        "metric_type": "COSINE"      
    }
    collection.create_index(
        field_name="embedding",
        index_params=index_params
    )
    
    # Memuat koleksi ke memori (diperlukan untuk pencarian)
    collection.load()

    print(f"Berhasil membuat dan memuat koleksi '{COLLECTION_NAME}' dengan index vektor.")

print("\nPembuatan koleksi telah diverifikasi.")
