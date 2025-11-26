import logging
import time
from vectorstores.zilliz_client import get_collection

def retrieve_from_zilliz(query_vector, collection_name: str, top_k=3):
    """
    Mengambil dokumen dari Zilliz menggunakan vektor query.
    Default top_k=3 untuk mengambil 3 tabel teratas.
    """
    logging.info(f"Mencari di koleksi '{collection_name}' dengan top_k={top_k}")
    
    collection = get_collection(collection_name)

    output_fields = [
        "page_content", "chunk_id", "link", "title",
        # table-specific fields:
        "unique", "id", "id_table", "id_subcat", "subcat", 
        "tablesource", "years", "last_update"
    ]

    start_time = time.time()

    try:
        results = collection.search(
            data=[query_vector],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"nprobe": 15}},
            limit=top_k,
            output_fields=output_fields
        )

        elapsed_time = time.time() - start_time
        logging.info(f"Pencarian Zilliz selesai dalam {elapsed_time:.2f} detik.")

        if not results or len(results[0]) == 0:
            logging.warning("⚠️ Tidak ada hasil ditemukan dari Zilliz.")
            return [[]]

        # Log seluruh judul hasil
        titles = [hit.entity.get("title", "❓(tanpa judul)") for hit in results[0]]
        logging.info(f"Jumlah hasil: {len(titles)}")
        for i, t in enumerate(titles, start=1):
            logging.info(f"Hasil {i}: {t}")

        return results

    except Exception as e:
        logging.error(f"❌ Terjadi kesalahan saat pencarian Zilliz: {e}", exc_info=True)
        return [[]]
