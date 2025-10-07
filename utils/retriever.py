from vectorstores.zilliz_client import get_collection
import logging

def retrieve_from_zilliz(query_vector, collection_name: str, top_k=5):
    """
    Mengambil dokumen dari Zilliz.
    """
    collection = get_collection(collection_name)
    
    # Output fields yang ada di seluruh koleksi
    output_fields = [
        "page_content", "chunk_id", "link", "title", "menu", "rl_date", 
        "pub_id", "brs_id", "news_id", "service_id", "abstract",
        "unique", "id", "id_table", "id_subcat", "subcat", "tablesource", "years", "last_update"
    ]
    
    logging.info(f"Mencari di koleksi '{collection_name}' dengan top_k={top_k}")
    
    results = collection.search(
        data=[query_vector],
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"nprobe": 15}},
        limit=top_k,
        output_fields=output_fields
    )
    logging.info(f"Pencarian Zilliz mengembalikan {len(results[0]) if results else 0} hasil.")
    return results

