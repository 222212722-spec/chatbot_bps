import json
import os
from pathlib import Path
from pymilvus import connections, utility, Collection, FieldSchema, CollectionSchema, DataType

# --- 1. Zilliz Cloud Connection Details (Use your details) ---
ZILLIZ_ENDPOINT = "XXX"
ZILLIZ_TOKEN = "XXX"
COLLECTION_NAME = "tables"
DIMENSION = 768

# --- 2. Define File Paths and Batch Size ---
input_filepath = Path(r"C:\qwe\database\alltables_embed.json")
BATCH_SIZE = 1000 

# --- 3. Connect to Zilliz Cloud ---
try:
    connections.connect(
        alias="default",
        uri=ZILLIZ_ENDPOINT,
        token=ZILLIZ_TOKEN
    )
    print("Successfully connected to Zilliz Cloud.")
except Exception as e:
    print(f"Failed to connect to Zilliz Cloud: {e}")
    exit()

# --- 4. Load the Zilliz Collection ---
if not utility.has_collection(COLLECTION_NAME):
    print(f"Collection '{COLLECTION_NAME}' does not exist. Please run the collection creation script first.")
    exit()

collection = Collection(COLLECTION_NAME)
print(f"Collection '{COLLECTION_NAME}' loaded successfully.")

# --- 5. Load the Embedded Data ---
try:
    with open(input_filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"JSON data loaded successfully from {input_filepath}")
except FileNotFoundError:
    print(f"Error: The file '{input_filepath}' was not found.")
    exit()
except json.JSONDecodeError:
    print(f"Error: The file '{input_filepath}' is not a valid JSON file.")
    exit()

# --- 6. Prepare and Insert Data in Batches ---
print(f"Starting batch insertion into '{COLLECTION_NAME}'...")

total_records = len(data)
inserted_count = 0

for i in range(0, total_records, BATCH_SIZE):
    batch = data[i:i + BATCH_SIZE]
    
    # Extract fields for batch insertion
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
        # Insert the batch
        collection.insert(batch_data)
        inserted_count += len(batch)
        print(f"Inserted batch {i // BATCH_SIZE + 1}. Total inserted: {inserted_count}/{total_records}")
    except Exception as e:
        print(f"Error inserting batch starting at index {i}: {e}")
        # Continue to the next batch to not fail the entire process

# Ensure data is flushed to the server and indexed
collection.flush()
print("\nBatch insertion complete. Data flushed to Zilliz Cloud.")