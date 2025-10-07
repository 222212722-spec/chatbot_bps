import os
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    Collection,
    DataType,
)

# --- 1. Zilliz Cloud Connection Details (Use your details) ---
ZILLIZ_ENDPOINT = "XXX"
ZILLIZ_TOKEN = "XXX"
DIMENSION = 768  
MAX_VARCHAR_LENGTH = 512
MAX_ID_LENGTH = 200 

# --- 2. Connect to Zilliz Cloud ---
try:
    connections.connect(
        alias="default",
        uri=ZILLIZ_ENDPOINT,
        token=ZILLIZ_TOKEN
    )
    print("Successfully connected to Zilliz Cloud.")
except Exception as e:
    print(f"Failed to connect to Zilliz Cloud: {e}")
    # exit() 

# --- 3. Base Fields (Common to ALL collections) ---
BASE_FIELDS = [
    # Primary Key: unique (ensures global uniqueness across all tables)
    FieldSchema(name="unique", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=MAX_ID_LENGTH),
    # Vector Field: embedding
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=DIMENSION),
    # Text Content for context/search (The combined text used to generate the embedding)
    FieldSchema(name="page_content", dtype=DataType.VARCHAR, max_length=65535),
    # Title and Link metadata
    FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=MAX_VARCHAR_LENGTH * 2),
    FieldSchema(name="link", dtype=DataType.VARCHAR, max_length=MAX_VARCHAR_LENGTH),
]

# --- 4. Define Specific Schema Fields for 'tables' Collection ---
COLLECTION_NAME = "tables"
CONFIG = {
    "description": "Vector collection for all BPS statistical tables (Static, Dynamic, and Simdasi).",
    "fields": [
        # Metadata from alltables.json:
        FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=MAX_ID_LENGTH), 
        FieldSchema(name="id_table", dtype=DataType.VARCHAR, max_length=MAX_ID_LENGTH), 
        FieldSchema(name="id_subcat", dtype=DataType.INT64), # Numerical ID 
        FieldSchema(name="subcat", dtype=DataType.VARCHAR, max_length=MAX_VARCHAR_LENGTH),
        FieldSchema(name="tablesource", dtype=DataType.VARCHAR, max_length=10), # 1, 2, atau 3
        FieldSchema(name="last_update", dtype=DataType.VARCHAR, max_length=50), 
        # Field 'years' stored as VARCHAR/JSON string, as requested
        FieldSchema(name="years", dtype=DataType.VARCHAR, max_length=MAX_VARCHAR_LENGTH * 2), 
    ]
}

# --- 5. Main Script Logic: Create Collection ---
print(f"\n--- Processing '{COLLECTION_NAME}' collection ---")

# Check if the collection already exists
if utility.has_collection(COLLECTION_NAME):
    print(f"Collection '{COLLECTION_NAME}' already exists. Skipping creation.")
else:
    # Combine base fields and specific fields
    all_fields = BASE_FIELDS + CONFIG["fields"]

    # Create the schema for the collection
    schema = CollectionSchema(
        fields=all_fields,
        description=CONFIG["description"],
        enable_dynamic_field=True 
    )

    # Create the collection
    collection = Collection(
        name=COLLECTION_NAME,
        schema=schema,
        using="default"
    )

    # Define and Create the vector index
    index_params = {
        "index_type": "AUTOINDEX", 
        "metric_type": "COSINE"      
    }
    collection.create_index(
        field_name="embedding",
        index_params=index_params
    )
    
    # Load the collection into memory (required for searching)
    collection.load()

    print(f"Successfully created and loaded collection '{COLLECTION_NAME}' with a vector index.")

print("\nCollection creation verified.")