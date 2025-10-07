import json
import torch
from sentence_transformers import SentenceTransformer
from google.colab import drive
import os

# --- Configuration ---
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "google/embeddinggemma-300m"
batch_size = 64

# Define file paths
input_filepath = "/content/drive/MyDrive/qwe/general_chunked.json"
output_filepath = "/content/drive/MyDrive/qwe/general_embed_final.json"

# 1. Load the model
try:
    # Set model to the correct device
    model = SentenceTransformer(model_id, device=device)
    print(f"Model '{model_id}' loaded successfully on device: {device}")
except Exception as e:
    print(f"Error loading the model: {e}")
    exit()

# 2. Load the JSON data
try:
    with open(input_filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"JSON file loaded successfully from {input_filepath}")
except Exception as e:
    print(f"Error loading JSON data: {e}")
    data = []

# 3. Prepare texts for efficient batch embedding
# Texts for document indexing (NO 'query:' prefix for embeddinggemma-300m)
texts_to_embed = [item.get('page_content', '') for item in data if item.get('page_content')]
data_with_content = [item for item in data if item.get('page_content')] # Keep only valid items

print(f"Prepared {len(texts_to_embed)} items for embedding.")

# 4. Batch Embed the data 🚀
embedded_data = []

if texts_to_embed:
    try:
        print(f"Starting batch embedding with batch size {batch_size} on {device}...")

        # model.encode returns a NumPy array by default (convert_to_tensor=False)
        embeddings = model.encode(
            texts_to_embed,
            convert_to_tensor=False,
            show_progress_bar=True,
            batch_size=batch_size,
            normalize_embeddings=True # Best practice for vector similarity
        )

        # 5. Add embeddings back to the flattened items
        for item, embedding in zip(data_with_content, embeddings):
            # Convert NumPy array to list for JSON serialization
            item['embedding'] = embedding.tolist()
            embedded_data.append(item)

        print(f"\nSuccessfully embedded {len(embedded_data)} items.")

    except Exception as e:
        print(f"Error during batch encoding: {e}")
        embedded_data = []

# 6. Save the new JSON file with proper indentation
if embedded_data:
    with open(output_filepath, "w", encoding="utf-8") as f:
        # Use indent=4 to format the JSON result with line breaks and indentation
        json.dump(embedded_data, f, indent=4)
    print(f"Embeddings saved to {output_filepath}")
else:
    print("No data to save. Check for errors in the previous steps.")