import os
import requests
import zipfile
from tqdm import tqdm

DATASET_URL = "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/region_descriptions.json.zip"
ZIP_PATH = "dataset/region_descriptions.json.zip"
EXTRACT_PATH = "dataset/region_descriptions.json"
DATASET_DIR = "dataset"

def download_file(url, filename):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024 * 1024 # 1 Megabyte
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True, desc="Downloading")
    with open(filename, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()

def generate_dataset():
    os.makedirs(DATASET_DIR, exist_ok=True)
    
    if os.path.exists(EXTRACT_PATH):
        print(f"Dataset already exists at {EXTRACT_PATH}.")
        return

    print(f"Downloading Visual Genome dataset from {DATASET_URL}...")
    download_file(DATASET_URL, ZIP_PATH)

    print("Extracting zip file...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        # Extract files specifically handling if it's nested or not
        zip_ref.extractall(DATASET_DIR)
        
    print("Cleaning up zip file...")
    if os.path.exists(ZIP_PATH):
        os.remove(ZIP_PATH)
        
    print(f"Actual dataset successfully generated at {EXTRACT_PATH}")

if __name__ == "__main__":
    generate_dataset()
