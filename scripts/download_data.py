import os
import requests
import zipfile
from pathlib import Path

def download_esc50():
    """Download ESC-50 dataset"""
    url = "https://github.com/karolpiczak/ESC-50/archive/master.zip"
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Download dataset
    print("Downloading ESC-50 dataset...")
    response = requests.get(url)
    with open(data_dir / "esc50.zip", "wb") as f:
        f.write(response.content)
    
    # Extract dataset
    with zipfile.ZipFile(data_dir / "esc50.zip", 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    
    print("Dataset downloaded and extracted!")

if __name__ == "__main__":
    download_esc50()