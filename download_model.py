# download_model.py
# Script untuk download model dari Google Drive
# Dijalankan SEKALI saat setup di PythonAnywhere

import os
import gdown

# Ganti dengan File ID dari Google Drive kamu
# Cara dapat File ID:
# 1. Buka Google Drive
# 2. Klik kanan fase2_best.keras → Share → Copy link
# 3. Link bentuknya: drive.google.com/file/d/FILE_ID/view
# 4. Salin FILE_ID nya

FASE2_FILE_ID    = "https://drive.google.com/file/d/1cQ6wGGmNt-Ld9GNfkMzp-Iq6gWCDxszv/view?usp=sharing"
EMBEDDING_FILE_ID = "https://drive.google.com/file/d/11CJAZuZT8UNOpT_pVKKTLWy6WOrmo7Yf/view?usp=sharing"

os.makedirs("models", exist_ok=True)

print("Downloading fase2_best.keras...")
gdown.download(
    f"https://drive.google.com/uc?id={FASE2_FILE_ID}",
    "models/fase2_best.keras",
    quiet=False
)

print("Downloading embedding_reference.json...")
gdown.download(
    f"https://drive.google.com/uc?id={EMBEDDING_FILE_ID}",
    "models/embedding_reference.json",
    quiet=False
)

print("✅ Semua model berhasil didownload.")