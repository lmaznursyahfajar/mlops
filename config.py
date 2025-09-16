# config.py
import os
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class Config:
    # ==== Dataset Paths ====
    NEW_DATA_DIR = os.path.join(BASE_DIR, "data", "new")      # folder dataset baru
    MERGED_DIR = os.path.join(BASE_DIR, "data", "merged")     # folder dataset gabungan
    MAIN_DATA_FILE = "main_dataset.csv"                       # file utama gabungan
    DEFAULT_DATA_PATH = os.path.join(NEW_DATA_DIR, "universitas_indonesia_main_dataset.csv")

    # ==== Model Paths ====
    MODELS_TORCH_DIR = os.path.join(BASE_DIR, "models_torch")  # model Torch
    MODELS_ONNX_DIR = os.path.join(BASE_DIR, "models_onnx")    # model ONNX

    # ==== Logs & Metadata ====
    LOGS_DIR = os.path.join(BASE_DIR, "logs")
    LAST_HASH_FILE = os.path.join(LOGS_DIR, "last_hash.txt")
    LAST_ACCURACY_FILE = os.path.join(LOGS_DIR, "last_accuracy.txt")

    # ==== Default Config ====
    DEFAULT_MODEL_NAME = "main_dataset"

    # ==== Database ====
    DB_HOST = os.getenv("DB_HOST", "146.190.99.120")
    DB_USER = os.getenv("DB_USER", "absa_user")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "PasswordKuat123!")
    DB_DATABASE = os.getenv("DB_DATABASE", "absa_dummy")

    # ==== Utility ====
    @staticmethod
    def timestamp():
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# Inisialisasi global config
config = Config()
