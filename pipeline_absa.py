import hashlib
import logging
import os
from datetime import datetime

import nltk
import pandas as pd
import torch

from modules.model_train_module import ABSAModelTrainer
from register_model import register_model
from config import config

# == SETUP LOGGING ==
os.makedirs(config.LOGS_DIR, exist_ok=True)
logger = logging.getLogger("pipeline")
logger.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

# log ke file
fh = logging.FileHandler(os.path.join(config.LOGS_DIR, "pipeline.log"))
fh.setFormatter(formatter)
logger.addHandler(fh)

# log ke terminal
sh = logging.StreamHandler()
sh.setFormatter(formatter)
logger.addHandler(sh)

# == CEK RESOURCE NLTK ==
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# == UTILS: CSV READER & HASHING ==
def safe_read_csv(path):
    encodings = ["utf-8", "utf-8-sig", "latin1"]
    separators = [",", ";", "\t"]
    for encoding in encodings:
        for sep in separators:
            try:
                df = pd.read_csv(path, sep=sep, encoding=encoding, on_bad_lines="skip")
                if df.shape[1] == 1:
                    logger.warning(
                        f"[WARNING] File hanya terbaca 1 kolom pakai sep='{sep}', encoding='{encoding}'"
                    )
                    continue
                logger.info(
                    f"Berhasil baca: {path} | encoding='{encoding}' | sep='{sep}' | shape={df.shape}"
                )
                return df
            except Exception as e:
                logger.warning(f"[GAGAL] encoding={encoding}, sep={sep} → {e}")
    logger.error(f"[ERROR] Tidak bisa baca file CSV: {path}")
    return pd.DataFrame()

def hash_file(filepath):
    try:
        df = safe_read_csv(filepath)
        if df.empty:
            return ""
        df = df.sort_values(by=df.columns.tolist()).reset_index(drop=True)
        df_str = df.astype(str)
        return hashlib.md5(pd.util.hash_pandas_object(df_str, index=True).values).hexdigest()
    except Exception as e:
        logger.error(f"Gagal hitung hash file {filepath}: {e}")
        return ""

def save_current_hash(data_path):
    with open(config.LAST_HASH_FILE, "w") as f:
        f.write(hash_file(data_path))

def is_new_data_different(new_data_path):
    new_hash = hash_file(new_data_path)
    if not os.path.exists(config.LAST_HASH_FILE):
        return True
    with open(config.LAST_HASH_FILE, "r") as f:
        old_hash = f.read().strip()
    return new_hash != old_hash

# == MANAJEMEN AKURASI ==
def read_last_accuracy():
    if not os.path.exists(config.LAST_ACCURACY_FILE):
        return 0.0
    with open(config.LAST_ACCURACY_FILE, "r") as f:
        return float(f.read().strip())

def save_accuracy(acc):
    with open(config.LAST_ACCURACY_FILE, "w") as f:
        f.write(str(acc))

# == DATASET MANAGER ==
def get_new_data_path():
    try:
        csv_files = [
            os.path.join(config.NEW_DATA_DIR, file)
            for file in os.listdir(config.NEW_DATA_DIR)
            if file.endswith(".csv") and file != config.MAIN_DATA_FILE
        ]
        if not csv_files:
            return None
        csv_files.sort(key=os.path.getmtime, reverse=True)
        return csv_files[0]
    except Exception as e:
        logger.error(f"Gagal mendapatkan data baru: {e}")
        return None

def merge_datasets(main_path, new_path, merged_path):
    logger.info("Menggabungkan data utama dan data baru...")
    df_main = safe_read_csv(main_path)
    df_new = safe_read_csv(new_path)
    if df_main.empty or df_new.empty:
        logger.error("File kosong/gagal dibaca → gabungan dibatalkan.")
        return None
    required_cols = ["object_name", "content", "final_sentiment"]
    for col in required_cols:
        if col not in df_main.columns or col not in df_new.columns:
            logger.error(f"Kolom wajib '{col}' tidak ditemukan.")
            return None
    df_main = df_main.dropna(subset=required_cols)
    df_new = df_new.dropna(subset=required_cols)
    df = pd.concat([df_main, df_new], ignore_index=True)
    df = df.drop_duplicates(subset=required_cols).reset_index(drop=True)
    df.to_csv(merged_path, index=False, sep=";")
    df.to_csv(os.path.join(config.MERGED_DIR, config.MAIN_DATA_FILE), index=False, sep=";")
    logger.info(f"Dataset gabungan disimpan di: {merged_path}")
    return merged_path

# == PIPELINE TRAINING ==
def run_training_pipeline(csv_path):
    try:
        dataset_name = os.path.splitext(os.path.basename(csv_path))[0]
        logger.info(f"== Training untuk dataset: {dataset_name} ==")

        trainer = ABSAModelTrainer(
            csv_input_path=csv_path,
            model_name=dataset_name,
            model_path_prefix=config.MODELS_TORCH_DIR,
            csv_output_path_prefix="dataset-fix",
            num_epochs=5,
            csv_input_sep=";",
            is_onm=False,
        )

        # urutan yang benar
        trainer.prepare_dataset(force_overwrite=True)
        trainer.load_model()
        trainer.load_dataset()

        # Training
        acc = trainer.train_model()

        prev_acc = read_last_accuracy()
        if acc > prev_acc:
            logger.info(f"Akurasi meningkat {acc:.4f} > {prev_acc:.4f} → Simpan model & ONNX")
            
            # buat nama unik
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            onnx_dir = os.path.join(config.MODELS_ONNX_DIR, f"{dataset_name}_{timestamp}")
            os.makedirs(onnx_dir, exist_ok=True)

            trainer.export_to_onnx(onnx_dir)
            onnx_path = os.path.join(onnx_dir, f"{dataset_name}.onnx")

            # register model
            register_model(
                name=dataset_name,
                path=onnx_path,
                description=f"IndoBERT model trained on {dataset_name} with accuracy {acc:.4f}"
            )

            save_accuracy(acc)
            save_current_hash(csv_path)

        else:
            logger.info(f"Akurasi tidak meningkat {acc:.4f} ≤ {prev_acc:.4f} → Skip ekspor")

        # Inference & stats
        trainer.load_saved_model()
        trainer.run_inference(text_col="content", aspect_col="object_name")
        trainer.stats()

    except Exception as e:
        logger.exception(f"[ERROR] Terjadi kesalahan fatal saat training pipeline: {e}")

# == MAIN ==
if __name__ == "__main__":
    logger.info("== MLOps Pipeline Dimulai ==")
    new_data_path = get_new_data_path()
    main_data_file = os.path.join(config.MERGED_DIR, config.MAIN_DATA_FILE)
    main_data_exists = os.path.exists(main_data_file)

    if new_data_path:
        logger.info(f"File baru ditemukan: {new_data_path}")
        if not main_data_exists:
            logger.warning("Tidak ada main_dataset.csv → ini data pertama.")
            df = safe_read_csv(new_data_path)
            if df.empty:
                logger.error("[ERROR] File tidak bisa digunakan.")
            else:
                df.to_csv(main_data_file, index=False, sep=";")
                save_current_hash(main_data_file)
                run_training_pipeline(main_data_file)
        elif is_new_data_different(new_data_path):
            merged_file_name = os.path.basename(new_data_path)
            merged_path = os.path.join(config.MERGED_DIR, merged_file_name)
            final_dataset_path = merge_datasets(
                main_data_file, new_data_path, merged_path
            )
            if final_dataset_path:
                run_training_pipeline(final_dataset_path)
        else:
            logger.info("Dataset sama seperti sebelumnya → training dilewati.")
    else:
        logger.info("Tidak ada file baru ditemukan di folder data/new/")
