<<<<<<< HEAD
import glob
import os
import onnxruntime
import numpy as np

from transformers import AutoTokenizer
from db import run_query
from config import config # <-- Import config

NUM_LABELS = 3
=======
import onnxruntime
import numpy as np
from transformers import AutoTokenizer
from db import run_query
import os
import glob

NUM_LABELS = 3  # Jumlah label
>>>>>>> ba0c469a (initial commit)
LABELS = ["positive", "neutral", "negative"]

def get_onnx_model_path():
    """Ambil path model ONNX dari database, fallback ke lokal"""
    try:
        query = """
        SELECT path FROM absa_inference_models 
        WHERE path LIKE '%.onnx' 
<<<<<<< HEAD
        ORDER BY created_at DESC 
        LIMIT 1
        """
        result = run_query(query, fetch=True)
        if result and result[0]["path"] and os.path.exists(result[0]["path"]):
            db_path = result[0]["path"]
            print(f"✅ Menggunakan model ONNX dari DB: {db_path}")
            return db_path
    except Exception as e:
        print(f"❌ Error fetching ONNX path dari DB: {e}. Fallback ke lokal.")
    
    # fallback ke folder lokal
    files = glob.glob(os.path.join(config.MODELS_ONNX_DIR, "*.onnx"))
    if files:
        latest_file = max(files, key=os.path.getmtime)
        print(f"⚠️ Menggunakan model ONNX lokal: {latest_file}")
        return latest_file
    
    print("❌ Tidak ada model ONNX yang ditemukan.")
    return None
=======
        ORDER BY updated_at DESC 
        LIMIT 1
        """
        result = run_query(query, fetch=True)

        if result and len(result) > 0:
            db_path = result[0]['path']
            if os.path.exists(db_path):
                return db_path
            else:
                print(f"⚠️ Path dari DB tidak ditemukan: {db_path}, fallback ke lokal")

        # fallback ke folder lokal
        files = glob.glob("models_onnx/*.onnx")
        if files:
            return max(files, key=os.path.getmtime)
        return None
    except Exception as e:
        print(f"❌ Error fetching ONNX path: {e}")
        return None
>>>>>>> ba0c469a (initial commit)

def load_onnx_model(model_path):
    """Load ONNX model untuk inference"""
    try:
        session = onnxruntime.InferenceSession(model_path)
<<<<<<< HEAD
        # Asumsi tokenizer berada di models_torch/main_dataset
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(config.MODELS_TORCH_DIR, "main_dataset"))
=======
        tokenizer = AutoTokenizer.from_pretrained("models_torch/main_dataset")
>>>>>>> ba0c469a (initial commit)
        return session, tokenizer
    except Exception as e:
        print(f"❌ Error loading ONNX model: {e}")
        return None, None

def predict_sentiment_onnx(text, session, tokenizer, max_length=128):
<<<<<<< HEAD
    # ... (kode Anda sudah bagus, biarkan saja) ...
    try:
        # Tokenize
        inputs = tokenizer(
            text,
            return_tensors="np",
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        input_ids = inputs["input_ids"].astype(np.int64)
        attention_mask = inputs["attention_mask"].astype(np.int64)

        # Run inference
        outputs = session.run(
            None, {"input_ids": input_ids, "attention_mask": attention_mask}
        )
        
        logits = outputs[0]
        # Softmax manual
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        pred_idx = int(np.argmax(probs, axis=1)[0])
        pred_label = LABELS[pred_idx]

        return pred_label, probs[0].tolist()

=======
    """Predict menggunakan ONNX model"""
    try:
        # Tokenize
        inputs = tokenizer(
            text, 
            return_tensors="np", 
            truncation=True, 
            padding="max_length",
            max_length=max_length
        )

        # ONNXRuntime expects int64
        input_ids = inputs["input_ids"].astype(np.int64)
        attention_mask = inputs["attention_mask"].astype(np.int64)
        
        # Run inference
        outputs = session.run(
            None, 
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask
            }
        )
        
        # Softmax
        logits = outputs[0]
        probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
        pred_idx = int(np.argmax(probs, axis=1)[0])
        pred_label = LABELS[pred_idx]
        
        return pred_label, probs[0].tolist()
    
>>>>>>> ba0c469a (initial commit)
    except Exception as e:
        print(f"❌ ONNX prediction error: {e}")
        return "error", [0] * NUM_LABELS

if __name__ == "__main__":
<<<<<<< HEAD
    onnx_path = get_onnx_model_path()
    if onnx_path:
        session, tokenizer = load_onnx_model(onnx_path)
=======
    # Coba load ONNX model dulu
    onnx_path = get_onnx_model_path()
    
    if onnx_path:
        print(f"🔄 Loading ONNX model from: {onnx_path}")
        session, tokenizer = load_onnx_model(onnx_path)
        
>>>>>>> ba0c469a (initial commit)
        if session and tokenizer:
            text = "Pelayanan aplikasi ini sangat bagus, saya puas!"
            label, probs = predict_sentiment_onnx(text, session, tokenizer)
            print(f"🎯 ONNX Prediction - Label: {label}, Probs: {probs}")
        else:
<<<<<<< HEAD
            print("❌ Gagal memuat ONNX model.")
=======
            print("❌ Falling back ke PyTorch inference...")
            import inference  # fallback ke PyTorch biasa
    else:
        print("❌ No ONNX model found, fallback ke PyTorch...")
        import inference
>>>>>>> ba0c469a (initial commit)
