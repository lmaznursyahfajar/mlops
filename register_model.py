import uuid
from datetime import datetime
from db import run_query

def get_next_version(model_name):
    query = """
        SELECT version FROM absa_inference_models 
        WHERE name=%s
        ORDER BY created_at DESC LIMIT 1
    """
    result = run_query(query, (model_name,), fetch=True)
    if result and len(result) > 0:
        last_version = result[0]["version"]
        try:
            new_version = str(round(float(last_version) + 0.1, 1))
            return new_version
        except:
            return "1.0"
    else:
        return "1.0"

def register_model(name, path, description=None):
    version = get_next_version(name)
    query = """
        INSERT INTO absa_inference_models (uuid, version, name, description, created_at, updated_at, path)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Generate UUID pendek 16 karakter
    short_uuid = uuid.uuid4().hex[:16]

    run_query(
        query,
        (
            short_uuid,
            version,
            name,
            description,
            now,
            now,
            path,
        ),
    )
    print(f"✅ Model '{name}' versi {version} berhasil diregistrasi ke DB dengan UUID {short_uuid}")

if __name__ == "__main__":
    latest_onnx = "models_onnx/contoh-file.onnx"
    register_model(
        name="main_dataset",
        path=latest_onnx,
        description="Model IndoBERT hasil training klasifikasi sentimen (ONNX)",
    )
