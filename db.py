# db.py
import mysql.connector
import pandas as pd
from config import config  # gunakan config dari file config.py

def get_connection():
    try:
        return mysql.connector.connect(
            host=config.DB_HOST,
            user=config.DB_USER,
            password=config.DB_PASSWORD,
            database=config.DB_DATABASE,
            charset="utf8mb4",
        )
    except Exception as e:
        print(f"❌ ERROR: Gagal koneksi ke database. Pastikan file config.py sudah benar. Error: {e}")
        return None

def run_query(query, params=None, fetch=False):
    conn = get_connection()
    if not conn:
        return None
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute(query, params or ())
        result = cursor.fetchall() if fetch else None
        conn.commit()
        cursor.close()
        return result
    except mysql.connector.Error as e:
        print(f"❌ ERROR: Gagal menjalankan query. Query: {query}, Error: {e}")
        conn.rollback()
        return None
    finally:
        if conn and conn.is_connected():
            conn.close()

def get_dataframe(query, params=None):
    conn = get_connection()
    if not conn:
        return pd.DataFrame()
    try:
        df = pd.read_sql(query, conn, params=params)
        return df
    except Exception as e:
        print(f"❌ ERROR: Gagal membuat dataframe dari query. Error: {e}")
        return pd.DataFrame()
    finally:
        if conn and conn.is_connected():
            conn.close()
