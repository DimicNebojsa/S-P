import os
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv()

# Load DB config from .env
conn_params = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"),
}

def test_connection():
    try:
        conn = psycopg2.connect(**conn_params)
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("SELECT NOW();")
        result = cursor.fetchone()
        print("✅ Connected. Current time in DB:", result["now"])
        cursor.close()
        conn.close()
    except Exception as e:
        print("❌ Connection failed:", e)

def create_table():
    try:
        conn = psycopg2.connect(**conn_params)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                DOCUMENTID VARCHAR(255) PRIMARY KEY,
                ORIGINAL_DOCUMENT TEXT
            );
        """)
        conn.commit()
        print("✅ Table 'documents' created.")
        cursor.close()
        conn.close()
    except Exception as e:
        print("❌ Failed to create table:", e)

def insert_document(document_id, original_document):
    try:
        conn = psycopg2.connect(**conn_params)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO documents (DOCUMENTID, ORIGINAL_DOCUMENT)
            VALUES (%s, %s)
            ON CONFLICT (DOCUMENTID) DO NOTHING;
        """, (document_id, original_document))
        conn.commit()
        print("✅ Document inserted.")
        cursor.close()
        conn.close()
    except Exception as e:
        print("❌ Insert failed:", e)

def fetch_documents():
    try:
        conn = psycopg2.connect(**conn_params)
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("SELECT * FROM documents;")
        rows = cursor.fetchall()
        print("📄 Documents:")
        for row in rows:
            print(f"\nDOCUMENTID: {row['documentid']}")
            print(f"ORIGINAL_DOCUMENT:\n{row['original_document']}")
        cursor.close()
        conn.close()
    except Exception as e:
        print("❌ Fetch failed:", e)

def drop_table():
    try:
        conn = psycopg2.connect(**conn_params)
        cursor = conn.cursor()
        cursor.execute("DROP TABLE IF EXISTS documents;")
        conn.commit()
        print("🧨 Table 'documents' dropped successfully.")
        cursor.close()
        conn.close()
    except Exception as e:
        print("❌ Failed to drop table:", e)

