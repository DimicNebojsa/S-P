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

# def create_table():
#     try:
#         conn = psycopg2.connect(**conn_params)
#         cursor = conn.cursor()
#         cursor.execute("""
#             CREATE TABLE IF NOT EXISTS documents (
#                 DOCUMENTID VARCHAR(255) PRIMARY KEY,
#                 ORIGINAL_DOCUMENT TEXT
#             );
#         """)
#         conn.commit()
#         print("✅ Table 'documents' created.")
#         cursor.close()
#         conn.close()
#     except Exception as e:
#         print("❌ Failed to create table:", e)

def create_table():
    """
    Creates the 'documents' table with appropriately inferred column types,
    if it doesn't already exist.
    """
    try:
        conn = psycopg2.connect(**conn_params)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                DOCUMENTID VARCHAR(36) PRIMARY KEY,
                NAME TEXT,
                DOCUMENTTITLE TEXT,
                PUBLISHED TIMESTAMP,
                UPDATED TIMESTAMP,
                FILETYPE VARCHAR(20),
                FILESIZE FLOAT,
                SOURCEURI TEXT,
                REPORTINGFREQUENCY TEXT,
                PRIMARYENTITYTYPE VARCHAR(50),
                PRIMARYENTITYNAME TEXT,
                DOCUMENT_PRIMARY_ENTITY_ID TEXT,
                OTHERDOCUMENTMETADATA JSONB,
                DOCUMENTTYPE VARCHAR(100),
                VERSION VARCHAR(10),
                ORIGINAL_DOCUMENT TEXT
            );
        """)
        conn.commit()
        print("✅ Table 'documents' created.")
        cursor.close()
        conn.close()
    except Exception as e:
        print("❌ Failed to create table:", e)



# def insert_document(document_id, original_document):
#     try:
#         conn = psycopg2.connect(**conn_params)
#         cursor = conn.cursor()
#         cursor.execute("""
#             INSERT INTO documents (DOCUMENTID, ORIGINAL_DOCUMENT)
#             VALUES (%s, %s)
#             ON CONFLICT (DOCUMENTID) DO NOTHING;
#         """, (document_id, original_document))
#         conn.commit()
#         print("✅ Document inserted.")
#         cursor.close()
#         conn.close()
#     except Exception as e:
#         print("❌ Insert failed:", e)

def insert_document(
    document_id, name, documenttitle, published, updated, filetype,
    filesize, sourceuri, reportingfrequency, primaryentitytype,
    primaryentityname, document_primary_entity_id, otherdocumentmetadata,
    documenttype, version, original_document
):
    try:
        conn = psycopg2.connect(**conn_params)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO documents (
                DOCUMENTID, NAME, DOCUMENTTITLE, PUBLISHED, UPDATED, FILETYPE,
                FILESIZE, SOURCEURI, REPORTINGFREQUENCY, PRIMARYENTITYTYPE,
                PRIMARYENTITYNAME, DOCUMENT_PRIMARY_ENTITY_ID, OTHERDOCUMENTMETADATA,
                DOCUMENTTYPE, VERSION, ORIGINAL_DOCUMENT
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (DOCUMENTID) DO NOTHING;
        """, (
            document_id, name, documenttitle, published, updated, filetype,
            filesize, sourceuri, reportingfrequency, primaryentitytype,
            primaryentityname, document_primary_entity_id, otherdocumentmetadata,
            documenttype, version, original_document
        ))
        conn.commit()
        print(f"✅ Document inserted: {document_id}")
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"❌ Insert failed for {document_id}: {e}")



# def fetch_documents():
#     try:
#         conn = psycopg2.connect(**conn_params)
#         cursor = conn.cursor(cursor_factory=RealDictCursor)
#         cursor.execute("SELECT * FROM documents;")
#         rows = cursor.fetchall()
#         print("📄 Documents:")
#         for row in rows:
#             print(f"\nDOCUMENTID: {row['documentid']}")
#             print(f"ORIGINAL_DOCUMENT:\n{row['original_document']}")
#         cursor.close()
#         conn.close()
#     except Exception as e:
#         print("❌ Fetch failed:", e)

import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor

import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor

def fetch_documents():
    try:
        conn = psycopg2.connect(**conn_params)
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("SELECT * FROM documents;")
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        return pd.DataFrame(rows)
    except Exception as e:
        # Print the error for debugging visibility
        print("❌ Fetch failed:", e)
        # Return an empty DataFrame to keep the code from crashing
        return pd.DataFrame()

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

