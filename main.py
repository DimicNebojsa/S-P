# === Imports ===
import os
import pandas as pd
import snowflake.connector
from dotenv import load_dotenv
from getpass import getpass

from pymilvus import connections
from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, utility

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from database import insert_document, drop_table, test_connection, create_table, fetch_documents, conn_params
import psycopg2
from psycopg2.extras import RealDictCursor

from datetime import datetime

# === Load environment variables ===
load_dotenv()

SNOWFLAKEUSER = os.getenv("SNOWFLAKEUSER")
SNOWFLAKEPASS = os.getenv("SNOWFLAKEPASS")

# Set OpenAI API key from environment or prompt interactively
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY") or getpass("Enter OpenAI API Key: ")

# === Snowflake connection helpers ===
def get_snowflake_connection():
    """
    Establishes and returns a Snowflake cursor connection.
    """
    ctx = snowflake.connector.connect(
        user=SNOWFLAKEUSER,
        password=SNOWFLAKEPASS,
        account='cbc51873.us-east-1',
        warehouse='CUSTOMER_WH',
        role='PUBLIC',
        database='AI_READY_DATA_TRIAL_CONSUMER',
        schema='AI_READY_DATA_TRIAL'
    )
    cs = ctx.cursor()
    return cs

def read_data_from_snowflake(cs, sql_query):
    """
    Executes a query and returns the result as a Pandas DataFrame.
    """
    cs.execute(sql_query)
    df = cs.fetch_pandas_all()
    return df

def prepare_final_df(df):
    """
    Prepares a final DataFrame by isolating metadata and initializing `raw_content`.
    """
    metadata_columns = df.columns.difference(['SOURCEURI'])
    df['metadata'] = df[metadata_columns].to_dict(orient='records')
    df['raw_content'] = None
    final_df = df[['metadata', 'SOURCEURI', 'raw_content']]
    return final_df

# === Load data from Snowflake ===
cs = get_snowflake_connection()

# Load DOCUMENT_METADATA
sql_query = 'SELECT * FROM "DOCUMENT_METADATA"'
df_document = read_data_from_snowflake(cs, sql_query)

# Load SEGMENT_METADATA (limit 20 for testing)
sql_query = 'SELECT * FROM "SEGMENT_METADATA" LIMIT 20'
df = read_data_from_snowflake(cs, sql_query)

# === Helper to extract chunks per DOCUMENTID ===
def get_chunks_by_documentid(df: pd.DataFrame, documentid: str) -> list:
    """
    Filters rows by DOCUMENTID and returns their PROCESSEDSEGMENTCONTENT values as a list.
    """
    filtered_df = df[df["DOCUMENTID"] == documentid]
    chunks = filtered_df["PROCESSEDSEGMENTCONTENT"].tolist()
    return chunks

# Optional: Show row count per document
row_counts = df.groupby("DOCUMENTID").size().reset_index(name='row_count')
print(row_counts)

# === OpenAI model setup ===
load_dotenv()  # Ensure .env is loaded again in case this file is run standalone

api_key = os.getenv("OPENAI_MINI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL")
#base_url = "https://api.openai.com/v1"


llm = ChatOpenAI(
    base_url=base_url,
    api_key=api_key,
    model="gpt-4o-mini"
)

# === Reconstruct documents using LLM ===
reconstruct_system_prompt = """
You are an AI assistant helping reconstruct the original document from a list of unordered chunks.

Your task is to organize the chunks in a logical order and rewrite them as a coherent, well-structured document.

Do NOT add any explanations or commentary — only return the final reconstructed document as clean text.
"""

reconstruct_prompt_template = ChatPromptTemplate.from_messages([
    ("system", reconstruct_system_prompt),
    ("user", "{chunks}"),
])

list_of_docs = df['DOCUMENTID'].unique().tolist()
result_dict = {}

index = 1
for doc in list_of_docs:
    chunks = get_chunks_by_documentid(df, doc)
    chunks_input = "\n".join(chunks)

    reconstruct_pipeline = reconstruct_prompt_template | llm
    reconstructed_document = reconstruct_pipeline.invoke({"chunks": chunks_input}).content

    result_dict[doc] = reconstructed_document
    print(index, doc)
    index += 1

# === Create DataFrame from results and join with metadata ===
result_dict_df = pd.DataFrame(result_dict.items(), columns=['DOCUMENTID', 'ORIGINAL DOCUMENT'])
joined_df = pd.merge(df_document, result_dict_df, on='DOCUMENTID', how='inner')
print("##########")
print(joined_df.columns.tolist())



# === TXT writer ===
def write_documents_to_txt(documents, output_folder):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"📁 Created output folder: {output_folder}")

    for _, row in documents.iterrows():  # Correctly iterate DataFrame rows
        row = row.to_dict()  # Convert Series to dict
        doc_id = row.get("documentid") or row.get("DOCUMENTID")
        if not doc_id:
            print("⚠️ Skipping document without DOCUMENTID.")
            continue

        filename = f"{doc_id}.txt"
        filepath = os.path.join(output_folder, filename)

        if os.path.exists(filepath):
            print("File exists, skipping:", filename)
            continue

        with open(filepath, "w", encoding="utf-8") as f:
            for key, value in row.items():
                f.write(f"{key}: {value}\n")

# === Insert logic with conditional overwrite ===
# def insert_df_into_db(joined_df, output_folder):
#     # Ensure output folder exists
#     os.makedirs(output_folder, exist_ok=True)
#
#     try:
#         conn = psycopg2.connect(**conn_params)
#         cursor = conn.cursor(cursor_factory=RealDictCursor)
#
#         for idx in range(len(joined_df)):
#             row = joined_df.iloc[idx]
#             doc_id = str(row["DOCUMENTID"])
#
#             # Check if DOCUMENTID already exists
#             cursor.execute("SELECT UPDATED FROM documents WHERE DOCUMENTID = %s;", (doc_id,))
#             result = cursor.fetchone()
#
#             new_updated = pd.to_datetime(row["UPDATED"]).to_pydatetime() if pd.notna(row["UPDATED"]) else None
#             should_write = False
#
#             if result is None:
#                 should_write = True
#                 action = "inserted"
#             else:
#                 existing_updated = result["updated"]
#                 if new_updated and existing_updated and new_updated > existing_updated:
#                     should_write = True
#                     action = "updated"
#
#             if should_write:
#                 cursor.execute("""
#                     INSERT INTO documents (
#                         DOCUMENTID, NAME, DOCUMENTTITLE, PUBLISHED, UPDATED, FILETYPE,
#                         FILESIZE, SOURCEURI, REPORTINGFREQUENCY, PRIMARYENTITYTYPE,
#                         PRIMARYENTITYNAME, DOCUMENT_PRIMARY_ENTITY_ID, OTHERDOCUMENTMETADATA,
#                         DOCUMENTTYPE, VERSION, ORIGINAL_DOCUMENT
#                     ) VALUES (
#                         %(DOCUMENTID)s, %(NAME)s, %(DOCUMENTTITLE)s, %(PUBLISHED)s, %(UPDATED)s,
#                         %(FILETYPE)s, %(FILESIZE)s, %(SOURCEURI)s, %(REPORTINGFREQUENCY)s,
#                         %(PRIMARYENTITYTYPE)s, %(PRIMARYENTITYNAME)s, %(DOCUMENT_PRIMARY_ENTITY_ID)s,
#                         %(OTHERDOCUMENTMETADATA)s, %(DOCUMENTTYPE)s, %(VERSION)s, %(ORIGINAL_DOCUMENT)s
#                     )
#                     ON CONFLICT (DOCUMENTID) DO UPDATE SET
#                         NAME = EXCLUDED.NAME,
#                         DOCUMENTTITLE = EXCLUDED.DOCUMENTTITLE,
#                         PUBLISHED = EXCLUDED.PUBLISHED,
#                         UPDATED = EXCLUDED.UPDATED,
#                         FILETYPE = EXCLUDED.FILETYPE,
#                         FILESIZE = EXCLUDED.FILESIZE,
#                         SOURCEURI = EXCLUDED.SOURCEURI,
#                         REPORTINGFREQUENCY = EXCLUDED.REPORTINGFREQUENCY,
#                         PRIMARYENTITYTYPE = EXCLUDED.PRIMARYENTITYTYPE,
#                         PRIMARYENTITYNAME = EXCLUDED.PRIMARYENTITYNAME,
#                         DOCUMENT_PRIMARY_ENTITY_ID = EXCLUDED.DOCUMENT_PRIMARY_ENTITY_ID,
#                         OTHERDOCUMENTMETADATA = EXCLUDED.OTHERDOCUMENTMETADATA,
#                         DOCUMENTTYPE = EXCLUDED.DOCUMENTTYPE,
#                         VERSION = EXCLUDED.VERSION,
#                         ORIGINAL_DOCUMENT = EXCLUDED.ORIGINAL_DOCUMENT;
#                 """, {
#                     "DOCUMENTID": doc_id,
#                     "NAME": str(row["NAME"]) if pd.notna(row["NAME"]) else None,
#                     "DOCUMENTTITLE": str(row["DOCUMENTTITLE"]) if pd.notna(row["DOCUMENTTITLE"]) else None,
#                     "PUBLISHED": pd.to_datetime(row["PUBLISHED"]).to_pydatetime() if pd.notna(row["PUBLISHED"]) else None,
#                     "UPDATED": new_updated,
#                     "FILETYPE": str(row["FILETYPE"]) if pd.notna(row["FILETYPE"]) else None,
#                     "FILESIZE": float(row["FILESIZE"]) if pd.notna(row["FILESIZE"]) else None,
#                     "SOURCEURI": str(row["SOURCEURI"]) if pd.notna(row["SOURCEURI"]) else None,
#                     "REPORTINGFREQUENCY": str(row["REPORTINGFREQUENCY"]) if pd.notna(row["REPORTINGFREQUENCY"]) else None,
#                     "PRIMARYENTITYTYPE": str(row["PRIMARYENTITYTYPE"]) if pd.notna(row["PRIMARYENTITYTYPE"]) else None,
#                     "PRIMARYENTITYNAME": str(row["PRIMARYENTITYNAME"]) if pd.notna(row["PRIMARYENTITYNAME"]) else None,
#                     "DOCUMENT_PRIMARY_ENTITY_ID": str(row["DOCUMENT_PRIMARY_ENTITY_ID"]) if pd.notna(row["DOCUMENT_PRIMARY_ENTITY_ID"]) else None,
#                     "OTHERDOCUMENTMETADATA": row["OTHERDOCUMENTMETADATA"] if isinstance(row["OTHERDOCUMENTMETADATA"], dict) else None,
#                     "DOCUMENTTYPE": str(row["DOCUMENTTYPE"]) if pd.notna(row["DOCUMENTTYPE"]) else None,
#                     "VERSION": str(row["VERSION"]) if pd.notna(row["VERSION"]) else None,
#                     "ORIGINAL_DOCUMENT": str(row["ORIGINAL DOCUMENT"]) if pd.notna(row["ORIGINAL DOCUMENT"]) else None,
#                 })
#                 print(f"✅ Document {action}: {doc_id}")
#
#                 # Write to TXT file
#                 filename = f"{doc_id}.txt"
#                 filepath = os.path.join(output_folder, filename)
#                 with open(filepath, "w", encoding="utf-8") as f:
#                     for col in row.index:
#                         f.write(f"{col}: {row[col]}\n")
#             else:
#                 print(f"⏭️  Skipped (not newer): {doc_id}")
#
#         conn.commit()
#         cursor.close()
#         conn.close()
#
#     except Exception as e:
#         print(f"❌ Insert or update failed: {e}")

from psycopg2.extras import execute_values

def insert_df_into_db_bulk(joined_df, output_folder, chunk_size=100):
    os.makedirs(output_folder, exist_ok=True)

    try:
        conn = psycopg2.connect(**conn_params)
        cursor = conn.cursor()

        # Get existing DOCUMENTID and UPDATED from DB
        cursor.execute("SELECT DOCUMENTID, UPDATED FROM documents;")
        existing_df = pd.DataFrame(cursor.fetchall(), columns=["DOCUMENTID", "UPDATED"])
        existing_df["DOCUMENTID"] = existing_df["DOCUMENTID"].astype(str)
        existing_df["UPDATED"] = pd.to_datetime(existing_df["UPDATED"])

        joined_df["DOCUMENTID"] = joined_df["DOCUMENTID"].astype(str)
        joined_df["UPDATED"] = pd.to_datetime(joined_df["UPDATED"])

        # ✅ REPLACEMENT BLOCK: Filter new or updated records using merge
        merged_df = pd.merge(joined_df, existing_df, on="DOCUMENTID", how="left", suffixes=("", "_db"))
        filtered_df = merged_df[
            merged_df["UPDATED_db"].isna() | (merged_df["UPDATED"] > merged_df["UPDATED_db"])
        ].drop(columns=["UPDATED_db"])

        # DB column names
        columns = [
            "DOCUMENTID", "NAME", "DOCUMENTTITLE", "PUBLISHED", "UPDATED", "FILETYPE",
            "FILESIZE", "SOURCEURI", "REPORTINGFREQUENCY", "PRIMARYENTITYTYPE",
            "PRIMARYENTITYNAME", "DOCUMENT_PRIMARY_ENTITY_ID", "OTHERDOCUMENTMETADATA",
            "DOCUMENTTYPE", "VERSION", "ORIGINAL_DOCUMENT"
        ]

        all_rows = []
        for _, row in filtered_df.iterrows():
            row_dict = {
                col: (
                    pd.to_datetime(row[col]).to_pydatetime()
                    if col in ["PUBLISHED", "UPDATED"] and pd.notna(row[col])
                    else row[col] if col not in ["FILESIZE", "ORIGINAL_DOCUMENT"]
                    else float(row["FILESIZE"]) if col == "FILESIZE" and pd.notna(row["FILESIZE"])
                    else str(row["ORIGINAL DOCUMENT"]) if col == "ORIGINAL_DOCUMENT" and pd.notna(row["ORIGINAL DOCUMENT"])
                    else None
                )
                for col in columns
            }
            row_dict["DOCUMENTID"] = str(row["DOCUMENTID"])
            row_dict["OTHERDOCUMENTMETADATA"] = (
                row["OTHERDOCUMENTMETADATA"] if isinstance(row["OTHERDOCUMENTMETADATA"], dict) else None
            )
            all_rows.append((row_dict, row))  # store DB row and full row for txt

        if not all_rows:
            print("✅ No new or updated documents to insert.")
            return

        insert_query = f"""
            INSERT INTO documents ({", ".join(columns)}) VALUES %s
            ON CONFLICT (DOCUMENTID) DO UPDATE SET
                {", ".join(f"{col} = EXCLUDED.{col}" for col in columns if col != "DOCUMENTID")};
        """

        for i in range(0, len(all_rows), chunk_size):
            chunk = all_rows[i:i+chunk_size]
            values = [[r[0][col] for col in columns] for r in chunk]
            execute_values(cursor, insert_query, values)

            # Write TXT files for this chunk
            for _, row in chunk:
                doc_id = str(row["DOCUMENTID"])
                filepath = os.path.join(output_folder, f"{doc_id}.txt")
                with open(filepath, "w", encoding="utf-8") as f:
                    for col in row.index:
                        f.write(f"{col}: {row[col]}\n")

            print(f"✅ Inserted/updated chunk {i//chunk_size + 1}, wrote {len(chunk)} TXT files")

        conn.commit()
        cursor.close()
        conn.close()

    except Exception as e:
        print(f"❌ Bulk insert/update failed: {e}")



# === Optional: Force-update first two rows for testing ===
#joined_df.at[0, "UPDATED"] = datetime.now()
#joined_df.at[1, "UPDATED"] = datetime.now()

# === Run full pipeline ===
if __name__ == "__main__":
    #drop_table()
    test_connection()
    create_table()
    insert_df_into_db_bulk(joined_df, os.getenv("OUTPUT_FOLDER"), chunk_size=2)
    documents = fetch_documents()
    print("Done.")

