# # === Imports ===
#
# import os
# import pandas as pd
# import snowflake.connector
# from dotenv import load_dotenv
# from getpass import getpass
#
# from pymilvus import connections
# from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, utility
#
# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import ChatPromptTemplate
#
# # === Load environment variables ===
#
# load_dotenv()
#
# SNOWFLAKEUSER = os.getenv("SNOWFLAKEUSER")
# SNOWFLAKEPASS = os.getenv("SNOWFLAKEPASS")
#
# # Set OpenAI API key from environment or prompt interactively
# os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY") or getpass("Enter OpenAI API Key: ")
#
# # === Snowflake Connection ===
#
# def get_snowflake_connection():
#     """
#     Establishes and returns a Snowflake cursor connection.
#     """
#     ctx = snowflake.connector.connect(
#         user=SNOWFLAKEUSER,
#         password=SNOWFLAKEPASS,
#         account='cbc51873.us-east-1',
#         warehouse='CUSTOMER_WH',
#         role='PUBLIC',
#         database='AI_READY_DATA_TRIAL_CONSUMER',
#         schema='AI_READY_DATA_TRIAL'
#     )
#     cs = ctx.cursor()
#     return cs
#
# def read_data_from_snowflake(cs, sql_query):
#     """
#     Executes a query and returns the result as a Pandas DataFrame.
#     """
#     cs.execute(sql_query)
#     df = cs.fetch_pandas_all()
#     return df
#
# def prepare_final_df(df):
#     """
#     Prepares a final DataFrame by isolating metadata and initializing `raw_content`.
#     """
#     metadata_columns = df.columns.difference(['SOURCEURI'])
#     df['metadata'] = df[metadata_columns].to_dict(orient='records')
#     df['raw_content'] = None
#     final_df = df[['metadata', 'SOURCEURI', 'raw_content']]
#     return final_df
#
# # === Load data from Snowflake ===
#
# cs = get_snowflake_connection()
#
# # Load DOCUMENT_METADATA
# sql_query = 'SELECT * FROM "DOCUMENT_METADATA"'
# df_document = read_data_from_snowflake(cs,sql_query)
#
# #print(df_document[:1000])
#
# # Load SEGMENT_METADATA (limit for testing)
# sql_query = 'SELECT * FROM "SEGMENT_METADATA" LIMIT 20'
# df = read_data_from_snowflake(cs, sql_query)
# #print(df)
#
# # === Helper to extract chunks per document ===
#
# def get_chunks_by_documentid(df: pd.DataFrame, documentid: str) -> list:
#     """
#     Filters rows by DOCUMENTID and returns their PROCESSEDSEGMENTCONTENT values as a list.
#     """
#     filtered_df = df[df["DOCUMENTID"] == documentid]
#     chunks = filtered_df["PROCESSEDSEGMENTCONTENT"].tolist()
#     return chunks
#
# # Show row count per document (optional)
# row_counts = df.groupby("DOCUMENTID").size().reset_index(name='row_count')
# print(row_counts)
#
# # === OpenAI model setup ===
#
# # Load environment variables from .env file
# load_dotenv()
#
# # Read values from environment
# # api_key = os.getenv("OPENAI_API_KEY")
# # base_url = os.getenv("OPENAI_BASE_URL")
#
#
#
# from langchain_openai import ChatOpenAI
#
# # Load environment variables from .env file
# load_dotenv()
#
# # Read credentials from environment
# api_key = os.getenv("OPENAI_API_KEY")
# base_url = os.getenv("OPENAI_BASE_URL")
#
# # Initialize the LLM
# llm = ChatOpenAI(
#     base_url=base_url,
#     api_key=api_key,
#     model="gpt-4o"
# )
#
# from langchain_core.prompts import ChatPromptTemplate
#
# list_of_docs = df['DOCUMENTID'].unique().tolist()
# result_dict = {}
#
#
# # Updated system prompt
# reconstruct_system_prompt = """
# You are an AI assistant helping reconstruct the original document from a list of unordered chunks.
#
# Your task is to organize the chunks in a logical order and rewrite them as a coherent, well-structured document.
#
# Do NOT add any explanations or commentary — only return the final reconstructed document as clean text.
# """
#
# # Prompt template with the new instruction
# reconstruct_prompt_template = ChatPromptTemplate.from_messages([
#     ("system", reconstruct_system_prompt),
#     ("user", "{chunks}"),
# ])
#
# index = 1
# for doc in list_of_docs:
#     # Example input chunks (replace this with your actual list)
#     chunks = get_chunks_by_documentid(df, doc)
#
#     # Convert to single string input
#     chunks_input = "\n".join(chunks)
#
#     # Create pipeline
#     reconstruct_pipeline = reconstruct_prompt_template | llm
#
#     # Run the chain
#     reconstructed_document = reconstruct_pipeline.invoke({"chunks": chunks_input}).content
#
#     # print(reconstructed_document)
#
#     result_dict[doc] = reconstructed_document
#
#     print(index, doc)
#     index += 1
#
#
# #print(result_dict)
#
# # === Create Dataframe from dictionary ===
# result_dict_df = pd.DataFrame(result_dict.items(), columns=['Key', 'Value'])
# result_dict_df
#
# # Rename 'Key' to 'documentid' in df1 to match df2
# result_dict_df = result_dict_df.rename(columns={'Key': 'DOCUMENTID'})
# result_dict_df = result_dict_df.rename(columns={'Value': 'ORIGINAL DOCUMENT'})
#
# # Perform the inner join on 'documentid'
# joined_df = pd.merge(df_document, result_dict_df, on='DOCUMENTID', how='inner')
#
# print(joined_df)
#
#
# from database import insert_document, drop_table, test_connection, create_table, fetch_documents
#
# def insert_df_into_db(joined_df):
#     for idx in range(len(joined_df)):
#         row = joined_df.iloc[idx]
#         document_id = str(row["DOCUMENTID"])
#         original_document = row["ORIGINAL DOCUMENT"]
#
#         try:
#             insert_document(document_id, original_document)
#         except Exception as e:
#             print(f"❌ Failed to insert {document_id}: {e}")
#
# # doc_id = "0c92f951-5314-42df-8d40-e1222feef39f"
# # original_text = """
# # In mid-December, during emergency talks in Brussels, EU energy ministers...
# # (remainder of your document)
# # """
#
# if __name__ == "__main__":
#     #drop_table()
#     test_connection()
#     create_table()
#     #insert_document(doc_id, original_text)
#     insert_df_into_db(joined_df)
#     print("-------------------------------------------------------")
#     fetch_documents()
#     print("Done...")

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

from database import insert_document, drop_table, test_connection, create_table, fetch_documents

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

api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL")

llm = ChatOpenAI(
    base_url=base_url,
    api_key=api_key,
    model="gpt-4o"
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

print(joined_df)

# === Insert into database ===
def insert_df_into_db(joined_df):
    for idx in range(len(joined_df)):
        row = joined_df.iloc[idx]
        document_id = str(row["DOCUMENTID"])
        original_document = row["ORIGINAL DOCUMENT"]

        try:
            insert_document(document_id, original_document)
        except Exception as e:
            print(f"❌ Failed to insert {document_id}: {e}")

# === Run full pipeline ===
if __name__ == "__main__":
    drop_table()
    test_connection()
    create_table()
    insert_df_into_db(joined_df)
    print("-------------------------------------------------------")
    fetch_documents()
    print("Done.")
