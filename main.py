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

# === Load environment variables ===

load_dotenv()

SNOWFLAKEUSER = os.getenv("SNOWFLAKEUSER")
SNOWFLAKEPASS = os.getenv("SNOWFLAKEPASS")

# Set OpenAI API key from environment or prompt interactively
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY") or getpass("Enter OpenAI API Key: ")

# === Snowflake Connection ===

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

# === Milvus Connection ===

def get_milvus_connection(alias, MILVUS_PORT, MILVUS_HOST):
    """
    Connects to Milvus server with provided host and port.
    """
    try:
        connections.connect(
            alias=alias,
            host=MILVUS_HOST,
            port=MILVUS_PORT
        )
        addr = connections.get_connection_addr(alias=alias)
        print(f"Connected to Milvus ({alias}): {addr}")
        return True
    except Exception as e:
        print(f"Failed to connect to Milvus ({alias}): {e}")
        return False

# === DataFrame Transformation ===

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
df = read_data_from_snowflake(cs, sql_query)

# Load SEGMENT_METADATA (limit for testing)
sql_query = 'SELECT * FROM "SEGMENT_METADATA" LIMIT 100'
df = read_data_from_snowflake(cs, sql_query)

# === Helper to extract chunks per document ===

def get_chunks_by_documentid(df: pd.DataFrame, documentid: str) -> list:
    """
    Filters rows by DOCUMENTID and returns their PROCESSEDSEGMENTCONTENT values as a list.
    """
    filtered_df = df[df["DOCUMENTID"] == documentid]
    chunks = filtered_df["PROCESSEDSEGMENTCONTENT"].tolist()
    return chunks

# Show row count per document (optional)
row_counts = df.groupby("DOCUMENTID").size().reset_index(name='row_count')

# === OpenAI model setup ===

openai_model = "gpt-4o-mini"
llm = ChatOpenAI(temperature=0.0, model=openai_model)

# === Prompt template for reconstruction ===

reconstruct_system_prompt = """
You are an AI assistant helping reconstruct the original document from a list of unordered chunks.

Your task is to organize the chunks in a logical order and rewrite them as a coherent, well-structured document.

Do NOT add any explanations or commentary — only return the final reconstructed document as clean text.
"""

reconstruct_prompt_template = ChatPromptTemplate.from_messages([
    ("system", reconstruct_system_prompt),
    ("user", "{chunks}"),
])

# === Reconstruct documents using LLM ===

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

print(result_dict)

# === Alternative: Fast Semantic Similarity-Based Reconstruction ===

def compute_overlap(a, b, min_len=0):
    """
    Computes character overlap length between end of `a` and start of `b`.
    """
    max_len = min(len(a), len(b))
    for i in range(max_len, min_len - 1, -1):
        if a[-i:] == b[:i]:
            return i
    return 0

def build_overlap_chain(chunks, start_idx):
    """
    Constructs document by chaining chunks based on maximum overlap.
    """
    used = set()
    used.add(start_idx)
    sequence = [chunks[start_idx]]
    current_idx = start_idx

    while len(used) < len(chunks):
        best_score = -1
        best_idx = -1
        for i, c in enumerate(chunks):
            if i in used:
                continue
            score = compute_overlap(sequence[-1], c)
            if score > best_score:
                best_score = score
                best_idx = i
        if best_idx == -1:
            for i in range(len(chunks)):
                if i not in used:
                    sequence.append(chunks[i])
                    used.add(i)
                    break
        else:
            overlap_len = compute_overlap(sequence[-1], chunks[best_idx])
            sequence.append(chunks[best_idx][overlap_len:])
            used.add(best_idx)

    return "".join(sequence)

def reconstruct_best_order(chunks):
    """
    Tries all starting positions to find the best reconstruction by overlap score.
    """
    best_result = None
    best_score = float('inf')
    for i in range(len(chunks)):
        result = build_overlap_chain(chunks, start_idx=i)
        score = len(result)  # smaller = better merging
        if score < best_score:
            best_score = score
            best_result = result
    return best_result

# Reconstruct documents using overlap-based heuristic
result_dict_second = {}

for doc in list_of_docs:
    chunks = get_chunks_by_documentid(df, doc)
    result = reconstruct_best_order(chunks)
    result_dict_second[doc] = result

print(result_dict_second)

print("Done....")
