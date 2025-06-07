

import pandas as pd
import os
import snowflake.connector
#from config import SNOWFLAKEUSER,SNOWFLAKEPASS
from pymilvus import connections
from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, utility

from dotenv import load_dotenv
from getpass import getpass
import os

load_dotenv()

SNOWFLAKEUSER = os.getenv("SNOWFLAKEUSER")
SNOWFLAKEPASS = os.getenv("SNOWFLAKEPASS")

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY") or getpass(
      "Enter OpenAI API Key: "
)


def get_snowflake_connection():
    ctx = snowflake.connector.connect(
        user=SNOWFLAKEUSER,
        password=SNOWFLAKEPASS,
        account='cbc51873.us-east-1',
        warehouse='CUSTOMER_WH',
        role='PUBLIC',  # As per screenshot, current role shown is PUBLIC
        database='AI_READY_DATA_TRIAL_CONSUMER',
        schema='AI_READY_DATA_TRIAL'
    )
    cs = ctx.cursor()
    return cs


def read_data_from_snowflake(cs, sql_query):
    cs.execute(sql_query)
    df = cs.fetch_pandas_all()
    return df


def get_milvus_connection(alias, MILVUS_PORT, MILVUS_HOST):
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


def prepare_final_df(df):
    # Create metadata column excluding SOURCEURI
    metadata_columns = df.columns.difference(['SOURCEURI'])  # all except SOURCEURI
    # Apply row-wise to_dict and assign to 'metadata' column
    df['metadata'] = df[metadata_columns].to_dict(orient='records')
    df['raw_content'] = None
    # Create final DataFrame with just 'metadata' and 'SOURCEURI'
    final_df = df[['metadata', 'SOURCEURI', 'raw_content']]
    # Preview result
    final_df
    return final_df


cs = get_snowflake_connection()
sql_query = 'SELECT * FROM "DOCUMENT_METADATA"'
df = read_data_from_snowflake(cs,sql_query)

#print(df.head())

sql_query = 'SELECT * FROM "SEGMENT_METADATA" LIMIT 100'
df = read_data_from_snowflake(cs,sql_query)

#print(df.head())


### Helper function

def get_chunks_by_documentid(df: pd.DataFrame, documentid: str) -> list:
    """
    Filters the DataFrame for rows with the given documentid,
    and returns a list of values from the 'PROCESSEDSEGMENTCONTENT' column.

    Args:
        df (pd.DataFrame): The DataFrame containing your data.
        documentid (str): The document ID to filter by.

    Returns:
        List[str]: A list of chunk values from PROCESSEDSEGMENTCONTENT.
    """
    filtered_df = df[df["DOCUMENTID"] == documentid]
    chunks = filtered_df["PROCESSEDSEGMENTCONTENT"].tolist()
    return chunks

row_counts = df.groupby("DOCUMENTID").size().reset_index(name='row_count')

#row_counts

openai_model = "gpt-4o-mini"

from langchain_openai import ChatOpenAI

# For normal accurate responses
llm = ChatOpenAI(temperature=0.0, model=openai_model)


list_of_docs = df['DOCUMENTID'].unique().tolist()

### dictionary with key which is document id and value that is original document

result_dict = {}

from langchain_core.prompts import ChatPromptTemplate

# Updated system prompt
reconstruct_system_prompt = """
You are an AI assistant helping reconstruct the original document from a list of unordered chunks.

Your task is to organize the chunks in a logical order and rewrite them as a coherent, well-structured document.

Do NOT add any explanations or commentary — only return the final reconstructed document as clean text.
"""

# Prompt template with the new instruction
reconstruct_prompt_template = ChatPromptTemplate.from_messages([
    ("system", reconstruct_system_prompt),
    ("user", "{chunks}"),
])

index = 1
for doc in list_of_docs:
    # Example input chunks (replace this with your actual list)
    chunks = get_chunks_by_documentid(df, doc)

    # Convert to single string input
    chunks_input = "\n".join(chunks)

    # Create pipeline
    reconstruct_pipeline = reconstruct_prompt_template | llm

    # Run the chain
    reconstructed_document = reconstruct_pipeline.invoke({"chunks": chunks_input}).content

    # print(reconstructed_document)

    result_dict[doc] = reconstructed_document

    print(index, doc)
    index += 1

print(result_dict)

## Semantic similarity approach (much faster then OpenAI, but now as acurate)

def compute_overlap(a, b, min_len=0):
    max_len = min(len(a), len(b))
    for i in range(max_len, min_len - 1, -1):
        if a[-i:] == b[:i]:
            return i
    return 0


def build_overlap_chain(chunks, start_idx):
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
    best_result = None
    best_score = float('inf')
    for i in range(len(chunks)):
        result = build_overlap_chain(chunks, start_idx=i)
        score = len(result)  # shorter result means better merging
        if score < best_score:
            best_score = score
            best_result = result
    return best_result


result_dict_second = {}


for doc in list_of_docs:
    chunks = get_chunks_by_documentid(df, doc)
    result = reconstruct_best_order(chunks)
    result_dict_second[doc] = result

print(result_dict_second)

print("Done....")

