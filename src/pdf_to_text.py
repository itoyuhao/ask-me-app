import argparse
import os

import pandas as pd
import tiktoken
from dotenv import load_dotenv
from langchain.document_loaders import DataFrameLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.vectorstores import Chroma
from langchain_community.vectorstores.pgvector import PGVector
from tqdm import tqdm



try:
    import constants as const
    from utils import PathHelper, get_connection_string, get_logger, timeit
except Exception as e:
    print(e)
    raise ("Please run this script from the root directory of the project")

# logger
logger = get_logger(__name__)

# load env variables
dotenv_path = PathHelper.root_dir / ".env"
load_dotenv(dotenv_path=dotenv_path)

# model name for creating embeddings
model_name = const.ENCODING_MODEL_NAME

def num_tokens_from_string(string: str, encoding_name="cl100k_base") -> int:
    if not string:
        return 0
    # Returns the number of tokens in a text string
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

@timeit
def main(args):
    # init embeddings
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512, chunk_overlap=100, length_function=len
    )

    # load data to dataframe
    docs_lst = []
    topic_ids = []
    for root, dirs, files in os.walk(PathHelper.doc_dir):
        for file in files:
            file_path = os.path.join(root, file)
            docs_lst.append(file_path)
            # Get the directory name
            top_id = os.path.basename(os.path.dirname(file_path))
            topic_ids.append(top_id)
    doc_ids = [os.path.splitext(os.path.basename(d))[0] for d in docs_lst]

    
    df_docs = pd.DataFrame({"doc_id": doc_ids, "topic_id": topic_ids})
    new_list = []


    # Create a new list by splitting up text into token sizes of around 512 tokens
    for _, row in df_docs.iterrows():
        doc_id = row["doc_id"]
        topic_id = row["topic_id"]
        try:
            file_path = PathHelper.doc_dir / f"{topic_id}" / f"{doc_id}.pdf"
            print(f"file path: {file_path}")
            loader = PyPDFLoader(str(file_path)) if str(file_path).endswith(".pdf") else TextLoader(str(file_path))
            # Use splitter and split text into chunks
            texts = loader.load_and_split(text_splitter)

            for j in range(len(texts)):
                new_list.append([doc_id, topic_id, texts[j].metadata.get('page'), texts[j].page_content])

        except Exception as e:
            logger.error(e)

    df_new = pd.DataFrame(new_list, columns=["doc_id", "topic_id", "doc_page", "content"])

    # create docs
    loader = DataFrameLoader(df_new, page_content_column="content")
    docs = loader.load()

    if args.db == "pgvector":
        db = PGVector(
            collection_name=args.collection,
            connection_string=get_connection_string(),
            embedding_function=embeddings,
        )

        # add documents
        for doc in tqdm(docs, total=len(docs)):
            db.add_documents([doc])
        logger.info("done loading from docs")

    elif args.db == "chroma":
        page_contents = [doc.page_content for doc in docs]
        db = Chroma.from_texts(
            page_contents,
            embeddings,
            persist_directory=str(PathHelper.db_dir / const.CHROMA_DB),
        )
    else:
        raise ValueError(f"db: {args.db} not supported")

    return db

if __name__ == "__main__":
    # calculate how many seconds to run
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--db",
        type=str,
        default="pgvector",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=const.COLLECTION_NAME
    )

    args = parser.parse_args()
    main(args)
