from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import shutil
import os

db_path = './chroma_db'
if os.path.exists(db_path):
    shutil.rmtree(db_path)
    print(f"Database at {db_path} has been deleted.")
else:
    print(f"No database found at {db_path}.")

paths = [
    "stripe-2022-update.pdf",
    "CoffeeB_Manual Globe_EN_10.08.2022.pdf"
]

all_chunks = []

for path in paths:
    loader = UnstructuredPDFLoader(file_path=path)
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_documents(data)

    all_chunks.extend(chunks)

vector_db = Chroma.from_documents(
    documents=all_chunks,
    embedding=OllamaEmbeddings(model="nomic-embed-text", show_progress=True),
    collection_name="local-rag",
    persist_directory="./chroma_db"
)


