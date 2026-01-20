from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import AzureOpenAIEmbeddings
from langchain_chroma import Chroma
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

all_docs = []
data_path = Path("./data")

if not data_path.exists() or not data_path.is_dir():
    raise FileNotFoundError(f"Data directory not found at: {data_path}")

text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n","\n"," ","","."],
        chunk_size=1000,
        chunk_overlap=250
    )

for filename in os.listdir(data_path):
    if not filename.lower().endswith(".pdf"):
        continue

    file_path = data_path / filename
    Loader = PyPDFLoader(str(file_path))
    data = Loader.load()
    doc_list = text_splitter.split_documents(data)

    for doc_chunk in doc_list:
        all_docs.append(Document(
            page_content=doc_chunk.page_content,
            metadata={"source": filename}
        ))



embedding_model = AzureOpenAIEmbeddings(
    azure_endpoint=os.getenv("EMBEDDINGS_ENDPOINT"),
    api_key=os.getenv("EMBEDDINGS_KEY"),
    azure_deployment=os.getenv("EMBEDDINGS_DEPLOYMENT"), 
    api_version=os.getenv("EMBEDDINGS_API_VERSION"),
)

vectorstore = Chroma.from_documents(
    all_docs,
    embedding=embedding_model
)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs= {"k":5}
)