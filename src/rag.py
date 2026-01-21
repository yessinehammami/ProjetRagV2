from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import AzureOpenAIEmbeddings
from langchain_chroma import Chroma
from pathlib import Path
import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n","\n"," ","","."],
        chunk_size=4000,
        chunk_overlap=250
    )

#Function for preprocessing the pdf document

local_data=[]

def preprocess_pdfs(file):
    all_docs = []
    Loader = PyPDFLoader(str(file))
    data = Loader.load()
    doc_list = text_splitter.split_documents(data)

    for doc_chunk in doc_list:
        all_docs.append(Document(
            page_content=doc_chunk.page_content,
            metadata={"source": str(file)}
        ))
    return all_docs

data_path = Path("./data")
for filename in os.listdir(data_path):
    file_path = data_path / filename
    all_docs = preprocess_pdfs(file_path)
    local_data.extend(all_docs)


embedding_model = AzureOpenAIEmbeddings(
    azure_endpoint=os.getenv("EMBEDDINGS_ENDPOINT"),
    api_key=os.getenv("EMBEDDINGS_KEY"),
    azure_deployment=os.getenv("EMBEDDINGS_DEPLOYMENT"), 
    api_version=os.getenv("EMBEDDINGS_API_VERSION"),
)

vectorstore = Chroma.from_documents(
    local_data,
    embedding=embedding_model
)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs= {"k":5}
)