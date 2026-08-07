from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
import os

def ingest_document(source_path: str, source_type: str = "url", db_path: str = "./chroma_db"):
    
    """Loads, splits, and embeds a document into Chroma DB."""
    try:
        if source_type == "url":
            loader = WebBaseLoader(source_path)
        elif source_type == "pdf":
            loader = PyPDFLoader(source_path)
        else:
            loader = TextLoader(source_path)
            
        docs = loader.load()

        for doc in docs:
            doc.metadata["source"] = source_path
            doc.metadata["type"] = source_type

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200,
            add_start_index=True
        )
        splits = text_splitter.split_documents(docs)

        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        
        Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=db_path
        )
        return f"Successfully ingested and embedded {len(splits)} chunks from {source_path}."
    except Exception as e:
        return f"Failed to ingest document: {str(e)}"