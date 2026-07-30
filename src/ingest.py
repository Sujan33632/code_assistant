import os
import glob
from dotenv import load_dotenv
from langchain_core.documents import Document  # Replaced TextLoader with core Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Automatically load environment variables from .env file
load_dotenv()

# 1. Read all code files from the data/ folder using standard Python I/O
def load_code_files(directory_path="data"):
    print(f"Loading files from {directory_path}...")
    documents = []
    
    # Define the extensions to ingest
    extensions = ["*.py", "*.c", "*.cpp"]
    
    for ext in extensions:
        pattern = os.path.join(directory_path, ext)
        for filepath in glob.glob(pattern):
            try:
                # Use standard Python open() instead of LangChain's deprecated TextLoader
                with open(filepath, 'r', encoding='utf-8') as file:
                    content = file.read()
                    # Manually create the LangChain Document object
                    doc = Document(page_content=content, metadata={"source": filepath})
                    documents.append(doc)
                print(f"Loaded: {filepath}")
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
                
    return documents

# 2. Split text into chunks
def split_into_chunks(documents):
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")
    return chunks

# 3 & 4. Create embeddings and store in ChromaDB
def store_in_chromadb(chunks, persist_directory="embeddings"):
    print("Generating embeddings and storing in ChromaDB...")
    
    # Local HuggingFace embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Store chunks + embeddings into ChromaDB
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    print(f"Successfully stored {len(chunks)} chunks in '{persist_directory}/' folder.")
    return vectorstore

if __name__ == "__main__":
    print("--- Starting Code Ingestion ---")
    docs = load_code_files()
    
    if not docs:
        print("No documents found. Please check files in the data/ folder.")
    else:
        chunks = split_into_chunks(docs)
        store_in_chromadb(chunks)
        print("--- Ingestion Complete ---")