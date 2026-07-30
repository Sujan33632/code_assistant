import os
import glob
import hashlib
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Automatically load environment variables from .env file
load_dotenv()

# Helper function to generate a unique hash for a string of text
def generate_hash(text: str) -> str:
    """Generate a SHA-256 hash for a given text chunk."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def load_code_files(directory_path="data"):
    print(f"Loading files from {directory_path}...")
    documents = []
    
    extensions = ["*.py", "*.c", "*.cpp"]
    
    for ext in extensions:
        pattern = os.path.join(directory_path, ext)
        for filepath in glob.glob(pattern):
            try:
                with open(filepath, 'r', encoding='utf-8') as file:
                    content = file.read()
                    doc = Document(page_content=content, metadata={"source": filepath})
                    documents.append(doc)
                print(f"Loaded: {filepath}")
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
                
    return documents

def split_into_chunks(documents):
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")
    return chunks

def store_in_chromadb(chunks, persist_directory="embeddings"):
    print("Generating hashes and storing in ChromaDB (Upserting)...")
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # 1. Initialize the Chroma database first
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    
    # 2. Generate deterministic IDs based on the exact text of each chunk
    chunk_ids = [generate_hash(chunk.page_content) for chunk in chunks]
    
    # 3. Add documents using the explicit IDs
    # If the ID already exists, Chroma will safely handle it without duplicating
    vectorstore.add_documents(documents=chunks, ids=chunk_ids)
    
    print(f"Successfully processed and upserted {len(chunks)} chunks into '{persist_directory}/'.")
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