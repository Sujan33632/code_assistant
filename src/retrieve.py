from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Load environment variables
load_dotenv()

# 1. Load the existing ChromaDB vector store
def load_vectorstore(persist_directory="embeddings"):
    print(f"Loading vector database from '{persist_directory}/'...")
    
    # We MUST use the exact same embedding model used during ingestion
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Load the database from disk
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    return vectorstore

# 2. Search relevant chunks for a question
def search_code(question, vectorstore, k=2):
    print(f"Searching for: '{question}'...")
    
    # similarity_search returns the top k most relevant text chunks
    results = vectorstore.similarity_search(question, k=k)
    return results

# 3. Simple test to verify it works
if __name__ == "__main__":
    print("--- Starting Retrieval Test ---")
    
    db = load_vectorstore()
    
    # The question we want to ask our codebase
    test_question = "How is the user login authentication handled?"
    
    # Perform the search (k=2 means return top 2 results)
    retrieved_chunks = search_code(test_question, db, k=2)
    
    print("\n--- Retrieved Code Snippets ---")
    for i, chunk in enumerate(retrieved_chunks, 1):
        source_file = chunk.metadata.get("source", "Unknown file")
        print(f"\nResult {i} (Source: {source_file}):")
        print("-" * 40)
        print(chunk.page_content)
        print("-" * 40)
        
    print("\n--- Retrieval Complete ---")