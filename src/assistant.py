from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from retrieve import load_vectorstore

# Load environment variables
load_dotenv()

# Helper function to combine multiple code chunks into one string
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def setup_assistant():
    print("Initializing AI Code Assistant (Pure LCEL)...")
    
    # 1. Load vector store and create retriever
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # 2. Initialize the Groq LLM
    llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)
    
    # 3. Create the Prompt Template
    template = """You are an expert software developer and coding assistant.
    Use the following pieces of retrieved codebase context to answer the question.
    If the answer is not in the context, say 'I cannot find the answer in the codebase'.
    Keep your answers concise and provide code snippets where helpful.

    Context:
    {context}

    Question: {question}
    
    Answer:"""
    prompt = ChatPromptTemplate.from_template(template)
    
    # 4. Build the Pure LCEL RAG Chain
    # This reads as: Get context & question -> Pass to Prompt -> Pass to LLM -> Parse as String
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain, retriever

def ask_question(rag_chain, retriever, question):
    print("\nThinking...")
    
    # Manually fetch the documents so we can print the source files later
    source_docs = retriever.invoke(question)
    
    # Invoke the pure LCEL chain to get the final answer
    answer = rag_chain.invoke(question)
    
    print("\n" + "="*60)
    print("🤖 ANSWER:")
    print(answer)
    print("="*60)
    
    # Print the files it used to get this answer
    print("\n📄 Sources Used:")
    sources = set([doc.metadata.get('source', 'Unknown') for doc in source_docs])
    for source in sources:
        print(f"- {source}")
    print("\n")

if __name__ == "__main__":
    chain, retriever = setup_assistant()
    print("\n✅ Assistant is ready! Type 'exit' or 'quit' to stop.")
    
    while True:
        try:
            user_question = input("\n> Ask a question about the codebase: ")
            
            if user_question.lower() in ['exit', 'quit']:
                print("Goodbye!")
                break
            if not user_question.strip():
                continue
                
            ask_question(chain, retriever, user_question)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break