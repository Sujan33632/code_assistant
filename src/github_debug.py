import sys
from assistant import setup_assistant

def analyze_failure(error_message):
    print(f"🔍 Analyzing CI/CD Failure:\n{error_message}\n")
    
    # Reusing the existing setup from assistant.py (DRY Principle)
    rag_chain, _ = setup_assistant()
    
    # Wrap the error message to give the AI debugging context
    query = (
        f"The CI/CD pipeline failed with this error log: \n'{error_message}'\n\n"
        f"Search the codebase context to find what might be causing this issue. "
        f"Provide a short root cause analysis and a suggested code fix."
    )
    
    try:
        response = rag_chain.invoke(query)
        print("\n" + "="*60)
        print("🛠️ AUTOMATED DEBUG SUGGESTIONS:")
        print(response)
        print("="*60)
    except Exception as e:
        print(f"Error generating debug suggestions: {e}")

if __name__ == "__main__":
    # Ensure the script receives an error message as an argument
    if len(sys.argv) < 2:
        print("Usage: python src/github_debug.py '<error_message>'")
        sys.exit(1)
        
    error_log = sys.argv[1]
    analyze_failure(error_log)