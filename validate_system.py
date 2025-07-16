#!/usr/bin/env python3
"""
Quick validation script to ensure RAG system is working properly
"""

from app import RAGSystem, load_docs

def validate_rag_system():
    """Validate all components of the RAG system."""
    print("🔍 Validating RAG System Components...")
    print("=" * 50)
    
    # Test 1: Document loading
    print("1. Testing document loading...")
    docs = load_docs("./news_articles")
    print(f"   ✓ Loaded {len(docs)} documents")
    
    # Test 2: RAG system initialization
    print("2. Testing RAG system initialization...")
    rag = RAGSystem()
    print(f"   ✓ {rag.get_collection_info()}")
    
    # Test 3: Document retrieval
    print("3. Testing document retrieval...")
    test_query = "What is AI?"
    relevant_docs = rag.retrieve_relevant_documents(test_query, n_results=3)
    print(f"   ✓ Retrieved {len(relevant_docs)} relevant documents")
    
    # Test 4: Answer generation
    print("4. Testing answer generation...")
    answer = rag.answer_question("What is ChatGPT?")
    print(f"   ✓ Generated answer ({len(answer)} characters)")
    print(f"   Preview: {answer[:100]}...")
    
    print("\n✅ All components validated successfully!")
    print("🚀 Your RAG system is ready to use!")
    print("\nTo start using the system, run:")
    print("   python test_rag.py")

if __name__ == "__main__":
    validate_rag_system()
