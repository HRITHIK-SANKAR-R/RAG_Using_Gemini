#!/usr/bin/env python3
"""
Complete RAG system with Gemini AI - Ready to use
"""

import os
import sys
from app import RAGSystem, load_docs

def main():
    """Main RAG system with Q&A functionality."""
    print("🤖 RAG System with Gemini AI")
    print("=" * 50)
    
    try:
        # Initialize RAG system
        print("Initializing RAG system...")
        rag = RAGSystem()
        
        # Check collection status
        collection_info = rag.get_collection_info()
        print(f"✓ {collection_info}")
        
        # Load documents if collection is empty
        if rag.collection.count() == 0:
            print("\nLoading documents from news_articles...")
            docs = load_docs("./news_articles")
            
            if docs:
                print(f"Found {len(docs)} documents. Processing...")
                rag.add_documents(docs)
                print("✓ Documents successfully added to the system")
            else:
                print("✗ No documents found in ./news_articles/")
                return
        else:
            print("✓ Collection already contains data")
        
        # Start Q&A session
        print("\n" + "="*60)
        print("🚀 RAG System Ready! Ask questions about the news articles.")
        print("Commands: 'quit' or 'exit' to stop, 'help' for examples")
        print("="*60)
        
        # Sample questions for user reference
        sample_questions = [
            "What is ChatGPT?",
            "Tell me about AI investments",
            "What companies are working on AI?",
            "What are the main AI trends mentioned?",
            "How is AI affecting different industries?"
        ]
        
        while True:
            query = input("\n💬 Your question: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if query.lower() == 'help':
                print("\n📝 Sample questions you can ask:")
                for i, q in enumerate(sample_questions, 1):
                    print(f"   {i}. {q}")
                continue
            
            if not query:
                print("❌ Please enter a valid question.")
                continue
            
            print("🔍 Processing your question...")
            
            # Get answer from RAG system
            answer = rag.answer_question(query)
            
            print(f"\n🤖 Answer:")
            print("-" * 40)
            print(answer)
            print("-" * 40)
    
    except KeyboardInterrupt:
        print("\n\n👋 Session interrupted. Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
