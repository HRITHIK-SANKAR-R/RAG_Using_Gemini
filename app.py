import os
import uuid
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
from google import genai
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get API key
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

# Initialize embedding function
google_embedding_functions = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
    model_name="models/embedding-001",
    api_key=GOOGLE_API_KEY
)


def load_docs(directory):
    """Load text documents from a directory."""
    logger.info(f"Loading documents from {directory}")
    docs = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:  # Only add non-empty documents
                        docs.append({"id": filename, "text": content})
                        logger.debug(f"Loaded document: {filename}")
            except Exception as e:
                logger.error(f"Error loading {filename}: {e}")
    logger.info(f"Successfully loaded {len(docs)} documents")
    return docs

def split_text(text, chunk_size=1000, chunk_overlap=200):
    """Split text into chunks with overlap."""
    if not text.strip():
        return []
    
    chunks = []
    text_length = len(text)
    
    for i in range(0, text_length, chunk_size - chunk_overlap):
        end_pos = min(i + chunk_size, text_length)
        chunk = text[i:end_pos].strip()
        
        if chunk:  # Only add non-empty chunks
            chunks.append(chunk)
        
        if end_pos >= text_length:
            break
    
    return chunks


class RAGSystem:
    """A complete Retrieval-Augmented Generation system."""
    
    def __init__(self, persist_directory="chromadb_database", collection_name="rag_collection"):
        """Initialize the RAG system."""
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(path=persist_directory)
        
        # Create or get collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=google_embedding_functions
        )
        
        # Initialize Gemini client
        self.google_client = genai.Client(api_key=GOOGLE_API_KEY)
        
        logger.info(f"RAG System initialized with collection: {collection_name}")
    
    def add_documents(self, documents):
        """Add documents to the vector database."""
        logger.info(f"Processing {len(documents)} documents...")
        
        all_chunks = []
        all_ids = []
        all_metadatas = []
        
        for doc in documents:
            # Split document into chunks
            chunks = split_text(doc["text"])
            
            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc['id']}_chunk_{i}_{uuid.uuid4().hex[:8]}"
                all_chunks.append(chunk)
                all_ids.append(chunk_id)
                all_metadatas.append({
                    "source": doc["id"],
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                })
        
        if all_chunks:
            try:
                # Add to ChromaDB collection
                self.collection.add(
                    documents=all_chunks,
                    ids=all_ids,
                    metadatas=all_metadatas
                )
                logger.info(f"Successfully added {len(all_chunks)} chunks to the database")
            except Exception as e:
                logger.error(f"Error adding documents to database: {e}")
                raise
        else:
            logger.warning("No chunks to add to the database")
    
    def retrieve_relevant_documents(self, query, n_results=5):
        """Retrieve relevant documents based on the query."""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            relevant_docs = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                    distance = results['distances'][0][i] if results['distances'] else None
                    
                    relevant_docs.append({
                        "content": doc,
                        "metadata": metadata,
                        "distance": distance
                    })
            
            logger.info(f"Retrieved {len(relevant_docs)} relevant documents for query")
            return relevant_docs
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []
    
    def generate_answer(self, query, relevant_docs, max_context_length=4000):
        """Generate an answer using retrieved documents and Gemini."""
        if not relevant_docs:
            return "I don't have enough information to answer your question based on the available documents."
        
        # Prepare context from relevant documents
        context_parts = []
        current_length = 0
        
        for doc in relevant_docs:
            content = doc["content"]
            source = doc["metadata"].get("source", "Unknown")
            
            # Add source information
            doc_text = f"Source: {source}\n{content}\n\n"
            
            if current_length + len(doc_text) <= max_context_length:
                context_parts.append(doc_text)
                current_length += len(doc_text)
            else:
                break
        
        context = "".join(context_parts)
        
        # Create prompt for Gemini
        prompt = f"""Based on the following context information, please answer the user's question. If the answer is not found in the context, please say so clearly.

Context:
{context}

Question: {query}

Please provide a comprehensive and accurate answer based only on the information provided in the context. If you need to make any inferences, please make it clear that you are doing so."""

        try:
            # Generate response using Gemini
            response = self.google_client.models.generate_content(
                model="gemini-1.5-flash",
                contents=prompt
            )
            
            return response.text
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"Sorry, I encountered an error while generating the answer: {str(e)}"
    
    def answer_question(self, query, n_results=5):
        """Complete RAG pipeline: retrieve relevant docs and generate answer."""
        logger.info(f"Processing question: {query}")
        
        # Step 1: Retrieve relevant documents
        relevant_docs = self.retrieve_relevant_documents(query, n_results)
        
        if not relevant_docs:
            return "I couldn't find any relevant information to answer your question."
        
        # Step 2: Generate answer
        answer = self.generate_answer(query, relevant_docs)
        
        return answer
    
    def get_collection_info(self):
        """Get information about the collection."""
        try:
            count = self.collection.count()
            return f"Collection '{self.collection_name}' contains {count} documents/chunks"
        except Exception as e:
            return f"Error getting collection info: {e}"