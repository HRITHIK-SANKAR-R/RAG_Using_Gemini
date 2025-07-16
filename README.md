# RAG System with Gemini AI

A complete Retrieval-Augmented Generation system using ChromaDB for vector storage and Google's Gemini AI for intelligent question answering.

## Features

- **Document Processing**: Automatically loads and processes text documents
- **Smart Chunking**: Splits documents into overlapping chunks for better context
- **Vector Search**: ChromaDB with Google embeddings for semantic similarity
- **AI Responses**: Gemini AI generates comprehensive answers from retrieved context
- **Interactive Q&A**: Simple command-line interface for asking questions
- **Persistent Storage**: ChromaDB maintains your document index across sessions

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up your Gemini API key**:
   Create a `.env` file with your API key:
   ```
   GEMINI_API_KEY="your-api-key-here"
   ```

3. **Run the RAG system**:
   ```bash
   python test_rag.py
   ```

4. **Start asking questions** about the news articles!

## Usage

Simply run `python test_rag.py` and start asking questions:

```
💬 Your question: What is ChatGPT?
🔍 Processing your question...

🤖 Answer:
ChatGPT is an AI-powered chatbot developed by OpenAI...
```

**Available commands:**
- Type any question to get an AI-generated answer
- `help` - Show sample questions
- `quit` or `exit` - Exit the system

## System Architecture

- **`app.py`** - Core RAG system implementation
- **`test_rag.py`** - Main interactive interface
- **`validate_system.py`** - System validation script
- **`news_articles/`** - Sample documents for testing
- **`chromadb_database/`** - Persistent vector database

## How It Works

1. **Document Loading**: Reads all `.txt` files from `news_articles/`
2. **Text Chunking**: Splits documents into overlapping chunks
3. **Embedding**: Creates vector embeddings using Google's embedding model
4. **Storage**: Stores vectors in ChromaDB for fast similarity search
5. **Retrieval**: Finds most relevant chunks for your question
6. **Generation**: Gemini AI creates answers based on retrieved context
   - Get your Gemini API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Replace `your_api_key_here` with your actual API key in the `.env` file

4. **Prepare your documents**:
   - Place your text documents (.txt files) in the `news_articles` directory
   - The system will automatically process all .txt files in this directory

## Usage

### Running the Interactive System

```bash
python app.py
```

This will:
1. Initialize the RAG system
2. Load and process documents (if not already done)
3. Start an interactive Q&A session

### Example Questions

Based on the included news articles, you can ask questions like:
- "What is ChatGPT?"
- "Tell me about AI startups and investments"
- "What are the latest developments in generative AI?"
- "What did Google announce at I/O 2023?"

### Using the RAG System Programmatically

```python
from app import RAGSystem, load_docs

# Initialize the system
rag = RAGSystem()

# Load documents
docs = load_docs("./news_articles")
rag.add_documents(docs)

# Ask a question
answer = rag.answer_question("What is ChatGPT?")
print(answer)
```

## System Architecture

### Components

1. **Document Loader** (`load_docs`):
   - Reads text files from a specified directory
   - Handles encoding and error cases

2. **Text Splitter** (`split_text`):
   - Splits documents into overlapping chunks
   - Configurable chunk size and overlap

3. **RAGSystem Class**:
   - **Vector Database**: ChromaDB with Google embeddings
   - **Retrieval**: Semantic search for relevant chunks
   - **Generation**: Gemini-powered answer generation

### Workflow

1. **Document Processing**:
   ```
   Raw Documents → Text Chunks → Embeddings → Vector Database
   ```

2. **Question Answering**:
   ```
   User Query → Embedding → Similarity Search → Relevant Chunks → Context + Query → Gemini → Answer
   ```

## Configuration

### Environment Variables

- `GEMINI_API_KEY`: Your Google Gemini API key

### System Parameters

You can customize the system by modifying these parameters in the code:

- `chunk_size`: Size of text chunks (default: 1000 characters)
- `chunk_overlap`: Overlap between chunks (default: 200 characters)
- `n_results`: Number of relevant chunks to retrieve (default: 5)
- `max_context_length`: Maximum context length for answer generation (default: 4000 characters)

## File Structure

```
RAG/
├── app.py                 # Main RAG system implementation
├── requirements.txt       # Python dependencies
├── .env.template         # Environment variables template
├── .env                  # Your environment variables (create this)
├── chromadb_database/    # Persistent vector database (auto-created)
└── news_articles/        # Directory containing text documents
    ├── article1.txt
    ├── article2.txt
    └── ...
```

## Error Handling

The system includes comprehensive error handling for:
- Missing API keys
- Document loading errors
- Database connection issues
- API call failures
- Empty or invalid queries

## Logging

The system uses Python's logging module to provide detailed information about:
- Document loading progress
- Database operations
- Query processing
- Error conditions

## Limitations

This is a "naive" RAG implementation, which means:
- Simple text chunking (no semantic awareness)
- Basic similarity search (no re-ranking)
- No query expansion or refinement
- Limited context window management

## Potential Improvements

- Implement semantic-aware chunking
- Add query expansion and refinement
- Implement re-ranking of retrieved chunks
- Add conversation memory
- Support for different document formats
- Add evaluation metrics
- Implement query routing for different types of questions

## Troubleshooting

### Common Issues

1. **API Key Error**: Make sure your `.env` file contains a valid Gemini API key
2. **No Documents Found**: Ensure your text files are in the `news_articles` directory
3. **Import Errors**: Install all required dependencies with `pip install -r requirements.txt`

### Performance Tips

- For large document collections, consider increasing chunk size
- Adjust the number of retrieved chunks based on your needs
- Monitor API usage to stay within rate limits

## License

This project is for educational purposes. Please ensure you comply with Google's API usage terms and conditions.
