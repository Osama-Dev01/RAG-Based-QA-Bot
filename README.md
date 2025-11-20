# RAG-based PDF Question-Answering Bot

A Retrieval-Augmented Generation (RAG) system that allows users to upload any PDF document and ask questions about its content. The system extracts text, chunks it, embeds it, retrieves relevant sections, and generates accurate answers along with source references.

## Features

- Upload any PDF and ask context-aware questions
- Extracts text using PyPDFLoader
- Splits text into overlapping chunks for accurate semantic retrieval
- Uses HuggingFace sentence-transformers (all-MiniLM-L6-v2) for embeddings
- FAISS vector database for fast similarity search
- FLAN-T5 LLM (via HuggingFacePipeline) for generating grounded answers
- Full RAG pipeline powered by LangChain
- Gradio-based frontend UI for interactive querying
- Returns answer along with source chunks and page numbers



## Architecture

![Architecture Diagram]([https://example.com/path/to/your/image.png](https://github.com/Osama-Dev01/RAG-Based-QA-Bot/blob/main/rag-pdf-qa-bot/docs/ChatGPT%20Image%20Nov%2020%2C%202025%2C%2009_41_06%20PM.png))

### PDF Processing Pipeline
- **PDF Loader**: Uses PyPDFLoader to extract text from PDF pages
- **Text Splitter**: Implements recursive character text splitting with configurable chunk size and overlap
- **Embedding Model**: Leverages sentence-transformers for generating semantic embeddings
- **Vector Database**: FAISS for efficient similarity search and retrieval
- **LLM Integration**: FLAN-T5 model for answer generation
- **RetrievalQA Chain**: LangChain pipeline that combines retrieval and generation

### Configuration
- Chunk Size: 1000 characters
- Chunk Overlap: 200 characters
- Embedding Model: sentence-transformers/all-MiniLM-L6-v2
- LLM Model: google/flan-t5-base
- Retrieval Count: 4 most relevant chunks

## Technical Details

### Text Processing
The system processes PDF documents by:
1. Loading PDF content page by page
2. Splitting text into manageable chunks with overlapping sections
3. Generating embeddings for each chunk using HuggingFace sentence transformers
4. Storing embeddings in FAISS vector database for efficient retrieval

### Question Answering
When a question is submitted:
1. The system generates an embedding for the question
2. Performs similarity search to find the most relevant text chunks
3. Passes the retrieved context and question to the FLAN-T5 LLM
4. Generates a natural language answer based on the provided context
5. Returns the answer along with source document references

### Source Attribution
The system maintains source metadata including:
- Original page numbers
- Document source information
- Similarity scores for retrieved chunks

## Use Cases 
Academic research  
Legal document analysis  
Technical documentation extraction  
Literature review automation  
Business reports and proposals  
Policy document question answering  
