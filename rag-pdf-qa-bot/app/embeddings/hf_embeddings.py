
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline as transformers_pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline

import os
import tempfile
from typing import Tuple , List , Optional
from ui.gradio_app import qa_interface, set_processing_function



from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

 

from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts import PromptTemplate
import numpy as np




def initialize_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> Tuple[HuggingFaceEmbeddings, str]:
    """
    Initializes the HuggingFace embedding model for converting text to vectors.
    
    Args:
        model_name: Name of the sentence transformer model to use
        
    Returns:
        Tuple containing:
        - HuggingFaceEmbeddings instance
        - Status message
    """
    try:
        print(f"Initializing embedding model: {model_name}")
        
        # Model configuration
        model_kwargs = {'device': 'cpu'}  # Use 'cuda' if GPU is available
        encode_kwargs = {'normalize_embeddings': True}  # Better for similarity search
        
        # Initialize the embedding model
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        
        # Test the model with a sample text to verify it's working
        sample_text = "This is a test sentence for embedding model."
        sample_embedding = embeddings.embed_query(sample_text)
        
        print(f"Embedding model initialized successfully!")
        print(f"Sample embedding dimension: {len(sample_embedding)}")
        print(f"Sample embedding first 5 values: {sample_embedding[:5]}")
        
        return embeddings, f"Embedding model '{model_name}' loaded successfully. Dimension: {len(sample_embedding)}"
        
    except Exception as e:
        error_msg = f"Error initializing embedding model: {str(e)}"
        print(error_msg)
        return None, error_msg

def create_embeddings_for_chunks(chunked_documents: List[Document], 
                                embeddings: HuggingFaceEmbeddings) -> Tuple[List[List[float]], str]:
    """
    Converts text chunks into numerical vectors (embeddings).
    
    Args:
        chunked_documents: List of Document objects (text chunks)
        embeddings: Initialized HuggingFaceEmbeddings instance
        
    Returns:
        Tuple containing:
        - List of embedding vectors for each chunk
        - Status message
    """
    try:
        print(f"Creating embeddings for {len(chunked_documents)} chunks...")
        
        if not chunked_documents:
            return [], "No documents to embed"
        
        if not embeddings:
            return [], "Embedding model not initialized"
        
        # Generate embeddings for all chunks
        chunk_texts = [doc.page_content for doc in chunked_documents]
        chunk_embeddings = embeddings.embed_documents(chunk_texts)
        
        # Calculate statistics
        embedding_dim = len(chunk_embeddings[0]) if chunk_embeddings else 0
        total_vectors = len(chunk_embeddings)
        
        print(f"Successfully created {total_vectors} embedding vectors")
        print(f"Embedding dimension: {embedding_dim}")
        print(f"First embedding sample: {chunk_embeddings[0][:5]}...")  # Show first 5 values
        
        # Show embedding statistics
        all_embeddings = np.array(chunk_embeddings)
        print(f"Embedding stats - Min: {np.min(all_embeddings):.4f}, "
              f"Max: {np.max(all_embeddings):.4f}, "
              f"Mean: {np.mean(all_embeddings):.4f}")
        
        status_message = (
            f"Created {total_vectors} embeddings with dimension {embedding_dim}. "
            f"Model: {embeddings.model_name}"
        )
        
        return chunk_embeddings, status_message
        
    except Exception as e:
        error_msg = f"Error creating embeddings: {str(e)}"
        print(error_msg)
        return [], error_msg
    
