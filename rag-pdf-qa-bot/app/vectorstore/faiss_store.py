
import os
import io
import hashlib
import json
from typing import Tuple, List


import gradio as gr
import numpy as np


from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

 

from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline as transformers_pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline

import os
import tempfile
from typing import Tuple , List , Optional


def create_vector_store(chunked_documents: List[Document], 
                       embeddings: HuggingFaceEmbeddings) -> Tuple[Optional[FAISS], str]:
    """
    Creates a FAISS vector store from document chunks and their embeddings.
    """
    try:
        print(f"Creating FAISS vector store for {len(chunked_documents)} chunks...")
        
        if not chunked_documents:
            return None, "No documents to index"
        
        if not embeddings:
            return None, "Embedding model not initialized"
        
        # Create FAISS vector store from documents
        vector_store = FAISS.from_documents(
            documents=chunked_documents,
            embedding=embeddings
        )
        
        # Get index statistics
        total_vectors = vector_store.index.ntotal
        
        print(f"FAISS vector store created successfully!")
        print(f"Total vectors in index: {total_vectors}")
        
        status_message = f"FAISS index created with {total_vectors} vectors"
        
        return vector_store, status_message
        
    except Exception as e:
        error_msg = f"Error creating vector store: {str(e)}"
        print(error_msg)
        return None, error_msg
