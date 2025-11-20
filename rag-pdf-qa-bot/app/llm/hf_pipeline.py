

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
from ui.gradio_app import qa_interface, set_processing_function
from embeddings.hf_embeddings import initialize_embedding_model





def initialize_llm(model_name: str = "google/flan-t5-small") -> Tuple[Optional[HuggingFacePipeline], str]:
    """
    Initializes the FLAN-T5 LLM for answer generation.
    
    Args:
        model_name: HuggingFace model name for FLAN-T5
        
    Returns:
        Tuple containing:
        - HuggingFacePipeline instance (LangChain LLM wrapper)
        - Status message
    """
    try:
        print(f"Initializing LLM: {model_name}")
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # Create transformers pipeline
        pipe = transformers_pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=512,
            temperature=0.1,
            do_sample=True,
            repetition_penalty=1.1
        )
        
        # Wrap in LangChain HuggingFacePipeline
        llm = HuggingFacePipeline(pipeline=pipe)
        
        # Test the LLM with a simple query
        test_query = "What is 2+2?"
        test_response = llm.invoke(test_query)
        print(f"LLM test response: {test_response}")
        
        print(f"LLM initialized successfully: {model_name}")
        
        return llm, f"LLM '{model_name}' loaded successfully"
        
    except Exception as e:
        error_msg = f"Error initializing LLM: {str(e)}"
        print(error_msg)
        return None, error_msg