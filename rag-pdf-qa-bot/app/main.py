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
from llm.hf_pipeline import initialize_llm
from vectorstore.faiss_store import create_vector_store




def load_pdf_documents(pdf_file) -> Tuple[List[Document], str]:
    """
    Loads PDF file and extracts text from each page as Document objects.
    
    Args:
        pdf_file: The uploaded PDF file object from Gradio
        
    Returns:
        Tuple containing:
        - List of Document objects (one per page)
        - Status message indicating success or error
        
    Raises:
        Exception: If PDF loading fails
    """
    try:
        print(f"Loading PDF: {pdf_file.name}")
        
        # Create temporary file for PDF processing
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            # Copy uploaded file content to temporary file
            with open(pdf_file.name, 'rb') as uploaded_file:
                tmp_file.write(uploaded_file.read())
            temp_pdf_path = tmp_file.name
        
        # Initialize PDF loader with the temporary file path
        pdf_loader = PyPDFLoader(temp_pdf_path)
        
        # Load and split the PDF into Document objects (one per page)
        documents = pdf_loader.load()
        
        # Print document statistics for debugging
        print(f"Successfully loaded {len(documents)} pages from PDF")
        
        # Display sample of first page content
        if documents:
            first_page_content = documents[0].page_content
            print(f"First page preview: {first_page_content[:200]}...")
        
        # Clean up temporary file
        os.unlink(temp_pdf_path)
        
        return documents, f"Successfully loaded {len(documents)} pages"
        
    except Exception as e:
        error_msg = f"Error loading PDF: {str(e)}"
        print(error_msg)
        
        # Clean up temporary file if it exists
        if 'temp_pdf_path' in locals() and os.path.exists(temp_pdf_path):
            os.unlink(temp_pdf_path)
            
        return [], error_msg





def split_documents_into_chunks(documents: List[Document], 
                               chunk_size: int = 1000, 
                               chunk_overlap: int = 200) -> Tuple[List[Document], str]:
    """
    Splits documents into smaller, overlapping chunks for better processing.
    
    Args:
        documents: List of Document objects from PDF loader
        chunk_size: Maximum size of each text chunk (in characters)
        chunk_overlap: Overlap between chunks to maintain context
        
    Returns:
        Tuple containing:
        - List of chunked Document objects
        - Status message with splitting statistics
    """
    try:
        print(f"Splitting {len(documents)} documents into chunks...")
        print(f"Chunk size: {chunk_size}, Overlap: {chunk_overlap}")
        
        # Calculate total characters before splitting
        total_chars_before = sum(len(doc.page_content) for doc in documents)
        print(f"Total characters before splitting: {total_chars_before}")
        
        # Initialize the text splitter with specified parameters
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]  # Hierarchical splitting
        )
        
        # Split the documents into chunks
        chunked_documents = text_splitter.split_documents(documents)
        
        # Calculate statistics after splitting
        total_chars_after = sum(len(doc.page_content) for doc in chunked_documents)
        avg_chunk_size = total_chars_after / len(chunked_documents) if chunked_documents else 0
        
        # Print detailed statistics
        print(f"Created {len(chunked_documents)} chunks from {len(documents)} original documents")
        print(f"Total characters after splitting: {total_chars_after}")
        print(f"Average chunk size: {avg_chunk_size:.2f} characters")
        
        # Show sample chunks for debugging
        if chunked_documents:
            print("\nSample chunks:")
            for i, chunk in enumerate(chunked_documents[:3]):  # Show first 3 chunks
                print(f"Chunk {i+1}: {chunk.page_content[:100]}...")
                print(f"  - Length: {len(chunk.page_content)} characters")
                print(f"  - Metadata: {chunk.metadata}")
        
        status_message = (
            f"Split {len(documents)} pages into {len(chunked_documents)} chunks. "
            f"Avg chunk size: {avg_chunk_size:.0f} chars"
        )
        
        return chunked_documents, status_message
        
    except Exception as e:
        error_msg = f"Error splitting documents: {str(e)}"
        print(error_msg)
        return [], error_msg













def create_retrieval_qa_chain(llm: HuggingFacePipeline, 
                             vector_store: FAISS, 
                             k: int = 4) -> Tuple[Optional[RetrievalQA], str]:
    """
    Creates a RetrievalQA chain that combines retriever + LLM.
    
    Args:
        llm: Initialized LangChain LLM
        vector_store: FAISS vector store
        k: Number of documents to retrieve
        
    Returns:
        Tuple containing:
        - RetrievalQA chain instance
        - Status message
    """
    try:
        print(f"Creating RetrievalQA chain with k={k}")
        
        if not llm:
            return None, "LLM not initialized"
        
        if not vector_store:
            return None, "Vector store not initialized"
        
        # Create retriever from vector store
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )
        
        # Custom prompt template for better answers
        prompt_template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        Context:
        {context}

        Question: {question}

        Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create RetrievalQA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",  # "stuff" means put all docs in context
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        print("RetrievalQA chain created successfully!")
        
        return qa_chain, f"QA chain ready with retriever (k={k})"
        
    except Exception as e:
        error_msg = f"Error creating RetrievalQA chain: {str(e)}"
        print(error_msg)
        return None, error_msg

def get_answer_from_qa_chain(qa_chain: RetrievalQA, 
                           question: str) -> Tuple[str, List[Document], str]:
    """
    Uses the RetrievalQA chain to generate an answer with sources.
    
    Args:
        qa_chain: Initialized RetrievalQA chain
        question: User's question
        
    Returns:
        Tuple containing:
        - Generated answer
        - List of source documents
        - Status message
    """
    try:
        print(f"Getting answer for question: '{question}'")
        
        if not qa_chain:
            return "", [], "QA chain not initialized"
        
        # Invoke the QA chain
        result = qa_chain.invoke({"query": question})
        
        # Extract answer and source documents
        answer = result["result"]
        source_documents = result["source_documents"]
        
        print(f"Answer generated successfully!")
        print(f"Answer: {answer}")
        print(f"Number of source documents: {len(source_documents)}")
        
        # Log source document details
        for i, doc in enumerate(source_documents):
            print(f"Source {i+1} - Page {doc.metadata.get('page', 'N/A')}: "
                  f"{doc.page_content[:100]}...")
        
        status_message = f"Generated answer using {len(source_documents)} source documents"
        
        return answer, source_documents, status_message
        
    except Exception as e:
        error_msg = f"Error getting answer from QA chain: {str(e)}"
        print(error_msg)
        return "", [], error_msg

def format_sources(source_documents: List[Document]) -> str:
    """
    Formats source documents into a readable string with page numbers.
    
    Args:
        source_documents: List of source documents from QA chain
        
    Returns:
        Formatted sources string
    """
    if not source_documents:
        return "No sources available"
    
    sources_list = []
    for i, doc in enumerate(source_documents):
        page_num = doc.metadata.get('page', 'Unknown')
        content_preview = doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
        sources_list.append(f"Source {i+1} (Page {page_num}): {content_preview}")
    
    return "\n\n".join(sources_list)














def process_pdf_and_question(pdf_file, question: str) -> Tuple[str, str]:
    """
    This function receives the uploaded document and question from Gradio
    """
    try:
        print(f"Processing PDF: {pdf_file.name}")
        print(f"User question: {question}")



        documents, load_status = load_pdf_documents(pdf_file)
        
        if not documents:
            return f"Error: {load_status}", "No sources available"
        
        # Step 2: Process the documents with RAG pipeline
        # (This is where you'll add text splitting, embedding, and retrieval)
        
        # For now, return basic information about the loaded documents
        total_pages = len(documents)
        total_chars = sum(len(doc.page_content) for doc in documents)

        print("total pages" , total_pages)

        chunk_size = 1000  # Characters per chunk
        chunk_overlap = 200  # Overlap between chunks
        
        chunked_documents, split_status = split_documents_into_chunks(
            documents, 
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )
        
        if not chunked_documents:
            return f"Error: {split_status}", "No sources available"




        embedding_model, model_status = initialize_embedding_model()
        
        if not embedding_model:
            return f"Error: {model_status}", "No sources available"
        
        

        


        vector_store, vector_status = create_vector_store(chunked_documents, embedding_model)
        
        if not vector_store:
            return f"Error: {vector_status}", "No sources available"
        
        # Step 5: Initialize LLM (FLAN-T5)
        llm, llm_status = initialize_llm("google/flan-t5-small")
        
        if not llm:
            return f"Error: {llm_status}", "No sources available"
        
        # Step 6: Create RetrievalQA chain
        qa_chain, chain_status = create_retrieval_qa_chain(llm, vector_store, k=4)
        
        if not qa_chain:
            return f"Error: {chain_status}", "No sources available"
        
        # Step 7: Get answer using the QA chain
        answer, source_documents, answer_status = get_answer_from_qa_chain(qa_chain, question)
        
        # Step 8: Format the response
        sources = format_sources(source_documents)
        
        final_answer = f"Question: {question}\n\nAnswer: {answer}"
        
        return final_answer, sources





        
        
        
    except Exception as e:
        return f"Error: {str(e)}", "No sources available"

def main():
    # Connect the processing function to Gradio
    set_processing_function(process_pdf_and_question)
    
    # Launch the app
    app = qa_interface()
    app.launch(server_name="127.0.0.1", server_port=7860)


if __name__ == "__main__":
    main()

