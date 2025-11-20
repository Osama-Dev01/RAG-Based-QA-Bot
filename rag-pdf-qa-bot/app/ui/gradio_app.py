import gradio as gr

# This will be set from main.py
process_document_and_question = None

def set_processing_function(processing_func):
    """Set the processing function from main.py"""
    global process_document_and_question
    process_document_and_question = processing_func

def qa_interface():
    with gr.Blocks() as demo:
        gr.Markdown("## RAG PDF Question Answering Bot")

        with gr.Row():
            pdf_file = gr.File(label="Upload PDF Document", file_types=[".pdf"])

        question = gr.Textbox(label="Ask a Question", placeholder="Enter your question here...")

        with gr.Row():
            answer_box = gr.Textbox(label="Answer", interactive=False)
            sources_box = gr.Textbox(label="Sources / Retrieved Chunks", interactive=False)

        submit_btn = gr.Button("Get Answer")

        def process(pdf, query):
            if pdf is None or query.strip() == "":
                return "Please upload a PDF and ask a question.", "No sources available."
            
            # Call the processing function from main.py
            if process_document_and_question:
                return process_document_and_question(pdf, query)
            else:
                return "Processing function not set up.", "Please check backend."

        submit_btn.click(process, inputs=[pdf_file, question], outputs=[answer_box, sources_box])

    return demo