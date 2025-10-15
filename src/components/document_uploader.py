import streamlit as st
from services.document_processor import DocumentProcessor

def upload_document():
    st.title("Document Uploader")
    uploaded_file = st.file_uploader("Choose a document...", type=["txt", "pdf", "docx"])
    
    if uploaded_file is not None:
        processor = DocumentProcessor()
        document_text = processor.process_document(uploaded_file)
        processor.store_embeddings(document_text)
        st.success("Document uploaded and processed successfully!")