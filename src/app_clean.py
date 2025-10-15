import streamlit as st
import uuid
from dotenv import load_dotenv
import os
import time

from services.vector_store import VectorStoreService
from services.document_processor import DocumentProcessor
from services.ollama_service import OllamaService
from components.advanced_search import AdvancedSearchInterface

# Load environment variables
load_dotenv()

# Initialize services
try:
    vector_store = VectorStoreService(clean_start=False)
except Exception as e:
    st.error(f"Could not initialize vector store: {str(e)}")
    st.stop()

document_processor = DocumentProcessor()
ollama_service = OllamaService()

def main():
    st.title("📚 RAG-powered Document Q&A")
    st.write("Upload documents and ask questions about them!")

    # Check Ollama connection in sidebar
    with st.sidebar:
        st.header("🔌 System Status")
        
        status_col, retry_col = st.columns([3, 1])
        
        with status_col:
            success, message = ollama_service.check_connection()
            if success:
                st.success("✅ Ollama Connection: OK")
            else:
                st.error(f"❌ Ollama Connection Error: {message}")
                st.warning("The Q&A functionality will not work without a proper Ollama connection.")
        
        with retry_col:
            if st.button("🔄 Retry"):
                with st.spinner("Checking connection..."):
                    success, message = ollama_service.check_connection()
                    if success:
                        st.success("✅ Connection restored!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("Still cannot connect")

    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["📚 Q&A Interface", "🔍 Advanced Search", "📄 Document Management"])
    
    # Advanced Search Tab
    with tab2:
        advanced_search = AdvancedSearchInterface(vector_store)
        advanced_search.render()

    # Document Management Tab
    with tab3:
        st.header("📄 Document Management")
        
        # Upload Section
        st.subheader("Upload New Document")
        
        # System Resources Status
        with st.expander("🖥️ System Resources", expanded=False):
            import psutil
            import torch
            
            cpu_count = psutil.cpu_count()
            memory = psutil.virtual_memory()
            memory_available = memory.available / (1024**3)
            memory_percent = memory.percent
            gpu_available = torch.cuda.is_available()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("CPU Cores", f"{cpu_count}")
                
            with col2:
                st.metric("Available RAM", f"{memory_available:.1f}GB ({100-memory_percent:.0f}% free)")
                
            with col3:
                if gpu_available:
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory
                    st.metric("GPU Memory", f"{gpu_memory/1e9:.1f}GB")
                else:
                    st.metric("GPU Status", "Not Available")
        
        # Document processing settings
        with st.expander("📝 Processing Settings", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                chunk_size = st.number_input("Chunk Size", min_value=100, max_value=2000, value=1000, step=100)
            with col2:
                chunk_overlap = st.number_input("Chunk Overlap", min_value=0, max_value=500, value=200, step=50)
            
            batch_size = st.slider("Batch Size (pages per batch)", min_value=1, max_value=50, value=10)
        
        # File Upload
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        
        if uploaded_file is not None:
            st.write(f"**File:** {uploaded_file.name}")
            st.write(f"**Size:** {uploaded_file.size / 1024:.2f} KB")
            
            if st.button("Process Document", type="primary"):
                try:
                    with st.spinner("Processing document..."):
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        def update_progress(current, total):
                            progress = current / total
                            progress_bar.progress(progress)
                            status_text.text(f"Processing page {current}/{total}...")
                        
                        # Process document
                        chunks = document_processor.process_pdf_stream(
                            uploaded_file,
                            callback=update_progress,
                            batch_size=batch_size
                        )
                        
                        # Generate IDs and metadata
                        doc_id = str(uuid.uuid4())
                        ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
                        metadatas = [
                            {"source": uploaded_file.name, "chunk": i, "page": i // 3 + 1}
                            for i in range(len(chunks))
                        ]
                        
                        # Add to vector store
                        vector_store.add_documents(chunks, metadatas, ids)
                        
                        progress_bar.progress(1.0)
                        status_text.text("✅ Document processed successfully!")
                        st.success(f"Document '{uploaded_file.name}' has been added to the knowledge base with {len(chunks)} chunks!")
                        time.sleep(2)
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"Error processing document: {str(e)}")
        
        # List existing documents
        st.subheader("📚 Existing Documents")
        
        available_sources = vector_store.get_unique_sources()
        
        if available_sources:
            st.write(f"**Total documents:** {len(available_sources)}")
            
            for source in available_sources:
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.write(f"📄 {source}")
                with col2:
                    if st.button("🗑️ Remove", key=f"remove_{source}"):
                        try:
                            vector_store.delete_documents(source)
                            st.success(f"Removed '{source}'")
                            time.sleep(1)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error removing document: {str(e)}")
        else:
            st.info("No documents uploaded yet. Upload a PDF to get started!")
        
        st.markdown("---")

    # Q&A Interface Tab
    with tab1:
        available_sources = vector_store.get_unique_sources()
        if not available_sources:
            st.info("No documents have been uploaded yet. Please go to the Document Management tab to upload documents.")
        else:
            # Document selection for search
            selected_sources = st.multiselect(
                "Select documents to search in:",
                available_sources,
                default=available_sources,
                help="Choose one or more documents to search in."
            )

            # Chat interface
            st.subheader("💬 Ask a Question")
            
            question = st.text_area("Your question:", placeholder="What would you like to know?", height=100)
            
            # Advanced search settings
            with st.expander("⚙️ Search Settings"):
                col1, col2 = st.columns(2)
                with col1:
                    num_results = st.slider("Number of context chunks", min_value=3, max_value=20, value=5)
                with col2:
                    temperature = st.slider("Response creativity", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
            
            if st.button("🔍 Get Answer", type="primary", disabled=not question):
                if question:
                    with st.spinner("Searching documents and generating response..."):
                        try:
                            # Search documents
                            if selected_sources:
                                results = vector_store.search_documents_in_source(
                                    query=question,
                                    source=selected_sources,
                                    n_results=num_results
                                )
                            else:
                                results = vector_store.search_documents(
                                    query=question,
                                    n_results=num_results
                                )
                            
                            # Extract context
                            if results and results.get("documents") and results["documents"][0]:
                                contexts = results["documents"][0]
                                metadatas = results["metadatas"][0] if results.get("metadatas") else []
                                
                                # Generate response
                                response = ollama_service.generate_response(
                                    prompt=question,
                                    context=contexts,
                                    temperature=temperature
                                )
                                
                                # Display response
                                st.markdown("### 💡 Answer")
                                st.markdown(response)
                                
                                # Display sources
                                st.markdown("### 📚 Sources")
                                for i, (context, metadata) in enumerate(zip(contexts, metadatas), 1):
                                    with st.expander(f"Source {i}: {metadata.get('source', 'Unknown')} (Page {metadata.get('page', 'N/A')})"):
                                        st.text(context[:500] + "..." if len(context) > 500 else context)
                            else:
                                st.warning("No relevant information found in the selected documents.")
                        
                        except Exception as e:
                            st.error(f"Error generating response: {str(e)}")

if __name__ == "__main__":
    main()
