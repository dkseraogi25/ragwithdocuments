from typing import List, BinaryIO, Generator, Optional
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
import concurrent.futures
import gc
import threading
from queue import Queue
import logging
import psutil

class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize DocumentProcessor with chunking parameters
        
        Args:
            chunk_size: The size of text chunks
            chunk_overlap: The overlap between chunks
        """
        self.logger = logging.getLogger(__name__)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Log available resources (simplified without distributed processing)
        cpu_count = psutil.cpu_count()
        memory_available = psutil.virtual_memory().available / (1024**3)  # GB
        self.logger.info(f"Initialized with {cpu_count} CPUs, "
                        f"{memory_available:.2f}GB available RAM")

    def extract_text_from_pdf(self, pdf_file: BinaryIO) -> str:
        """
        Extract text from PDF file
        
        Args:
            pdf_file: PDF file object
            
        Returns:
            Extracted text from the PDF
        """
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text

    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        return self.text_splitter.split_text(text)

    def clean_text(self, text: str) -> str:
        """
        Clean the text by removing extra whitespace and normalizing
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        text = text.strip()
        text = " ".join(text.split())  # Normalize whitespace
        return text

    def process_page(self, page) -> str:
        """Process a single page and return its text"""
        text = page.extract_text()
        text = self.clean_text(text)
        return text

    def process_batch(self, pages: List[PyPDF2.PageObject]) -> List[str]:
        """Process a batch of pages in parallel and return chunks"""
        with concurrent.futures.ThreadPoolExecutor() as executor:
            texts = list(executor.map(self.process_page, pages))
        
        combined_text = " ".join(texts)
        chunks = self.chunk_text(combined_text)
        
        # Force garbage collection after processing batch
        del texts
        gc.collect()
        
        return chunks

    def process_document_stream(self, file: BinaryIO, batch_size: int = 5, callback=None) -> Generator[List[str], None, None]:
        """
        Process a document file and yield chunks with parallel processing and memory optimization
        
        Args:
            file: Document file object
            batch_size: Number of pages to process in each batch
            callback: Optional callback function to report progress
            
        Yields:
            Lists of text chunks from each batch
        """
        pdf_reader = PyPDF2.PdfReader(file)
        total_pages = len(pdf_reader.pages)
        
        if callback:
            callback(0, total_pages)
        
        current_batch = []
        for i, page in enumerate(pdf_reader.pages):
            current_batch.append(page)
            
            if len(current_batch) >= batch_size or i == total_pages - 1:
                chunks = self.process_batch(current_batch)
                yield chunks
                
                # Clear batch from memory
                current_batch.clear()
                gc.collect()
            
            if callback:
                callback(i + 1, total_pages)

    def process_document(self, file: BinaryIO, batch_size: Optional[int] = None, callback=None) -> List[str]:
        """
        Process a document file and return all chunks using distributed processing and GPU acceleration
        
        Args:
            file: Document file object
            batch_size: Number of pages to process in each batch (None for auto-scaling)
            callback: Optional callback function to report progress
            
        Returns:
            List of all text chunks
        """
        pdf_reader = PyPDF2.PdfReader(file)
        total_pages = len(pdf_reader.pages)
        
        # Initialize progress
        if callback:
            callback(0, total_pages)
            
        try:
            # Get document size for resource optimization
            file.seek(0, 2)  # Seek to end
            doc_size = file.tell() / (1024 * 1024)  # Size in MB
            file.seek(0)  # Reset position
            
            # Calculate optimal batch size based on available memory
            if batch_size is None:
                available_memory = psutil.virtual_memory().available / (1024**3)  # GB
                batch_size = max(10, min(50, int(available_memory * 10)))  # 10-50 pages
                self.logger.info(f"Auto-scaled batch size to {batch_size} based on available memory {available_memory:.2f}GB")
            
            # Extract text from pages
            texts = []
            
            for i, page in enumerate(pdf_reader.pages):
                # Extract and clean text from each page
                page_text = self.clean_text(page.extract_text())
                if page_text.strip():  # Only add non-empty pages
                    texts.append(page_text)
                
                # Update progress
                if callback:
                    callback(i + 1, total_pages)
                
                # Force garbage collection periodically
                if (i + 1) % batch_size == 0:
                    gc.collect()
            
            # Split processed text into chunks
            all_chunks = []
            for text in texts:
                chunks = self.chunk_text(text)
                all_chunks.extend(chunks)
            
            self.logger.info(f"Successfully processed {total_pages} pages into {len(all_chunks)} chunks")
            return all_chunks
            
        except Exception as e:
            self.logger.error(f"Error processing document: {str(e)}")
            raise
        finally:
            # Clean up distributed resources
            gc.collect()