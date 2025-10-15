from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings

class VectorStoreService:
    def __init__(self, persist_directory: str = "data/vector_store", clean_start: bool = False):
        """Initialize ChromaDB client with persistence
        
        Args:
            persist_directory (str): Directory to store the vector database
            clean_start (bool): If True, attempts to clean the existing database
        """
        import os
        import shutil
        import logging
        import time
        
        self.logger = logging.getLogger(__name__)
        self.persist_directory = persist_directory
        
        try:
            if clean_start and os.path.exists(persist_directory):
                self.logger.info(f"Attempting to clean vector store at {persist_directory}")
                max_retries = 3
                retry_delay = 1  # seconds
                
                for attempt in range(max_retries):
                    try:
                        # Try to create a temporary ChromaDB instance to ensure all connections are closed
                        temp_settings = Settings(
                            persist_directory=persist_directory,
                            is_persistent=True,
                            anonymized_telemetry=False
                        )
                        temp_client = chromadb.PersistentClient(settings=temp_settings)
                        temp_client.reset()  # This closes connections properly
                        del temp_client  # Explicitly delete the temporary client
                        
                        # Now try to remove the directory
                        if os.path.exists(persist_directory):
                            shutil.rmtree(persist_directory)
                        break
                    except (PermissionError, OSError) as e:
                        if attempt == max_retries - 1:
                            self.logger.warning(f"Could not clean vector store after {max_retries} attempts: {e}")
                            self.logger.warning("Proceeding with existing vector store")
                        else:
                            self.logger.info(f"Retry {attempt + 1}/{max_retries} after {retry_delay}s...")
                            time.sleep(retry_delay)
            
            # Create directory if it doesn't exist
            os.makedirs(persist_directory, exist_ok=True)
            
            # Use Settings for better control
            settings = Settings(
                persist_directory=persist_directory,
                is_persistent=True,
                anonymized_telemetry=False
            )
            
            # Initialize the client with settings
            self.client = chromadb.Client(settings)
            
            # Get or create collection with embedding function
            self.collection = self.client.get_or_create_collection(
                name="documents",
                metadata={
                    "hnsw:space": "cosine",
                    "description": "Document collection for RAG application"
                }
            )
            
            self.logger.info("Successfully initialized vector store")
            
        except Exception as e:
            self.logger.error(f"Error initializing vector store: {str(e)}")
            raise
    
    def add_documents(self, texts: List[str], metadatas: List[Dict], ids: List[str]) -> None:
        """
        Add documents to the vector store
        
        Args:
            texts: List of document texts
            metadatas: List of metadata dictionaries for each document
            ids: List of unique IDs for each document
        """
        self.collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
    
    def search_documents(self, query: str, n_results: int = 3) -> Dict:
        """
        Search for similar documents
        
        Args:
            query: Search query text
            n_results: Number of results to return
            
        Returns:
            Dictionary containing search results
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        return results
    
    def delete_document(self, document_id: str) -> None:
        """
        Delete a document from the vector store
        
        Args:
            document_id: ID of the document to delete
        """
        self.collection.delete(ids=[document_id])
    
    def list_documents(self) -> List[str]:
        """
        List all document IDs in the collection
        
        Returns:
            List of document IDs
        """
        return self.collection.get()["ids"]
    
    def get_all_documents(self) -> Dict:
        """
        Get all documents from the collection
        
        Returns:
            Dictionary containing all documents with their metadata
        """
        return self.collection.get()
        
    def get_unique_sources(self) -> List[str]:
        """
        Get list of unique document sources
        
        Returns:
            List of unique source names
        """
        all_docs = self.collection.get()
        if all_docs and all_docs["metadatas"]:
            return list(set(meta["source"] for meta in all_docs["metadatas"] if "source" in meta))
        return []
        
    def delete_documents(self, source: str) -> None:
        """
        Delete all documents from a specific source
        
        Args:
            source: The source name of documents to delete
        """
        # Get all documents from this source
        results = self.collection.get(
            where={"source": source}
        )
        
        if results and results["ids"]:
            # Delete the documents
            self.collection.delete(
                ids=results["ids"]
            )

    def search_documents_in_source(
        self, 
        query: str, 
        source: Optional[List[str]] = None, 
        n_results: int = 3,
        offset: int = 0
    ) -> Dict:
        """
        Search for similar documents with pagination, optionally filtering by source
        
        Args:
            query: Search query text
            source: Optional list of source names to filter by
            n_results: Number of results per page
            offset: Number of results to skip (for pagination)
            
        Returns:
            Dictionary containing search results and pagination info
        """
        # First get total count
        total_results = self.collection.query(
            query_texts=[query],
            where={"source": {"$in": source}} if source else None,
            n_results=1000  # Use a large number to get total count
        )
        total_count = len(total_results["ids"][0]) if total_results["ids"] else 0
        
        # Then get paginated results
        if source:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results + offset,  # Get results including offset
                where={"source": {"$in": source}}
            )
        else:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results + offset
            )
        
        # Apply pagination manually
        paginated_results = {
            "ids": [results["ids"][0][offset:offset + n_results]],
            "embeddings": [results["embeddings"][0][offset:offset + n_results]] if results.get("embeddings") else None,
            "documents": [results["documents"][0][offset:offset + n_results]],
            "metadatas": [results["metadatas"][0][offset:offset + n_results]] if results.get("metadatas") else None,
            "distances": [results["distances"][0][offset:offset + n_results]] if results.get("distances") else None,
            "pagination": {
                "total": total_count,
                "offset": offset,
                "limit": n_results,
                "has_more": (offset + n_results) < total_count
            }
        }
        
        return paginated_results

    def advanced_search(
        self,
        query: str,
        n_results: int = 10,
        filters: Optional[Dict] = None,
        min_similarity: Optional[float] = None,
        offset: int = 0
    ) -> Dict:
        """
        Advanced search with multiple filter options
        
        Args:
            query: Search query text for semantic similarity
            n_results: Number of results to return
            filters: Dictionary of metadata filters (e.g., {"source": "doc1.pdf", "page": 5})
            min_similarity: Minimum similarity threshold (0-1, lower distance = higher similarity)
            offset: Number of results to skip for pagination
            
        Returns:
            Dictionary containing filtered search results
        """
        # Build where clause from filters
        where_clause = None
        if filters:
            where_conditions = {}
            for key, value in filters.items():
                if isinstance(value, list):
                    where_conditions[key] = {"$in": value}
                else:
                    where_conditions[key] = value
            where_clause = where_conditions if where_conditions else None
        
        # Perform search
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results + offset,
            where=where_clause
        )
        
        # Apply similarity threshold if specified
        if min_similarity is not None and results["distances"]:
            max_distance = 1.0 - min_similarity  # Convert similarity to distance
            filtered_indices = [
                i for i, dist in enumerate(results["distances"][0])
                if dist <= max_distance
            ]
            
            results = {
                "ids": [[results["ids"][0][i] for i in filtered_indices]],
                "documents": [[results["documents"][0][i] for i in filtered_indices]],
                "metadatas": [[results["metadatas"][0][i] for i in filtered_indices]] if results.get("metadatas") else None,
                "distances": [[results["distances"][0][i] for i in filtered_indices]],
            }
        
        # Apply pagination
        paginated_results = {
            "ids": [results["ids"][0][offset:offset + n_results]],
            "documents": [results["documents"][0][offset:offset + n_results]],
            "metadatas": [results["metadatas"][0][offset:offset + n_results]] if results.get("metadatas") else None,
            "distances": [results["distances"][0][offset:offset + n_results]] if results.get("distances") else None,
        }
        
        return paginated_results
    
    def regex_search(
        self,
        pattern: str,
        field: str = "document",
        metadata_filters: Optional[Dict] = None,
        case_sensitive: bool = False
    ) -> Dict:
        """
        Search documents using regular expressions
        
        Args:
            pattern: Regular expression pattern
            field: Field to search in ('document' or a metadata field name)
            metadata_filters: Optional metadata filters to apply
            case_sensitive: Whether the search should be case-sensitive
            
        Returns:
            Dictionary containing matching documents
        """
        import re
        
        # Get all documents
        all_docs = self.collection.get(
            where=metadata_filters if metadata_filters else None
        )
        
        if not all_docs or not all_docs["ids"]:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "matches": [[]]}
        
        # Compile regex pattern
        flags = 0 if case_sensitive else re.IGNORECASE
        regex = re.compile(pattern, flags)
        
        # Filter documents based on regex
        matching_indices = []
        matches_info = []
        
        for i, doc_id in enumerate(all_docs["ids"]):
            text_to_search = ""
            
            if field == "document":
                text_to_search = all_docs["documents"][i] if all_docs["documents"] else ""
            elif all_docs["metadatas"] and i < len(all_docs["metadatas"]):
                text_to_search = str(all_docs["metadatas"][i].get(field, ""))
            
            matches = list(regex.finditer(text_to_search))
            if matches:
                matching_indices.append(i)
                matches_info.append([{
                    "match": m.group(),
                    "start": m.start(),
                    "end": m.end(),
                    "context": text_to_search[max(0, m.start()-50):min(len(text_to_search), m.end()+50)]
                } for m in matches[:5]])  # Limit to first 5 matches per document
        
        # Build results
        results = {
            "ids": [[all_docs["ids"][i] for i in matching_indices]],
            "documents": [[all_docs["documents"][i] for i in matching_indices]] if all_docs["documents"] else [[]],
            "metadatas": [[all_docs["metadatas"][i] for i in matching_indices]] if all_docs["metadatas"] else [[]],
            "matches": [matches_info]
        }
        
        return results
    
    def metadata_search(
        self,
        metadata_filters: Dict,
        include_embeddings: bool = False
    ) -> Dict:
        """
        Search documents by metadata only (no semantic search)
        
        Args:
            metadata_filters: Dictionary of metadata conditions
            include_embeddings: Whether to include embeddings in results
            
        Returns:
            Dictionary containing matching documents
        """
        # Build where clause
        where_clause = {}
        for key, value in metadata_filters.items():
            if isinstance(value, dict):
                # Support for operators like $in, $gt, $lt, etc.
                where_clause[key] = value
            elif isinstance(value, list):
                where_clause[key] = {"$in": value}
            else:
                where_clause[key] = value
        
        # Get matching documents
        results = self.collection.get(
            where=where_clause if where_clause else None,
            include=["documents", "metadatas", "embeddings"] if include_embeddings else ["documents", "metadatas"]
        )
        
        return results
    
    def semantic_search_with_metadata(
        self,
        query: str,
        metadata_filters: Optional[Dict] = None,
        n_results: int = 10,
        similarity_threshold: float = 0.5
    ) -> Dict:
        """
        Combine semantic similarity search with metadata filtering
        
        Args:
            query: Search query for semantic matching
            metadata_filters: Metadata conditions to filter results
            n_results: Maximum number of results
            similarity_threshold: Minimum similarity score (0-1)
            
        Returns:
            Dictionary containing filtered results with similarity scores
        """
        return self.advanced_search(
            query=query,
            n_results=n_results,
            filters=metadata_filters,
            min_similarity=similarity_threshold
        )