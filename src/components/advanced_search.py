import streamlit as st
from typing import Dict, List, Optional
import re

class AdvancedSearchInterface:
    """Component for advanced search features"""
    
    def __init__(self, vector_store):
        self.vector_store = vector_store
    
    def render(self):
        """Render the advanced search interface"""
        st.subheader("🔍 Advanced Search")
        
        # Create tabs for different search modes
        search_tabs = st.tabs([
            "📊 Semantic Search",
            "🔤 Regex Search", 
            "🏷️ Metadata Search",
            "🎯 Combined Search"
        ])
        
        # Tab 1: Semantic Similarity Search
        with search_tabs[0]:
            self._render_semantic_search()
        
        # Tab 2: Regex Search
        with search_tabs[1]:
            self._render_regex_search()
        
        # Tab 3: Metadata Search
        with search_tabs[2]:
            self._render_metadata_search()
        
        # Tab 4: Combined Search
        with search_tabs[3]:
            self._render_combined_search()
    
    def _render_semantic_search(self):
        """Render semantic similarity search interface"""
        st.markdown("Search using natural language and find semantically similar content.")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            query = st.text_input(
                "Search Query",
                placeholder="Enter your search query...",
                key="semantic_query"
            )
        
        with col2:
            num_results = st.number_input(
                "Results",
                min_value=1,
                max_value=50,
                value=10,
                key="semantic_num_results"
            )
        
        # Similarity threshold
        similarity_threshold = st.slider(
            "Minimum Similarity",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Only show results above this similarity threshold",
            key="semantic_threshold"
        )
        
        if st.button("Search", key="semantic_search_btn"):
            if query:
                with st.spinner("Searching..."):
                    results = self.vector_store.advanced_search(
                        query=query,
                        n_results=num_results,
                        min_similarity=similarity_threshold
                    )
                    self._display_results(results, show_similarity=True)
            else:
                st.warning("Please enter a search query.")
    
    def _render_regex_search(self):
        """Render regex search interface"""
        st.markdown("Search using regular expressions for precise pattern matching.")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            pattern = st.text_input(
                "Regex Pattern",
                placeholder="e.g., \\b[A-Z]{2,}\\b (finds uppercase words)",
                key="regex_pattern",
                help="Enter a valid Python regular expression"
            )
        
        with col2:
            case_sensitive = st.checkbox(
                "Case Sensitive",
                value=False,
                key="regex_case"
            )
        
        # Field selection
        search_field = st.selectbox(
            "Search In",
            options=["document", "source", "page"],
            key="regex_field",
            help="Select which field to search in"
        )
        
        # Optional metadata filters
        with st.expander("Additional Filters", expanded=False):
            sources = self.vector_store.get_unique_sources()
            if sources:
                selected_sources = st.multiselect(
                    "Filter by Source",
                    options=sources,
                    key="regex_sources"
                )
                metadata_filters = {"source": {"$in": selected_sources}} if selected_sources else None
            else:
                metadata_filters = None
        
        if st.button("Search with Regex", key="regex_search_btn"):
            if pattern:
                try:
                    # Test if pattern is valid
                    re.compile(pattern)
                    
                    with st.spinner("Searching..."):
                        results = self.vector_store.regex_search(
                            pattern=pattern,
                            field=search_field,
                            metadata_filters=metadata_filters,
                            case_sensitive=case_sensitive
                        )
                        self._display_regex_results(results)
                except re.error as e:
                    st.error(f"Invalid regex pattern: {str(e)}")
            else:
                st.warning("Please enter a regex pattern.")
    
    def _render_metadata_search(self):
        """Render metadata-only search interface"""
        st.markdown("Search documents by their metadata attributes.")
        
        # Get available sources
        sources = self.vector_store.get_unique_sources()
        
        if not sources:
            st.info("No documents available. Please upload some documents first.")
            return
        
        # Source filter
        selected_sources = st.multiselect(
            "Sources",
            options=sources,
            key="metadata_sources",
            help="Select one or more document sources"
        )
        
        # Page filter
        col1, col2 = st.columns(2)
        with col1:
            use_page_filter = st.checkbox("Filter by Page Number", key="use_page_filter")
        
        if use_page_filter:
            with col2:
                page_number = st.number_input(
                    "Page Number",
                    min_value=1,
                    value=1,
                    key="metadata_page"
                )
        
        # Custom metadata filter
        with st.expander("Custom Metadata Filters", expanded=False):
            st.markdown("Add custom key-value filters:")
            custom_key = st.text_input("Metadata Key", key="custom_meta_key")
            custom_value = st.text_input("Metadata Value", key="custom_meta_value")
        
        if st.button("Search by Metadata", key="metadata_search_btn"):
            # Build filters
            filters = {}
            
            if selected_sources:
                filters["source"] = {"$in": selected_sources}
            
            if use_page_filter:
                filters["page"] = page_number
            
            if custom_key and custom_value:
                filters[custom_key] = custom_value
            
            if filters:
                with st.spinner("Searching..."):
                    results = self.vector_store.metadata_search(filters)
                    self._display_metadata_results(results)
            else:
                st.warning("Please select at least one filter.")
    
    def _render_combined_search(self):
        """Render combined semantic + metadata search"""
        st.markdown("Combine semantic similarity with metadata filtering for precise results.")
        
        # Semantic query
        query = st.text_input(
            "Search Query",
            placeholder="Enter your search query...",
            key="combined_query"
        )
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            num_results = st.number_input(
                "Max Results",
                min_value=1,
                max_value=50,
                value=10,
                key="combined_num_results"
            )
        
        with col2:
            similarity_threshold = st.slider(
                "Min Similarity",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                key="combined_threshold"
            )
        
        # Metadata filters
        st.markdown("#### Metadata Filters")
        sources = self.vector_store.get_unique_sources()
        
        if sources:
            selected_sources = st.multiselect(
                "Filter by Source",
                options=sources,
                key="combined_sources"
            )
            
            metadata_filters = {}
            if selected_sources:
                metadata_filters["source"] = {"$in": selected_sources}
        else:
            metadata_filters = None
        
        if st.button("Search with Filters", key="combined_search_btn"):
            if query:
                with st.spinner("Searching..."):
                    results = self.vector_store.semantic_search_with_metadata(
                        query=query,
                        metadata_filters=metadata_filters,
                        n_results=num_results,
                        similarity_threshold=similarity_threshold
                    )
                    self._display_results(results, show_similarity=True)
            else:
                st.warning("Please enter a search query.")
    
    def _display_results(self, results: Dict, show_similarity: bool = False):
        """Display search results"""
        if not results or not results.get("ids") or not results["ids"][0]:
            st.info("No results found.")
            return
        
        st.success(f"Found {len(results['ids'][0])} results")
        
        for i, doc_id in enumerate(results["ids"][0]):
            with st.expander(f"📄 Result {i+1}", expanded=(i == 0)):
                # Document content
                if results.get("documents"):
                    st.markdown("**Content:**")
                    st.text(results["documents"][0][i][:500] + "..." if len(results["documents"][0][i]) > 500 else results["documents"][0][i])
                
                # Metadata
                if results.get("metadatas") and results["metadatas"][0][i]:
                    st.markdown("**Metadata:**")
                    metadata = results["metadatas"][0][i]
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Source:** {metadata.get('source', 'N/A')}")
                    with col2:
                        st.write(f"**Page:** {metadata.get('page', 'N/A')}")
                
                # Similarity score
                if show_similarity and results.get("distances"):
                    distance = results["distances"][0][i]
                    similarity = 1.0 - distance
                    st.metric("Similarity Score", f"{similarity:.2%}")
    
    def _display_regex_results(self, results: Dict):
        """Display regex search results with match highlights"""
        if not results or not results.get("ids") or not results["ids"][0]:
            st.info("No matches found.")
            return
        
        st.success(f"Found {len(results['ids'][0])} documents with matches")
        
        for i, doc_id in enumerate(results["ids"][0]):
            with st.expander(f"📄 Document {i+1}", expanded=(i == 0)):
                # Show matches
                if results.get("matches") and results["matches"][0][i]:
                    st.markdown("**Matches:**")
                    for j, match in enumerate(results["matches"][0][i], 1):
                        st.markdown(f"**Match {j}:** `{match['match']}`")
                        st.text(f"Context: ...{match['context']}...")
                        st.markdown("---")
                
                # Metadata
                if results.get("metadatas") and results["metadatas"][0][i]:
                    st.markdown("**Metadata:**")
                    metadata = results["metadatas"][0][i]
                    st.write(f"**Source:** {metadata.get('source', 'N/A')}")
                    st.write(f"**Page:** {metadata.get('page', 'N/A')}")
    
    def _display_metadata_results(self, results: Dict):
        """Display metadata search results"""
        if not results or not results.get("ids"):
            st.info("No results found.")
            return
        
        st.success(f"Found {len(results['ids'])} results")
        
        for i, doc_id in enumerate(results["ids"]):
            with st.expander(f"📄 Document {i+1}", expanded=(i == 0)):
                # Document content
                if results.get("documents") and i < len(results["documents"]):
                    st.markdown("**Content:**")
                    content = results["documents"][i]
                    st.text(content[:500] + "..." if len(content) > 500 else content)
                
                # Metadata
                if results.get("metadatas") and i < len(results["metadatas"]):
                    st.markdown("**Metadata:**")
                    for key, value in results["metadatas"][i].items():
                        st.write(f"**{key}:** {value}")
