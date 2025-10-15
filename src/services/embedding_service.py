class EmbeddingService:
    def __init__(self, ollama_service):
        self.ollama_service = ollama_service

    def generate_embeddings(self, document):
        # Interact with the Ollama setup to generate embeddings for the document
        embeddings = self.ollama_service.get_embeddings(document)
        return embeddings