from typing import List, Optional
import requests
from dotenv import load_dotenv
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class OllamaService:
    def __init__(self, base_url: Optional[str] = None):
        """
        Initialize Ollama service
        
        Args:
            base_url: Base URL for Ollama API, defaults to environment variable or localhost
        """
        self.logger = logging.getLogger(__name__)
        load_dotenv()
        self.base_url = base_url or os.getenv("OLLAMA_API_URL", "http://localhost:11434")
        self.model = os.getenv("OLLAMA_MODEL", "phi3:latest")
        # self.model = os.getenv("OLLAMA_MODEL", "llama3:8B")
        
        self.logger.info(f"Initialized OllamaService with base_url: {self.base_url}, model: {self.model}")
        
    def check_connection(self) -> tuple[bool, str]:
        """
        Check if Ollama service is accessible and the model is available
        
        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            # First check if the server is running
            self.logger.info("Checking Ollama server connection...")
            response = requests.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            
            # Check if our model is available
            self.logger.info(f"Checking if model '{self.model}' is available...")
            available_models = [model["name"] for model in response.json()["models"]]
            
            if self.model in available_models:
                msg = f"Successfully connected to Ollama server and found model '{self.model}'"
                self.logger.info(msg)
                return True, msg
            else:
                if not available_models:
                    msg = (f"Ollama server is running but no models are available. "
                          f"Run 'ollama pull {self.model}' to download the model.")
                else:
                    msg = (f"Model '{self.model}' is not available. Available models: {', '.join(available_models)}. "
                          f"Run 'ollama pull {self.model}' to download the model.")
                self.logger.warning(msg)
                return False, msg
                
        except requests.exceptions.ConnectionError:
            msg = f"Could not connect to Ollama server at {self.base_url}. Is Ollama running?"
            self.logger.error(msg)
            return False, msg
        except requests.exceptions.RequestException as e:
            msg = f"Error checking Ollama connection: {str(e)}"
            self.logger.error(msg)
            return False, msg

    def generate_response(self, prompt: str, context: List[str], temperature: float = 0.7) -> str:
        """
        Generate response using Ollama
        
        Args:
            prompt: The user's question
            context: List of relevant document chunks
            temperature: Controls randomness in the response
            
        Returns:
            Generated response from the model
        """
        self.logger.info(f"Generating response for prompt: {prompt[:100]}...")
        self.logger.debug(f"Context length: {len(context)} chunks")
        
        # Combine context and prompt into a well-structured prompt
        full_prompt = self._create_prompt(prompt, context)
        self.logger.debug(f"Full prompt length: {len(full_prompt)} characters")
        
        try:
            self.logger.debug(f"Making API request to {self.base_url}/api/generate")
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": full_prompt,
                    "temperature": temperature,
                    "stream": False
                }
            )
            response.raise_for_status()
            response_text = response.json()["response"]
            self.logger.info(f"Successfully generated response of length {len(response_text)}")
            return response_text
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Error generating response: {str(e)}"
            self.logger.error(error_msg)
            raise Exception(error_msg)

    def _create_prompt(self, question: str, context: List[str]) -> str:
        """
        Create a well-structured prompt combining context and question
        
        Args:
            question: The user's question
            context: List of relevant document chunks
            
        Returns:
            Formatted prompt string
        """
        self.logger.debug(f"Creating prompt for question: {question}")
        context_text = "\n".join(context)
        self.logger.debug(f"Combined context length: {len(context_text)} characters")
        
        prompt = f"""Please provide a detailed and accurate answer based on the following context. If you cannot find the answer in the context, say so.

Context:
{context_text}

Question: {question}

Answer:"""
        self.logger.debug(f"Final prompt length: {len(prompt)} characters")
        return prompt