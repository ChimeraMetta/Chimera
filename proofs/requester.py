import logging
import requests
from typing import Dict, Any

class OpenAIRequests:
    """
    A minimal OpenAI API client using the requests library instead of the official SDK.
    This avoids the dependency issues with PyPy 3.8.
    """
    
    def __init__(self, api_key: str, model: str = "gpt-4"):
        """
        Initialize the OpenAI client.
        
        Args:
            api_key: Your OpenAI API key
            model: The model to use for completions
        """
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.openai.com/v1"
        
    def chat_completion(self, messages: list, temperature: float = 0.3, max_tokens: int = 2048) -> Dict[str, Any]:
        """
        Send a chat completion request to the OpenAI API.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            temperature: Controls randomness (0 to 1, lower is more deterministic)
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            The API response as a dictionary
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()  # Raise exception for HTTP errors
            return response.json()
        except Exception as e:
            logging.info("OpenAI API key for requester: " + self.api_key)
            logging.error(f"OpenAI API request failed: {e}")
            return {"error": str(e)}
    
    def get_completion_text(self, messages: list, temperature: float = 0.3, max_tokens: int = 2048) -> str:
        """
        Get just the completion text from a chat completion request.
        
        Args:
            messages: List of message dictionaries
            temperature: Controls randomness
            max_tokens: Maximum tokens to generate
            
        Returns:
            The text of the completion, or an error message
        """
        response = self.chat_completion(messages, temperature, max_tokens)
        
        try:
            return response["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            error_msg = response.get("error", {}).get("message", "Unknown error")
            logging.error(f"Failed to extract completion text: {error_msg}")
            return f"Error: {error_msg}"
