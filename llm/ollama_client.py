"""
llm/ollama_client.py

Minimal client for a LOCAL Ollama server (https://ollama.com).
No API keys. No external/paid services. Assumes Ollama is installed and
running on the same machine (default: http://localhost:11434).

Setup (manual, one-time, outside this repo):
    1. Install Ollama: https://ollama.com/download
    2. Pull a model, e.g.:
         ollama pull llama3.1
       or a smaller model if your laptop is limited, e.g.:
         ollama pull phi3
    3. Ollama runs its server automatically after install (or run `ollama serve`)

This client does NOT generate or alter vision-model probabilities. It is
only used for the natural-language explanation / doctor-summary layer,
fed with the vision model's output + questionnaire + knowledge base
context (see llm/prompt_engine.py).
"""

# llm/ollama_client.py

import requests
import json
from typing import Optional

class OllamaClient:
    """
    Client for communicating with local Ollama LLM server.
    """
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "phi3"):
        """
        Initialize Ollama client.
        
        Args:
            base_url: Ollama API endpoint (default: local)
            model: Model name to use (default: phi3)
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
    
    def is_available(self) -> bool:
        """
        Check if Ollama server is running and accessible.
        
        Returns:
            True if server responds, False otherwise
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return response.status_code == 200
        except Exception as e:
            print(f"⚠️ Ollama not available: {e}")
            return False
    
    def list_models(self) -> list:
        """
        List available models on Ollama server.
        
        Returns:
            List of model names
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            data = response.json()
            models = [model["name"] for model in data.get("models", [])]
            return models
        except Exception as e:
            print(f"Error listing models: {e}")
            return []
    
    def generate(self, prompt: str, max_tokens: int = 500, temperature: float = 0.7) -> str:
        """
        Generate text using Ollama.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate (default: 500)
            temperature: Sampling temperature (0-1, higher = more creative)
        
        Returns:
            Generated text response
        """
        if not self.is_available():
            raise ConnectionError(
                f"Ollama server not available at {self.base_url}\n"
                "Make sure Ollama is running: ollama serve"
            )
        
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "temperature": temperature,
                "num_predict": max_tokens,
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=60
            )
            
            if response.status_code != 200:
                raise Exception(f"Ollama error: {response.status_code} {response.text}")
            
            data = response.json()
            return data.get("response", "").strip()
        
        except requests.Timeout:
            raise TimeoutError("Ollama request timed out (>60s). Model may be processing.")
        except Exception as e:
            raise Exception(f"Ollama generation failed: {e}")


# Test Ollama connection
if __name__ == "__main__":
    client = OllamaClient()
    
    # Check availability
    if client.is_available():
        print("✅ Ollama is running!")
        
        # List models
        models = client.list_models()
        print(f"📦 Available models: {models}")
        
        # Test generation
        response = client.generate("What is eczema? (1 sentence)", max_tokens=100)
        print(f"\n🤖 Ollama response:\n{response}")
    else:
        print("❌ Ollama not running. Start it with: ollama serve")
