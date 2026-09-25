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

import json
from typing import Optional

import requests

DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.1"  # change to whatever you have pulled, e.g. "phi3"
DEFAULT_TIMEOUT_SECONDS = 120


class OllamaClient:
    def __init__(self, base_url: str = DEFAULT_BASE_URL, model: str = DEFAULT_MODEL):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def is_available(self) -> bool:
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return resp.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def list_models(self):
        resp = requests.get(f"{self.base_url}/api/tags", timeout=10)
        resp.raise_for_status()
        return [m["name"] for m in resp.json().get("models", [])]

    def generate(self, prompt: str, model: Optional[str] = None, temperature: float = 0.2) -> str:
        """
        Single-shot generation (non-streaming) against /api/generate.
        Raises requests.exceptions.RequestException if Ollama is unreachable.
        """
        payload = {
            "model": model or self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        resp = requests.post(
            f"{self.base_url}/api/generate",
            data=json.dumps(payload),
            headers={"Content-Type": "application/json"},
            timeout=DEFAULT_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        return resp.json().get("response", "")

    def chat(self, messages: list, model: Optional[str] = None, temperature: float = 0.2) -> str:
        """
        messages: list of {"role": "user"|"assistant"|"system", "content": str}
        """
        payload = {
            "model": model or self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        resp = requests.post(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload),
            headers={"Content-Type": "application/json"},
            timeout=DEFAULT_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        return resp.json().get("message", {}).get("content", "")


if __name__ == "__main__":
    client = OllamaClient()
    if not client.is_available():
        print(f"[WARN] Could not reach Ollama at {client.base_url}")
        print("       Make sure Ollama is installed and running (`ollama serve`).")
    else:
        print(f"[OK] Ollama reachable at {client.base_url}")
        print(f"Models available: {client.list_models()}")
