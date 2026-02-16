"""
Local LLM client using llama-cpp-python.
Runs LLaMA 3 8B Instruct (Q4_K_M) with full GPU offload.
No paid APIs — fully local inference.
"""

import time
import os
from typing import Dict, Any
from config import settings


class LLMClient:
    """
    Local LLM inference via llama-cpp-python.

    Loads a GGUF-quantized model with GPU layer offloading.
    Provides token usage tracking and latency measurement.
    """

    def __init__(self, model_path: str = None):
        self.model_path = model_path or settings.LLAMA_MODEL_PATH
        self._llm = None

    @property
    def llm(self):
        """Lazy-load the LLM."""
        if self._llm is None:
            self._llm = self._load_model()
        return self._llm

    def _load_model(self):
        """Load the GGUF model with GPU offload."""
        from llama_cpp import Llama

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"LLM model not found at {self.model_path}. "
                f"Download with: huggingface-cli download TheBloke/Meta-Llama-3-8B-Instruct-GGUF "
                f"Meta-Llama-3-8B-Instruct-Q4_K_M.gguf --local-dir ./models/"
            )

        return Llama(
            model_path=self.model_path,
            n_ctx=settings.LLM_CONTEXT_LENGTH,
            n_gpu_layers=settings.LLM_GPU_LAYERS,  # -1 = all layers
            verbose=False,
        )

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = None,
        temperature: float = None,
    ) -> Dict[str, Any]:
        """
        Generate a response using the local LLM.

        Args:
            system_prompt: System instructions
            user_prompt: User query with context
            max_tokens: Max tokens to generate
            temperature: Sampling temperature

        Returns:
            Dict with 'content', 'usage', 'latency_ms'
        """
        max_tokens = max_tokens or settings.LLM_MAX_TOKENS
        temperature = (
            temperature if temperature is not None else settings.LLM_TEMPERATURE
        )

        start = time.time()

        response = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
            stop=["<|eot_id|>", "<|end_of_text|>"],
        )

        elapsed_ms = (time.time() - start) * 1000

        # Extract content
        content = ""
        if response.get("choices"):
            content = response["choices"][0].get("message", {}).get("content", "")

        # Extract token usage
        usage = response.get("usage", {})

        return {
            "content": content.strip(),
            "usage": {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("prompt_tokens", 0)
                + usage.get("completion_tokens", 0),
            },
            "latency_ms": elapsed_ms,
            "model": os.path.basename(self.model_path),
        }

    def generate_simple(self, prompt: str, max_tokens: int = 256) -> str:
        """Simple text completion without chat format."""
        result = self.generate(
            system_prompt="You are a helpful AI assistant.",
            user_prompt=prompt,
            max_tokens=max_tokens,
        )
        return result["content"]
