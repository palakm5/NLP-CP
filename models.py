"""Model backends with unified interface, caching, exponential retries, and latency tracking.

Supports:
- OpenAI-compatible API (OPENAI_API_KEY, OPENAI_BASE_URL)
- Anthropic API (ANTHROPIC_API_KEY)
- Local Ollama (OLLAMA_BASE_URL)
- Mock / offline backend for testing and demos
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

CACHE_DIR = Path(__file__).parent / ".cache" / "llm"


@dataclass
class ModelResponse:
    text: str
    latency: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached: bool = False
    model: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DiskCache:
    """Persistent on-disk cache hashed on (model, messages, parameters)."""

    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _hash_key(self, model_name: str, messages: List[Dict[str, str]], kwargs: Dict[str, Any]) -> str:
        payload = {
            "model": model_name,
            "messages": messages,
            "kwargs": {k: v for k, v in kwargs.items() if k not in ("self",)},
        }
        raw = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, model_name: str, messages: List[Dict[str, str]], kwargs: Dict[str, Any]) -> Optional[ModelResponse]:
        key = self._hash_key(model_name, messages, kwargs)
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return ModelResponse(
                        text=data["text"],
                        latency=0.0,
                        prompt_tokens=data.get("prompt_tokens", 0),
                        completion_tokens=data.get("completion_tokens", 0),
                        cached=True,
                        model=model_name,
                    )
            except Exception:
                return None
        return None

    def set(self, model_name: str, messages: List[Dict[str, str]], kwargs: Dict[str, Any], response: ModelResponse):
        key = self._hash_key(model_name, messages, kwargs)
        path = self.cache_dir / f"{key}.json"
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump({
                    "text": response.text,
                    "prompt_tokens": response.prompt_tokens,
                    "completion_tokens": response.completion_tokens,
                    "model": model_name,
                }, f, ensure_ascii=False)
        except Exception:
            pass


class ModelBackend(ABC):
    """Abstract base class for all LLM model backends."""

    def __init__(self, model_name: str, enable_cache: bool = True):
        self.model_name = model_name
        self.enable_cache = enable_cache
        self.cache = DiskCache() if enable_cache else None

    @abstractmethod
    def _call_api(self, messages: List[Dict[str, str]], **kwargs) -> ModelResponse:
        pass

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 2048,
        retries: int = 3,
        **kwargs,
    ) -> ModelResponse:
        """Unified inference call with caching and exponential backoff retry."""
        call_kwargs = {"temperature": temperature, "max_tokens": max_tokens, **kwargs}

        if self.enable_cache and self.cache:
            cached_resp = self.cache.get(self.model_name, messages, call_kwargs)
            if cached_resp is not None:
                return cached_resp

        last_err = None
        for attempt in range(retries):
            try:
                start_time = time.perf_counter()
                resp = self._call_api(messages, **call_kwargs)
                latency = time.perf_counter() - start_time
                resp.latency = round(latency, 3)
                resp.model = self.model_name

                if self.enable_cache and self.cache:
                    self.cache.set(self.model_name, messages, call_kwargs, resp)
                return resp
            except Exception as e:
                last_err = e
                if attempt < retries - 1:
                    sleep_time = (2 ** attempt) * 1.0
                    time.sleep(sleep_time)

        raise RuntimeError(f"Model generation failed after {retries} retries: {str(last_err)}") from last_err


class OpenAICompatibleModel(ModelBackend):
    """Backend for OpenAI and OpenAI-compatible API providers (vLLM, Groq, DeepSeek, etc.)."""

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        enable_cache: bool = True,
    ):
        super().__init__(model_name, enable_cache)
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")

    def _call_api(self, messages: List[Dict[str, str]], **kwargs) -> ModelResponse:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.0),
            "max_tokens": kwargs.get("max_tokens", 2048),
        }
        res = requests.post(url, headers=headers, json=payload, timeout=60)
        res.raise_for_status()
        data = res.json()
        choice = data["choices"][0]
        usage = data.get("usage", {})
        return ModelResponse(
            text=choice["message"]["content"] or "",
            latency=0.0,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            model=self.model_name,
        )


class AnthropicModel(ModelBackend):
    """Backend for Anthropic Claude models."""

    def __init__(
        self,
        model_name: str = "claude-3-5-haiku-20241022",
        api_key: Optional[str] = None,
        enable_cache: bool = True,
    ):
        super().__init__(model_name, enable_cache)
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")

    def _call_api(self, messages: List[Dict[str, str]], **kwargs) -> ModelResponse:
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }
        # Separate system messages if any
        system_prompt = ""
        anthropic_messages = []
        for m in messages:
            if m["role"] == "system":
                system_prompt += m["content"] + "\n"
            else:
                anthropic_messages.append({"role": m["role"], "content": m["content"]})

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": anthropic_messages,
            "max_tokens": kwargs.get("max_tokens", 2048),
            "temperature": kwargs.get("temperature", 0.0),
        }
        if system_prompt.strip():
            payload["system"] = system_prompt.strip()

        res = requests.post(url, headers=headers, json=payload, timeout=60)
        res.raise_for_status()
        data = res.json()
        text = "".join(b["text"] for b in data.get("content", []) if b.get("type") == "text")
        usage = data.get("usage", {})
        return ModelResponse(
            text=text,
            latency=0.0,
            prompt_tokens=usage.get("input_tokens", 0),
            completion_tokens=usage.get("output_tokens", 0),
            model=self.model_name,
        )


class OllamaModel(ModelBackend):
    """Backend for local Ollama instances."""

    def __init__(
        self,
        model_name: str = "llama3.2",
        base_url: Optional[str] = None,
        enable_cache: bool = True,
    ):
        super().__init__(model_name, enable_cache)
        self.base_url = (base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")

    def _call_api(self, messages: List[Dict[str, str]], **kwargs) -> ModelResponse:
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", 0.0),
                "num_predict": kwargs.get("max_tokens", 2048),
            },
        }
        res = requests.post(url, json=payload, timeout=120)
        res.raise_for_status()
        data = res.json()
        return ModelResponse(
            text=data.get("message", {}).get("content", ""),
            latency=0.0,
            prompt_tokens=data.get("prompt_eval_count", 0),
            completion_tokens=data.get("eval_count", 0),
            model=self.model_name,
        )


class MockModel(ModelBackend):
    """Mock backend returning canned reference solutions or customizable outputs for tests."""

    def __init__(
        self,
        model_name: str = "mock-model",
        custom_response: Optional[str] = None,
        enable_cache: bool = False,
    ):
        super().__init__(model_name, enable_cache=enable_cache)
        self.custom_response = custom_response

    def _call_api(self, messages: List[Dict[str, str]], **kwargs) -> ModelResponse:
        if self.custom_response is not None:
            return ModelResponse(
                text=self.custom_response,
                latency=0.005,
                prompt_tokens=len(str(messages)) // 4,
                completion_tokens=len(self.custom_response) // 4,
                model=self.model_name,
            )

        # Fallback: check if the last message contains task description
        last_user = messages[-1]["content"] if messages else ""
        from tasks import TASKS
        matched_code = ""
        for t in TASKS:
            if t.entry_point in last_user:
                matched_code = f"```python\n{t.reference_code}\n```"
                break
        if not matched_code:
            matched_code = "```python\ndef solve(*args):\n    return None\n```"

        return ModelResponse(
            text=matched_code,
            latency=0.005,
            prompt_tokens=len(str(messages)) // 4,
            completion_tokens=len(matched_code) // 4,
            model=self.model_name,
        )


def get_model(spec: str, **kwargs) -> ModelBackend:
    """Factory creating ModelBackend by specification prefix.

    Examples:
    - 'mock' or 'mock/test' -> MockModel
    - 'openai/gpt-4o-mini' -> OpenAICompatibleModel
    - 'anthropic/claude-3-5-haiku-20241022' -> AnthropicModel
    - 'ollama/llama3.2' -> OllamaModel
    """
    if spec.startswith("mock"):
        name = spec.split("/", 1)[1] if "/" in spec else "mock-model"
        return MockModel(model_name=name, **kwargs)
    elif spec.startswith("anthropic"):
        name = spec.split("/", 1)[1] if "/" in spec else "claude-3-5-haiku-20241022"
        return AnthropicModel(model_name=name, **kwargs)
    elif spec.startswith("ollama"):
        name = spec.split("/", 1)[1] if "/" in spec else "llama3.2"
        return OllamaModel(model_name=name, **kwargs)
    else:
        # Default to OpenAI compatible
        name = spec.split("/", 1)[1] if "/" in spec else spec
        return OpenAICompatibleModel(model_name=name, **kwargs)
