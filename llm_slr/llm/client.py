import json
import time
from dataclasses import dataclass

import requests

from llm_slr.config import OLLAMA_URL, SEED, TEMPERATURE


@dataclass(frozen=True)
class LLMResponse:
    content: dict
    latency_s: float
    model: str
    raw_text: str


class OllamaClient:

    def __init__(self, model, base_url=OLLAMA_URL, temperature=TEMPERATURE,
                 seed=SEED, num_predict=300, num_ctx=8192, timeout_s=300):
        self.model = model
        self._base_url = base_url
        self._options = {
            "temperature": temperature,
            "seed": seed,
            "num_predict": num_predict,
            "num_ctx": num_ctx,
        }
        self._timeout_s = timeout_s

    def complete_json(self, system, user):
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "format": "json",
            "stream": False,
            "options": self._options,
        }
        start = time.perf_counter()
        resp = requests.post(
            f"{self._base_url}/api/chat", json=payload, timeout=self._timeout_s
        )
        resp.raise_for_status()
        latency = time.perf_counter() - start

        raw = resp.json()["message"]["content"]
        return LLMResponse(
            content=json.loads(raw),
            latency_s=latency,
            model=self.model,
            raw_text=raw,
        )
