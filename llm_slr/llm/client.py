"""Clientes de inferência LLM.

Todo cliente fixa temperature=0 e seed, para que as rodadas sejam
reprodutíveis. A interface é curta de propósito (entra system + user, sai um
dict JSON), o que permite trocar o Ollama por outro backend sem mexer no
resto do pipeline.
"""
import json
import time
from dataclasses import dataclass

import requests

from llm_slr.config import OLLAMA_URL, SEED, TEMPERATURE


@dataclass(frozen=True)
class LLMResponse:
    content: dict          # JSON retornado pelo modelo, já parseado
    latency_s: float       # tempo de inferência
    model: str
    raw_text: str


class OllamaClient:
    """Adaptador do endpoint /api/chat do Ollama, com saída em JSON."""

    def __init__(self, model, base_url=OLLAMA_URL, temperature=TEMPERATURE,
                 seed=SEED, num_predict=300, num_ctx=8192, timeout_s=300):
        self.model = model
        self._base_url = base_url
        self._options = {
            "temperature": temperature,
            "seed": seed,
            "num_predict": num_predict,
            # o default do Ollama (2048) trunca prompts few-shot em silêncio
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
            # sem format="json" os modelos menores erram bastante o parsing
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
