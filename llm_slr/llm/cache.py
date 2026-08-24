import hashlib
import json
from pathlib import Path

from llm_slr.config import CACHE_DIR


def _key(system, user, model):
    return hashlib.sha256(f"{model}\x00{system}\x00{user}".encode()).hexdigest()


class ResponseCache:
    def __init__(self, theme, model, strategy, cache_dir=CACHE_DIR):
        safe_model = model.replace(":", "_").replace("/", "_")
        self._path = Path(cache_dir) / theme / safe_model / f"{strategy}.jsonl"
        self._entries = {}
        if self._path.exists():
            with open(self._path, encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    self._entries[row["key"]] = row

    def get(self, system, user, model):
        row = self._entries.get(_key(system, user, model))
        return row["response"] if row else None

    def put(self, system, user, model, response):
        key = _key(system, user, model)
        row = {"key": key, "model": model, "response": response}
        self._entries[key] = row
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def __len__(self):
        return len(self._entries)
