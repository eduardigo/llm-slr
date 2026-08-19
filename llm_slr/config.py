"""Configuração central do pipeline.

Caminhos e mapa de temas seguem o repositório legado
(slr-sentence-embedding-master), para que os experimentos com LLMs rodem
sobre os mesmos dados e folds da baseline.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LEGACY_ROOT = PROJECT_ROOT / "slr-sentence-embedding-master"
BIBS_DIR = LEGACY_ROOT / "bibs"
RESULTS_DIR = PROJECT_ROOT / "llm_slr" / "results"
CACHE_DIR = PROJECT_ROOT / "llm_slr" / "cache"
CRITERIA_DIR = Path(__file__).resolve().parent / "prompts" / "criteria"

# Mesmos arquivos de slrs_files, em config/__init__.py do legado.
THEMES = {
    "games": ["games/round1-todos.bib"],
    "slr": ["slr/round1-todos.bib"],
    "pair": ["pair/round1-todos.bib"],
    "illiterate": ["illiterate/round1-others.bib"],
    "mdwe": [
        "mdwe/round1-acm.bib",
        "mdwe/round1-ieee.bib",
        "mdwe/round1-sciencedirect.bib",
    ],
    "testing": [
        "testing/round1-google.bib",
        "testing/round1-ieee.bib",
        "testing/round1-outros.bib",
        "testing/round2-google.bib",
        "testing/round2-ieee.bib",
        "testing/round2-outros.bib",
        "testing/round3-google.bib",
    ],
    "ontologies": [
        "ontologies/round1-google.bib",
        "ontologies/round1-ieee.bib",
        "ontologies/round1-outros.bib",
        "ontologies/round2-google.bib",
        "ontologies/round2-ieee.bib",
        "ontologies/round3-google.bib",
    ],
    "xbi": [
        "xbi/round1-google.bib",
        "xbi/round1-ieee.bib",
        "xbi/round1-outros.bib",
        "xbi/round2-google.bib",
        "xbi/round2-ieee.bib",
        "xbi/round3-google.bib",
    ],
}

SEED = 42          # mesmo seed do legado
N_SPLITS = 3       # 3 folds temporais, como no main.py legado

RELEVANCE_SCALE_MIN = 1
RELEVANCE_SCALE_MAX = 7
TEMPERATURE = 0.0

# Modelos que cabem nos 4GB de VRAM da GTX 1050 Ti.
DEFAULT_MODELS = ["llama3.2:3b", "gemma2:2b", "qwen2.5:3b"]
OLLAMA_URL = "http://localhost:11434"


def theme_bib_files(theme):
    """Caminhos dos .bib de um tema, na mesma ordem usada pelo legado."""
    return [BIBS_DIR / rel for rel in THEMES[theme]]
