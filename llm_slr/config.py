from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LEGACY_ROOT = PROJECT_ROOT / "slr-sentence-embedding-master"
BIBS_DIR = LEGACY_ROOT / "bibs"
RESULTS_DIR = PROJECT_ROOT / "llm_slr" / "results"
CACHE_DIR = PROJECT_ROOT / "llm_slr" / "cache"
FIGURES_DIR = PROJECT_ROOT / "llm_slr" / "results" / "figuras"
CRITERIA_DIR = Path(__file__).resolve().parent / "prompts" / "criteria"

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

SEED = 42
N_SPLITS = 3

RELEVANCE_SCALE_MIN = 1
RELEVANCE_SCALE_MAX = 7
TEMPERATURE = 0.0

DEFAULT_MODELS = ["llama3.2:3b", "gemma2:2b", "qwen2.5:3b"]
OLLAMA_URL = "http://localhost:11434"


def theme_bib_files(theme):
    return [BIBS_DIR / rel for rel in THEMES[theme]]
