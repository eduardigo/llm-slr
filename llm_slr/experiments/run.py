"""Experimento principal: temas x modelos x estratégias x folds.

Grava um resultado por artigo em results/raw.csv, em append. A rodada é
retomável: combinações (tema, modelo, estratégia, fold) já completas são
puladas e o cache de respostas cobre o resto. Score inválido vira uma linha
com a coluna error preenchida, sem interromper a rodada.

Uso:
  python -m llm_slr.experiments.run                       # tudo
  python -m llm_slr.experiments.run --themes slr,games --models llama3.2:3b
"""
import argparse
import csv
from pathlib import Path

from llm_slr.config import DEFAULT_MODELS, RESULTS_DIR, THEMES
from llm_slr.data.loader import load_theme
from llm_slr.data.splits import temporal_folds
from llm_slr.fewshot import BalancedRandomSelector, SimilaritySelector
from llm_slr.llm.cache import ResponseCache
from llm_slr.llm.client import OllamaClient
from llm_slr.prompts.criteria_loader import load_criteria
from llm_slr.screening import screen_article

RAW_CSV = Path(RESULTS_DIR) / "raw.csv"
FIELDS = ["theme", "model", "strategy", "fold", "title", "year", "label",
          "score", "latency_s", "from_cache", "error"]

STRATEGIES = ("zero-shot", "few-shot-random", "few-shot-similar")

# temas menores primeiro, para ter resultado parcial antes
THEME_ORDER = ["slr", "mdwe", "illiterate", "pair", "games",
               "testing", "ontologies", "xbi"]


def make_selector(strategy, train_pool, k_per_class):
    if strategy == "zero-shot":
        return None
    if strategy == "few-shot-random":
        return BalancedRandomSelector(train_pool, k_per_class)
    if strategy == "few-shot-similar":
        return SimilaritySelector(train_pool, k_per_class)
    raise ValueError(f"estratégia desconhecida: {strategy}")


def strategy_label(strategy, k_per_class):
    """Com k=2 o rótulo não muda; outros k ganham sufixo (-k1, -k4)."""
    if strategy == "zero-shot" or k_per_class == 2:
        return strategy
    return f"{strategy}-k{k_per_class}"


def load_done_counts():
    """Conta as linhas já gravadas por (theme, model, strategy, fold)."""
    done = {}
    if RAW_CSV.exists():
        with open(RAW_CSV, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                key = (row["theme"], row["model"], row["strategy"],
                       int(row["fold"]))
                done[key] = done.get(key, 0) + 1
    return done


def append_rows(rows):
    RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
    new_file = not RAW_CSV.exists()
    with open(RAW_CSV, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerows(rows)


def run_combination(theme, model, strategy, fold_id, train, test,
                    criteria, client, k_per_class):
    selector = make_selector(strategy, train, k_per_class)
    label = strategy_label(strategy, k_per_class)
    cache = ResponseCache(theme, model, label)
    rows = []
    for article in test:
        examples = selector(article) if selector else ()
        base = {
            "theme": theme, "model": model, "strategy": label,
            "fold": fold_id, "title": article.title[:120],
            "year": article.year, "label": article.label,
        }
        try:
            r = screen_article(article, criteria, client, cache, examples)
            rows.append({**base, "score": r.score,
                         "latency_s": round(r.latency_s, 3),
                         "from_cache": r.from_cache, "error": ""})
        except Exception as exc:  # registra e segue, para a rodada não cair
            rows.append({**base, "score": "", "latency_s": "",
                         "from_cache": False, "error": str(exc)[:200]})
    append_rows(rows)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--themes", default=",".join(THEME_ORDER))
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--strategies", default=",".join(STRATEGIES))
    parser.add_argument("--k", type=int, default=2)
    args = parser.parse_args()

    themes = [t for t in args.themes.split(",") if t in THEMES]
    done = load_done_counts()

    for theme in themes:
        articles = load_theme(theme)
        criteria = load_criteria(theme)
        folds = temporal_folds(articles)
        for model in args.models.split(","):
            client = OllamaClient(model)
            for strategy in args.strategies.split(","):
                for fold_id, (train, test) in enumerate(folds, start=1):
                    key = (theme, model, strategy_label(strategy, args.k),
                           fold_id)
                    if done.get(key, 0) >= len(test):
                        print(f"skip {key} (completo)", flush=True)
                        continue
                    rows = run_combination(theme, model, strategy, fold_id,
                                           train, test, criteria, client,
                                           args.k)
                    errs = sum(1 for r in rows if r["error"])
                    fresh = [r["latency_s"] for r in rows
                             if r["latency_s"] != "" and not r["from_cache"]]
                    avg = (f"{sum(fresh)/len(fresh):.1f}s/art"
                           if fresh else "cache")
                    print(f"ok   {key} n={len(rows)} err={errs} {avg}",
                          flush=True)


if __name__ == "__main__":
    main()
