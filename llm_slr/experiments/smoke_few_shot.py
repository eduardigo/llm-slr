"""Teste rápido comparando zero-shot e few-shot nos mesmos artigos.

Roda as três estratégias (zero-shot, few-shot aleatório balanceado e
few-shot por similaridade) sobre o bloco de teste do fold 1. Os exemplos
saem apenas do bloco de treino do fold.

Uso: python -m llm_slr.experiments.smoke_few_shot [tema] [modelo] [k]
"""
import sys

from llm_slr.data.loader import load_theme
from llm_slr.data.splits import temporal_folds
from llm_slr.fewshot import BalancedRandomSelector, SimilaritySelector
from llm_slr.llm.cache import ResponseCache
from llm_slr.llm.client import OllamaClient
from llm_slr.prompts.criteria_loader import load_criteria
from llm_slr.screening import screen_batch
from llm_slr.experiments.smoke_zero_shot import sample_articles


def main(theme="games", model="llama3.2:3b", k_per_class=2):
    criteria = load_criteria(theme)
    client = OllamaClient(model)
    sample = sample_articles(theme)

    train_pool, _ = temporal_folds(load_theme(theme))[0]
    strategies = {
        "zero-shot": None,
        "few-shot-random": BalancedRandomSelector(train_pool, int(k_per_class)),
        "few-shot-similar": SimilaritySelector(train_pool, int(k_per_class)),
    }

    print(f"tema={theme} modelo={model} k/classe={k_per_class} "
          f"treino={len(train_pool)} artigos={len(sample)}\n")
    all_scores = {}
    for name, selector in strategies.items():
        cache = ResponseCache(theme, model, f"smoke-{name}")
        results = screen_batch(sample, criteria, client, cache,
                               selector=selector)
        all_scores[name] = results
        inc = [r.score for r in results if r.label == 1]
        exc = [r.score for r in results if r.label == 0]
        lat = [r.latency_s for r in results if not r.from_cache]
        avg = f"{sum(lat)/len(lat):4.1f}s/artigo" if lat else "(cache)"
        print(f"{name:18s} INC={inc} EXC={exc}  {avg}")

    print("\ndetalhe por artigo (gold | zero | random | similar):")
    for i, r0 in enumerate(all_scores["zero-shot"]):
        rr = all_scores["few-shot-random"][i]
        rs = all_scores["few-shot-similar"][i]
        gold = "INC" if r0.label else "EXC"
        print(f"  {gold} | {r0.score} | {rr.score} | {rs.score} | "
              f"{r0.title[:58]!r}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
