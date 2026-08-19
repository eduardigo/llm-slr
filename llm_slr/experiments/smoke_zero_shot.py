"""Teste rápido de triagem zero-shot com 10 artigos de um tema.

Pega uma amostra fixa do bloco de teste do fold 1, com 5 incluídos e 5
excluídos, e no fim repete 2 inferências sem cache para conferir se o
resultado se mantém.

Uso: python -m llm_slr.experiments.smoke_zero_shot [tema] [modelo]
"""
import sys

from llm_slr.data.loader import load_theme
from llm_slr.data.splits import temporal_folds
from llm_slr.llm.cache import ResponseCache
from llm_slr.llm.client import OllamaClient
from llm_slr.prompts.criteria_loader import load_criteria
from llm_slr.screening import screen_article, screen_batch


def sample_articles(theme, per_class=5):
    articles = load_theme(theme)
    _, test_block = temporal_folds(articles)[0]
    included = [a for a in test_block if a.label == 1][:per_class]
    excluded = [a for a in test_block if a.label == 0][:per_class]
    return included + excluded


def main(theme="games", model="llama3.2:3b"):
    criteria = load_criteria(theme)
    client = OllamaClient(model)
    cache = ResponseCache(theme, model, "smoke-zero-shot")
    sample = sample_articles(theme)

    print(f"tema={theme} modelo={model} artigos={len(sample)}\n")
    results = screen_batch(
        sample, criteria, client, cache,
        on_result=lambda r: print(
            f"  gold={'INC' if r.label else 'EXC'} score={r.score} "
            f"{'(cache)' if r.from_cache else f'{r.latency_s:5.1f}s'}  "
            f"{r.title[:60]!r}\n        -> {r.justification[:100]}"
        ),
    )

    scores_inc = [r.score for r in results if r.label == 1]
    scores_exc = [r.score for r in results if r.label == 0]
    print(f"\nscores dos incluídos: {scores_inc}")
    print(f"scores dos excluídos:  {scores_exc}")

    print("\nverificação de determinismo (2 artigos, sem cache):")
    for article in (sample[0], sample[-1]):
        rerun = screen_article(article, criteria, client, cache=None)
        original = next(r for r in results if r.title == article.title)
        status = "OK" if rerun.score == original.score else "DIVERGIU"
        print(f"  {status}: score {original.score} -> {rerun.score}  "
              f"{article.title[:55]!r}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
