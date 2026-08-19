"""Seleção de exemplos few-shot a partir do histórico da revisão original.

A ideia é usar os artigos já avaliados na RSL original (o bloco de treino do
fold temporal) para calibrar o LLM por In-Context Learning, sem
retreinamento, com exemplos aceitos e rejeitados.

Duas estratégias:
- BalancedRandomSelector: k exemplos fixos por classe, sorteados uma vez com
  seed, iguais para todos os candidatos do fold.
- SimilaritySelector: k exemplos por classe mais próximos do candidato
  (TF-IDF + cosseno), variando artigo a artigo.

As duas são determinísticas e leem só o bloco de treino, de modo que nada do
bloco de teste chega ao prompt.
"""
import random

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from llm_slr.config import SEED


def _split_by_label(pool):
    included = [a for a in pool if a.label == 1]
    excluded = [a for a in pool if a.label == 0]
    return included, excluded


class BalancedRandomSelector:
    """k exemplos por classe, os mesmos para todos os candidatos."""

    name = "few-shot-random"

    def __init__(self, train_pool, k_per_class=2, seed=SEED):
        included, excluded = _split_by_label(train_pool)
        rng = random.Random(seed)
        self._examples = tuple(
            rng.sample(included, min(k_per_class, len(included)))
            + rng.sample(excluded, min(k_per_class, len(excluded)))
        )

    def __call__(self, article):
        return self._examples


class SimilaritySelector:
    """k exemplos por classe mais próximos do candidato (TF-IDF/cosseno)."""

    name = "few-shot-similar"

    def __init__(self, train_pool, k_per_class=2):
        self._included, self._excluded = _split_by_label(train_pool)
        self._k = k_per_class
        self._vectorizer = TfidfVectorizer(stop_words="english")
        corpus = [a.content for a in self._included + self._excluded]
        self._matrix = self._vectorizer.fit_transform(corpus)

    def __call__(self, article):
        query = self._vectorizer.transform([article.content])
        scores = cosine_similarity(query, self._matrix)[0]

        n_inc = len(self._included)
        top_inc = sorted(range(n_inc), key=lambda i: -scores[i])[: self._k]
        top_exc = sorted(range(n_inc, len(scores)), key=lambda i: -scores[i])[: self._k]

        pool = self._included + self._excluded
        return tuple(pool[i] for i in top_inc + top_exc)
