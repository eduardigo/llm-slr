from dataclasses import dataclass

from llm_slr.config import RELEVANCE_SCALE_MAX, RELEVANCE_SCALE_MIN
from llm_slr.prompts.builder import SYSTEM_PROMPT, build_user_prompt


@dataclass(frozen=True)
class ScreeningResult:
    score: int
    justification: str
    model: str
    latency_s: float
    from_cache: bool
    label: int
    title: str


def _validate(content):
    score = content.get("score")
    if not isinstance(score, int) or not (
        RELEVANCE_SCALE_MIN <= score <= RELEVANCE_SCALE_MAX
    ):
        raise ValueError(f"score inválido na resposta do modelo: {content!r}")
    return score, str(content.get("justification", ""))


def screen_article(article, criteria, client, cache=None, examples=()):
    user = build_user_prompt(article, criteria, examples)

    if cache is not None:
        cached = cache.get(SYSTEM_PROMPT, user, client.model)
        if cached is not None:
            score, justification = _validate(cached)
            return ScreeningResult(
                score=score, justification=justification, model=client.model,
                latency_s=0.0, from_cache=True, label=article.label,
                title=article.title,
            )

    response = client.complete_json(SYSTEM_PROMPT, user)
    score, justification = _validate(response.content)

    if cache is not None:
        cache.put(SYSTEM_PROMPT, user, client.model, response.content)

    return ScreeningResult(
        score=score, justification=justification, model=client.model,
        latency_s=response.latency_s, from_cache=False, label=article.label,
        title=article.title,
    )


def screen_batch(articles, criteria, client, cache=None, examples=(),
                 selector=None, on_result=None):
    if examples and selector:
        raise ValueError("use 'examples' fixos OU 'selector', não ambos")

    results = []
    for article in articles:
        chosen = selector(article) if selector else examples
        result = screen_article(article, criteria, client, cache, chosen)
        results.append(result)
        if on_result:
            on_result(result)
    return results
