import pytest

from llm_slr.data.loader import Article
from llm_slr.llm.cache import ResponseCache
from llm_slr.prompts.builder import SYSTEM_PROMPT, build_user_prompt
from llm_slr.prompts.criteria_loader import load_criteria
from llm_slr.screening import screen_article, screen_batch

ARTICLE = Article(
    title="Design patterns for games",
    abstract="We present design patterns applied to game development.",
    year=2002, label=1, source_file="x.bib",
)
INCLUDED_EXAMPLE = Article(
    title="A game engine study", abstract="Engine internals.",
    year=2001, label=1, source_file="x.bib",
)
EXCLUDED_EXAMPLE = Article(
    title="Game theory of auctions", abstract="Economics.",
    year=2001, label=0, source_file="x.bib",
)


class FakeClient:
    model = "fake:1b"

    def __init__(self, content):
        self._content = content
        self.calls = 0

    def complete_json(self, system, user):
        from llm_slr.llm.client import LLMResponse
        self.calls += 1
        return LLMResponse(content=self._content, latency_s=0.5,
                           model=self.model, raw_text="{}")


@pytest.fixture
def criteria():
    return load_criteria("games")


def test_prompt_has_criteria_scale_and_no_exclusion(criteria):
    user = build_user_prompt(ARTICLE, criteria)

    assert criteria.inclusion_criteria[0] in user
    assert criteria.research_question in user
    assert "strongly agree" in user and '"score"' in user
    assert ARTICLE.title in user and ARTICLE.abstract in user
    assert "exclusion criteria" not in (SYSTEM_PROMPT + user).lower()


def test_prompt_few_shot_includes_examples_and_decisions(criteria):
    user = build_user_prompt(
        ARTICLE, criteria, examples=(INCLUDED_EXAMPLE, EXCLUDED_EXAMPLE)
    )
    assert INCLUDED_EXAMPLE.title in user
    assert "Decision by the original reviewers: included" in user
    assert "Decision by the original reviewers: not included" in user


def test_screen_article_validates_score_and_uses_cache(tmp_path, criteria):
    cache = ResponseCache("games", "fake:1b", "zero-shot", cache_dir=tmp_path)
    client = FakeClient({"score": 6, "justification": "relevant"})

    first = screen_article(ARTICLE, criteria, client, cache)
    assert (first.score, first.from_cache, first.label) == (6, False, 1)
    assert first.latency_s > 0

    second = screen_article(ARTICLE, criteria, client, cache)
    assert (second.score, second.from_cache) == (6, True)
    assert client.calls == 1

    reopened = ResponseCache("games", "fake:1b", "zero-shot", cache_dir=tmp_path)
    assert len(reopened) == 1


@pytest.mark.parametrize("bad", [
    {"score": 0}, {"score": 8}, {"score": "6"}, {"justification": "no score"},
])
def test_screen_article_rejects_invalid_scores(bad, criteria):
    with pytest.raises(ValueError):
        screen_article(ARTICLE, criteria, FakeClient(bad))


def test_screen_batch_reports_progress(criteria):
    client = FakeClient({"score": 3, "justification": "x"})
    seen = []
    results = screen_batch([ARTICLE, INCLUDED_EXAMPLE], criteria, client,
                           on_result=seen.append)
    assert [r.score for r in results] == [3, 3]
    assert len(seen) == 2
