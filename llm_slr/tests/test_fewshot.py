import pytest

from llm_slr.data.loader import Article
from llm_slr.fewshot import BalancedRandomSelector, SimilaritySelector


def _art(title, abstract, label, year=2005):
    return Article(title=title, abstract=abstract, year=year, label=label,
                   source_file="x.bib")


POOL = [
    _art("Game engine architecture", "Rendering pipelines and engine design.", 1),
    _art("Requirements for games", "Eliciting requirements in game projects.", 1),
    _art("Playability heuristics", "Evaluating playability of video games.", 1),
    _art("Game theory of auctions", "Nash equilibrium in auction markets.", 0),
    _art("Network latency in MMOs", "Measuring latency on game servers.", 0),
    _art("AI for chess programs", "Search algorithms for chess playing.", 0),
]
CANDIDATE = _art("A new game engine design", "We design a rendering engine.", 1)


def test_balanced_random_is_deterministic_and_balanced():
    a = BalancedRandomSelector(POOL, k_per_class=2, seed=42)(CANDIDATE)
    b = BalancedRandomSelector(POOL, k_per_class=2, seed=42)(CANDIDATE)

    assert a == b
    assert len(a) == 4
    assert sum(e.label for e in a) == 2


def test_balanced_random_caps_k_at_pool_size():
    examples = BalancedRandomSelector(POOL, k_per_class=10)(CANDIDATE)
    assert len(examples) == len(POOL)


def test_similarity_selects_most_similar_per_class():
    selector = SimilaritySelector(POOL, k_per_class=1)
    examples = selector(CANDIDATE)

    assert len(examples) == 2
    included, excluded = examples
    assert included.title == "Game engine architecture"
    assert excluded.label == 0


def test_similarity_is_deterministic():
    s1 = SimilaritySelector(POOL, k_per_class=2)(CANDIDATE)
    s2 = SimilaritySelector(POOL, k_per_class=2)(CANDIDATE)
    assert s1 == s2


def test_selectors_never_leak_candidate_into_examples():
    for selector in (BalancedRandomSelector(POOL), SimilaritySelector(POOL)):
        assert CANDIDATE not in selector(CANDIDATE)


def test_screen_batch_rejects_examples_and_selector_together():
    from llm_slr.prompts.criteria_loader import load_criteria
    from llm_slr.screening import screen_batch

    class Dummy:
        model = "d"

    with pytest.raises(ValueError):
        screen_batch([CANDIDATE], load_criteria("games"), Dummy(),
                     examples=(POOL[0],), selector=SimilaritySelector(POOL))
