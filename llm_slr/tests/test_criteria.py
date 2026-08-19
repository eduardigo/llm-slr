"""Valida os arquivos de critérios de inclusão de todos os temas."""
import pytest

from llm_slr.config import THEMES
from llm_slr.prompts.criteria_loader import load_criteria


@pytest.mark.parametrize("theme", sorted(THEMES))
def test_criteria_load_and_are_complete(theme):
    crit = load_criteria(theme)

    assert crit.theme == theme
    assert crit.topic
    assert crit.research_question.endswith("?")
    assert len(crit.inclusion_criteria) >= 3
    assert all(len(c) > 20 for c in crit.inclusion_criteria)

    # o schema não deve ganhar critérios de exclusão
    assert not hasattr(crit, "exclusion_criteria")
