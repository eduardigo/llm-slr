"""Paridade com o código legado, sobre os .bib reais.

Confere que o pipeline lê os mesmos dados, na mesma ordem, e produz os
mesmos folds temporais que a baseline. Se algo divergir aqui, a comparação
fold a fold deixa de valer.
"""
import sys

import pytest

from llm_slr.config import LEGACY_ROOT, N_SPLITS, THEMES, theme_bib_files
from llm_slr.data.loader import load_theme, to_xy
from llm_slr.data.splits import YearsSplit

sys.path.insert(0, str(LEGACY_ROOT))
from util.bib_loader import load as legacy_load  # noqa: E402
from util.years_split import YearsSplit as LegacyYearsSplit  # noqa: E402


@pytest.mark.parametrize("theme", sorted(THEMES))
def test_loader_matches_legacy(theme):
    legacy_X, legacy_y, legacy_years = legacy_load(
        [str(p) for p in theme_bib_files(theme)]
    )
    X, y, years = to_xy(load_theme(theme))

    assert X == legacy_X
    assert y == legacy_y
    assert years == legacy_years


@pytest.mark.parametrize("theme", sorted(THEMES))
def test_folds_match_legacy(theme):
    _, _, years = to_xy(load_theme(theme))

    ours = list(YearsSplit(n_split=N_SPLITS, years=years).split([], []))
    legacy = list(LegacyYearsSplit(n_split=N_SPLITS, years=years).split([], []))

    assert ours == legacy
