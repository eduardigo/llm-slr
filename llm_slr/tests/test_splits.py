"""Testes do split temporal, vindos de test/test_years_split.py do legado."""
from llm_slr.data.splits import YearsSplit


def test_years_split_separate_years():
    split = YearsSplit(n_split=3, years=[0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3])
    generator = split.split([], [])

    train, test = next(generator)
    assert train == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert test == [10, 11, 12, 13, 14]

    train, test = next(generator)
    assert train == [0, 1, 2, 3, 4]
    assert test == [5, 6, 7, 8, 9]

    train, test = next(generator)
    assert train == [0, 1]
    assert test == [2, 3, 4]

    try:
        next(generator)
        assert False, "esperava StopIteration"
    except StopIteration:
        pass


def test_years_split_should_group_years_with_less_than_5():
    split = YearsSplit(
        n_split=3, years=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3]
    )
    generator = split.split([], [])

    train, test = next(generator)
    assert train == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert test == [10, 11, 12, 13, 14, 15, 16, 17]
