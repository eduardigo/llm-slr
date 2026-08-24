import textwrap

from llm_slr.data.loader import load_articles, to_xy

FIXTURE = textwrap.dedent("""\
    @ARTICLE{I[1],
      inserir = {true},
      title = {Newer relevant study},
      abstract = {We study things.},
      year = {2019},
    }

    @ARTICLE{E[1],
      inserir = {false},
      title = {Old irrelevant study},
      abstract = {Unrelated content.},
      year = {2010},
    }
""")


def _write_fixture(tmp_path):
    path = tmp_path / "sample.bib"
    path.write_text(FIXTURE, encoding="utf-8")
    return path


def test_load_articles_sorts_by_year_and_parses_labels(tmp_path):
    articles = load_articles([_write_fixture(tmp_path)])

    assert [a.year for a in articles] == [2010, 2019]
    assert [a.label for a in articles] == [0, 1]
    assert articles[1].title == "Newer relevant study"
    assert articles[1].abstract == "We study things."


def test_to_xy_matches_legacy_format(tmp_path):
    articles = load_articles([_write_fixture(tmp_path)])

    X, y, years = to_xy(articles)
    assert X == [
        "Old irrelevant study\nUnrelated content.",
        "Newer relevant study\nWe study things.",
    ]
    assert y == [0, 1]
    assert years == [2010, 2019]

    X_titles, _, _ = to_xy(articles, titles_only=True)
    assert X_titles == ["Old irrelevant study", "Newer relevant study"]
