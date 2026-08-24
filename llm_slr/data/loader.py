import codecs
from dataclasses import dataclass

import bibtexparser


@dataclass(frozen=True)
class Article:

    title: str
    abstract: str
    year: int
    label: int
    source_file: str

    @property
    def content(self):
        return f"{self.title}\n{self.abstract}"


def load_articles(file_list):
    articles = []
    for path in file_list:
        with codecs.open(str(path), "r", encoding="utf-8") as bib_f:
            db = bibtexparser.load(bib_f)
        for entry in db.entries:
            articles.append(
                Article(
                    title=entry["title"],
                    abstract=entry["abstract"],
                    year=int(entry["year"]),
                    label=1 if entry["inserir"] == "true" else 0,
                    source_file=str(path),
                )
            )
    articles.sort(key=lambda a: a.year)
    return articles


def to_xy(articles, titles_only=False):
    X = [a.title if titles_only else a.content for a in articles]
    y = [a.label for a in articles]
    years = [a.year for a in articles]
    return X, y, years


def load_theme(theme):
    from llm_slr.config import theme_bib_files

    return load_articles(theme_bib_files(theme))
