"""Leitura dos datasets de RSL a partir dos .bib do repositório legado.

Segue util/bib_loader.py do legado, com uma diferença: além do formato
(X, y, years) que a baseline consome, devolve os artigos como registros com
título e abstract separados, que é o que a montagem dos prompts precisa.

A ordenação por ano é estável, como no legado, então os índices dos folds
temporais coincidem com os da baseline.
"""
import codecs
from dataclasses import dataclass

import bibtexparser


@dataclass(frozen=True)
class Article:
    """Um estudo candidato e a decisão tomada na RSL original."""

    title: str
    abstract: str
    year: int
    label: int          # 1 = incluído (inserir=true), 0 = excluído
    source_file: str    # .bib de origem

    @property
    def content(self):
        """Texto no formato que a baseline usa (título\\nabstract)."""
        return f"{self.title}\n{self.abstract}"


def load_articles(file_list):
    """Carrega e ordena os artigos por ano, como o loader legado.

    A ordenação usa list.sort com chave 'year', que é estável, então o fold k
    daqui corresponde ao fold k da baseline.
    """
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
    """Converte para o formato (X, y, years) que a baseline consome."""
    X = [a.title if titles_only else a.content for a in articles]
    y = [a.label for a in articles]
    years = [a.year for a in articles]
    return X, y, years


def load_theme(theme):
    """Carrega todos os artigos de um tema listado em config.THEMES."""
    from llm_slr.config import theme_bib_files

    return load_articles(theme_bib_files(theme))
