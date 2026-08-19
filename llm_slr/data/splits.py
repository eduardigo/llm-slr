"""Split temporal que simula a atualização de uma RSL.

Segue util/years_split.py do legado: assume o dataset ordenado por ano
crescente e gera folds em que o treino são os artigos mais antigos e o teste
é o bloco mais recente, que é o cenário de uma atualização.

Aqui o bloco de treino de cada fold é de onde saem os exemplos few-shot; o
bloco de teste são os artigos novos a classificar.
"""


class YearsSplit:
    """Gera os folds temporais, como o years_split.py do legado."""

    def __init__(self, n_split=3, years=()):
        self._years = list(years)
        self._n_split = n_split

    def split(self, X, y=None, groups=None):
        max_value = max(self._years)
        max_index = self._years.index(max_value)
        prev_len = len(self._years)

        # bloco de teste com menos de 5 artigos é fundido ao anterior
        if len(self._years[max_index:]) < 5:
            max_value = max(self._years[:max_index])
            max_index = self._years[:max_index].index(max_value)

        for _ in range(self._n_split):
            yield list(range(0, max_index)), list(range(max_index, prev_len))
            prev_len = max_index
            max_value = max(self._years[:max_index])
            max_index = self._years[:max_index].index(max_value)


def temporal_folds(articles, n_split=3):
    """Mesmos folds, porém como listas de Article: [(treino, teste), ...]."""
    years = [a.year for a in articles]
    splitter = YearsSplit(n_split=n_split, years=years)
    return [
        ([articles[i] for i in train], [articles[i] for i in test])
        for train, test in splitter.split(articles)
    ]
