"""Baseline SVM/TF-IDF de Watanabe et al. (2020), nos folds usados aqui.

Reproduz o main.py legado na configuração tfidf-svm-selectkbest, que é a
melhor reportada no run.sh: importa os filtros de texto do próprio legado e
repete o extractor TF-IDF, o grid de hiperparâmetros e a calibração de
threshold por recall no treino. Duas diferenças, nenhuma delas afeta os
resultados:
- n_jobs=-1 no GridSearchCV, só paralelismo;
- além das predições binárias, devolve as probabilidades do teste, para que
  eval.metrics calcule o WSS@95 da baseline com o mesmo código dos LLMs.
"""
import sys

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from llm_slr.config import LEGACY_ROOT, N_SPLITS, SEED
from llm_slr.data.loader import load_theme, to_xy
from llm_slr.data.splits import YearsSplit

sys.path.insert(0, str(LEGACY_ROOT))
from util.text_filter import (  # noqa: E402
    FilterComposite, LemmatizerFilter, StopwordsFilter,
)

# Mesmo grid do config legado.
LEGACY_GRID = {
    "classifier__kernel": ["linear", "rbf"],
    "classifier__C": [1, 10, 100],
    "classifier__coef0": [0, 10, 100],
    "classifier__tol": [0.001, 0.1, 1],
    "classifier__class_weight": ["balanced", None],
    "selector__k": [25, 50, 100, 200, "all"],
}


def build_extractor():
    return Pipeline([
        ("extractor", TfidfVectorizer(ngram_range=(1, 3))),
        ("scaler", StandardScaler(with_mean=False)),
    ])


def build_grid_search(grid=None, seed=SEED):
    return GridSearchCV(
        Pipeline([
            ("selector", SelectKBest(chi2)),
            ("classifier", SVC(random_state=seed, probability=True)),
        ]),
        grid if grid is not None else LEGACY_GRID,
        cv=3, scoring="f1", n_jobs=-1,
    )


def calibrate_threshold(pipeline, X_train_ext, y_train):
    """Maior threshold que mantém recall 1 no treino, limitado a 0.5.

    Mesma lógica do main.py legado: prioriza não perder artigos relevantes.
    """
    true_index = list(pipeline.classes_).index(1)
    y_prob = pipeline.predict_proba(X_train_ext)[:, true_index]
    _, recall, thresholds = precision_recall_curve(y_train, y_prob)

    ind = -1
    for value in recall.tolist():
        if value == 1:
            ind += 1
        else:
            break
    return min(thresholds[ind], 0.5)


def run_baseline_theme(theme, titles_only=False, grid=None):
    """Roda a baseline nos folds temporais de um tema.

    Devolve, por fold, as probabilidades do teste, os labels, o threshold
    calibrado e as predições binárias antes e depois da calibração, que é o
    necessário para reproduzir as métricas do main.py legado.
    """
    X, y, years = to_xy(load_theme(theme), titles_only=titles_only)
    preprocessor = FilterComposite(
        filters=[StopwordsFilter(), LemmatizerFilter()]
    )
    X = np.array(preprocessor.fit_transform(X))
    y = np.array(y)

    folds = []
    splitter = YearsSplit(n_split=N_SPLITS, years=years)
    for train_index, test_index in splitter.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        extractor = build_extractor()
        X_train_ext = extractor.fit_transform(X_train)
        X_test_ext = extractor.transform(X_test)

        pipeline = build_grid_search(grid)
        pipeline.fit(X_train_ext, y_train)

        true_index = list(pipeline.classes_).index(1)
        threshold = calibrate_threshold(pipeline, X_train_ext, y_train)
        probabilities = pipeline.predict_proba(X_test_ext)[:, true_index]

        folds.append({
            "y_true": y_test.tolist(),
            "y_pred_raw": pipeline.predict(X_test_ext).tolist(),
            "probabilities": probabilities.tolist(),
            "threshold": float(threshold),
            "y_pred_calibrated": [
                0 if p < threshold else 1 for p in probabilities.tolist()
            ],
            "best_params": pipeline.best_params_,
        })
    return folds
