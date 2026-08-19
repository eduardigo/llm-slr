"""Consolida raw.csv e baseline_raw.csv nas tabelas de resultado.

Cada linha do resumo é um (tema, abordagem, fold), com abordagem igual a
"{modelo}/{estratégia}" nos LLMs ou "svm-baseline". Todas passam por
eval.metrics.evaluate_fold: os LLMs com thresholds de 1 a 7, a baseline com
as probabilidades.

Uso: python -m llm_slr.eval.summary, que imprime o que houver em results/.
"""
from pathlib import Path

import pandas as pd

from llm_slr.config import RESULTS_DIR
from llm_slr.eval.metrics import evaluate_fold, metrics_at

RAW_CSV = Path(RESULTS_DIR) / "raw.csv"
BASELINE_RAW = Path(RESULTS_DIR) / "baseline_raw.csv"
SUMMARY_CSV = Path(RESULTS_DIR) / "summary.csv"

LIKERT_THRESHOLDS = range(1, 8)


def _fold_row(theme, approach, fold, evaluation, extra=None):
    best = evaluation["best_f1"]
    return {
        "theme": theme, "approach": approach, "fold": fold,
        "wss95": evaluation["wss"],
        "wss95_threshold": evaluation["wss_threshold"],
        "roc_auc": evaluation["roc_auc"],
        "best_f1": best.f1, "best_f1_threshold": best.threshold,
        "precision_at_best_f1": best.precision,
        "recall_at_best_f1": best.recall,
        **(extra or {}),
    }


def llm_summary(raw=None):
    if raw is None:
        raw = pd.read_csv(RAW_CSV)
    valid = raw[raw["score"].notna() & (raw["error"].fillna("") == "")]
    rows = []
    group_cols = ["theme", "model", "strategy", "fold"]
    for (theme, model, strategy, fold), g in valid.groupby(group_cols):
        if g["label"].nunique() < 2:
            continue  # fold com uma classe só não tem métrica definida
        evaluation = evaluate_fold(
            g["score"].astype(int).tolist(), g["label"].tolist(),
            thresholds=LIKERT_THRESHOLDS,
        )
        fresh = g[~g["from_cache"].astype(bool)]["latency_s"]
        n_total = len(raw[(raw["theme"] == theme) & (raw["model"] == model)
                          & (raw["strategy"] == strategy)
                          & (raw["fold"] == fold)])
        rows.append(_fold_row(
            theme, f"{model}/{strategy}", fold, evaluation,
            extra={
                "n": len(g),
                "n_errors": n_total - len(g),
                "mean_latency_s": (round(fresh.mean(), 2)
                                   if len(fresh) else None),
            },
        ))
    return pd.DataFrame(rows)


def baseline_summary(raw=None):
    if raw is None:
        raw = pd.read_csv(BASELINE_RAW)
    rows = []
    for (theme, fold), g in raw.groupby(["theme", "fold"]):
        if g["label"].nunique() < 2:
            continue
        labels = g["label"].tolist()
        evaluation = evaluate_fold(g["probability"].tolist(), labels)
        calibrated = metrics_at(g["pred_calibrated"].tolist(), labels, 1)
        rows.append(_fold_row(
            theme, "svm-baseline", fold, evaluation,
            extra={
                "n": len(g), "n_errors": 0, "mean_latency_s": None,
                "calibrated_recall": calibrated.recall,
                "calibrated_precision": calibrated.precision,
                "calibrated_excluded": calibrated.excluded,
                "calibrated_missed": calibrated.missed,
            },
        ))
    return pd.DataFrame(rows)


def build_summary(save=True):
    parts = []
    if RAW_CSV.exists():
        parts.append(llm_summary())
    if BASELINE_RAW.exists():
        parts.append(baseline_summary())
    if not parts:
        raise FileNotFoundError("nenhum resultado em results/ ainda")
    summary = pd.concat(parts, ignore_index=True)
    if save:
        SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(SUMMARY_CSV, index=False)
    return summary


def pivot_mean(summary, metric="wss95"):
    """Tabela tema x abordagem com a média da métrica entre os folds."""
    return summary.pivot_table(index="approach", columns="theme",
                               values=metric, aggfunc="mean").round(3)


if __name__ == "__main__":
    df = build_summary()
    combos = df.groupby("approach")["fold"].count()
    print(f"folds avaliados por abordagem:\n{combos}\n")
    for metric in ("wss95", "best_f1", "roc_auc"):
        print(f"===== média de {metric} por tema × abordagem =====")
        print(pivot_mean(df, metric).to_string(), "\n")
