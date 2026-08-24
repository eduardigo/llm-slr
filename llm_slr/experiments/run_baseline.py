import argparse
import csv
import json
from pathlib import Path

from llm_slr.config import RESULTS_DIR, THEMES
from llm_slr.eval.baseline import run_baseline_theme
from llm_slr.experiments.run import THEME_ORDER

RAW = Path(RESULTS_DIR) / "baseline_raw.csv"
FOLDS = Path(RESULTS_DIR) / "baseline_folds.csv"


def _append(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if new_file:
            writer.writeheader()
        writer.writerows(rows)


def _done_themes():
    if not FOLDS.exists():
        return set()
    with open(FOLDS, encoding="utf-8", newline="") as f:
        return {row["theme"] for row in csv.DictReader(f)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--themes", default=",".join(THEME_ORDER))
    args = parser.parse_args()

    done = _done_themes()
    for theme in [t for t in args.themes.split(",") if t in THEMES]:
        if theme in done:
            print(f"skip {theme} (completo)", flush=True)
            continue
        folds = run_baseline_theme(theme)
        raw_rows, fold_rows = [], []
        for fold_id, fold in enumerate(folds, start=1):
            for label, prob, raw_pred, cal_pred in zip(
                fold["y_true"], fold["probabilities"],
                fold["y_pred_raw"], fold["y_pred_calibrated"],
            ):
                raw_rows.append({
                    "theme": theme, "fold": fold_id, "label": label,
                    "probability": round(prob, 6),
                    "pred_raw": raw_pred, "pred_calibrated": cal_pred,
                })
            fold_rows.append({
                "theme": theme, "fold": fold_id,
                "threshold": round(fold["threshold"], 6),
                "best_params": json.dumps(fold["best_params"]),
            })
        _append(RAW, list(raw_rows[0]), raw_rows)
        _append(FOLDS, list(fold_rows[0]), fold_rows)
        print(f"ok   {theme} ({sum(len(f['y_true']) for f in folds)} artigos)",
              flush=True)


if __name__ == "__main__":
    main()
