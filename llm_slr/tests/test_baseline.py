from llm_slr.eval.baseline import run_baseline_theme
from llm_slr.eval.metrics import evaluate_fold

TINY_GRID = {
    "classifier__kernel": ["linear"],
    "classifier__C": [1],
    "classifier__class_weight": ["balanced"],
    "selector__k": ["all"],
}


def test_baseline_runs_and_outputs_are_consistent():
    folds = run_baseline_theme("slr", grid=TINY_GRID)

    assert len(folds) == 3
    for fold in folds:
        n = len(fold["y_true"])
        assert n >= 5
        assert len(fold["probabilities"]) == n
        assert len(fold["y_pred_calibrated"]) == n
        assert 0.0 < fold["threshold"] <= 0.5
        assert all(0.0 <= p <= 1.0 for p in fold["probabilities"])

        result = evaluate_fold(fold["probabilities"], fold["y_true"])
        assert -0.05 - 1e-9 <= result["wss"] <= 0.95
