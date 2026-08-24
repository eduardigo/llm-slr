import pytest

from llm_slr.eval.metrics import evaluate_fold, metrics_at, sweep, wss_at_recall

SCORES = [7, 6, 3, 6, 4, 2, 1, 1]
LABELS = [1, 1, 1, 0, 0, 0, 0, 0]


def test_metrics_at_threshold_5():
    m = metrics_at(SCORES, LABELS, 5)
    assert (m.tp, m.fp, m.tn, m.fn) == (2, 1, 4, 1)
    assert m.precision == pytest.approx(2 / 3)
    assert m.recall == pytest.approx(2 / 3)
    assert m.f1 == pytest.approx(2 / 3)
    assert m.excluded == pytest.approx(4 / 8)
    assert m.missed == pytest.approx(1 / 3)


def test_metrics_at_include_all_threshold():
    m = metrics_at(SCORES, LABELS, 1)
    assert m.recall == 1.0
    assert m.excluded == 0.0
    assert m.missed == 0.0


def test_sweep_covers_unique_scores_descending():
    rows = sweep(SCORES, LABELS)
    assert [r.threshold for r in rows] == [7, 6, 4, 3, 2, 1]
    recalls = [r.recall for r in rows]
    assert recalls == sorted(recalls)


def test_wss_at_95_hand_computed():
    wss, threshold = wss_at_recall(SCORES, LABELS, target=0.95)
    assert threshold == 3
    assert wss == pytest.approx(3 / 8 - 0.05)


def test_wss_include_all_saves_nothing():
    wss, _ = wss_at_recall([5, 5, 5, 5], [1, 0, 1, 0], target=0.95)
    assert wss == pytest.approx(-0.05)


def test_evaluate_fold_summary():
    result = evaluate_fold(SCORES, LABELS)
    assert result["wss"] == pytest.approx(3 / 8 - 0.05)
    assert result["roc_auc"] == pytest.approx(5 / 6)
    assert result["best_f1"].f1 >= max(r.f1 for r in result["by_threshold"]) - 1e-9


def test_likert_thresholds_explicit():
    rows = sweep(SCORES, LABELS, thresholds=range(1, 8))
    assert len(rows) == 7
