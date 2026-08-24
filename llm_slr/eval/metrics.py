from dataclasses import dataclass

from sklearn.metrics import roc_auc_score


@dataclass(frozen=True)
class ThresholdMetrics:
    threshold: float
    tp: int
    fp: int
    tn: int
    fn: int
    precision: float
    recall: float
    f1: float
    excluded: float
    missed: float

    @property
    def n(self):
        return self.tp + self.fp + self.tn + self.fn


def metrics_at(scores, labels, threshold):
    if len(scores) != len(labels) or not scores:
        raise ValueError("scores e labels devem ter o mesmo tamanho (>0)")

    tp = fp = tn = fn = 0
    for score, label in zip(scores, labels):
        predicted = score >= threshold
        if predicted and label == 1:
            tp += 1
        elif predicted and label == 0:
            fp += 1
        elif not predicted and label == 0:
            tn += 1
        else:
            fn += 1

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall else 0.0)
    n = len(scores)
    return ThresholdMetrics(
        threshold=threshold, tp=tp, fp=fp, tn=tn, fn=fn,
        precision=precision, recall=recall, f1=f1,
        excluded=tn / n,
        missed=fn / (tp + fn) if tp + fn else 0.0,
    )


def sweep(scores, labels, thresholds=None):
    if thresholds is None:
        thresholds = sorted(set(scores), reverse=True)
    return [metrics_at(scores, labels, t) for t in thresholds]


def wss_at_recall(scores, labels, target=0.95, thresholds=None):
    rows = sweep(scores, labels, thresholds)
    feasible = [r for r in rows if r.recall >= target]
    best = max(feasible, key=lambda r: (r.tn + r.fn) / r.n)
    return (best.tn + best.fn) / best.n - (1 - target), best.threshold


def evaluate_fold(scores, labels, thresholds=None, wss_target=0.95):
    rows = sweep(scores, labels, thresholds)
    wss, wss_threshold = wss_at_recall(scores, labels, wss_target, thresholds)
    auc = (roc_auc_score(labels, scores)
           if len(set(labels)) > 1 else float("nan"))
    return {
        "by_threshold": rows,
        "wss": wss,
        "wss_threshold": wss_threshold,
        "roc_auc": auc,
        "best_f1": max(rows, key=lambda r: r.f1),
    }
