import numpy as np
from collections import Counter


# ==========================================================
# BASIC METRICS
# ==========================================================

def accuracy(preds, labels):
    return (preds == labels).mean() * 100


def confusion_matrix(preds, labels, num_classes):
    """
    Returns confusion matrix of shape (num_classes, num_classes).
    cm[true][pred]
    """
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(labels, preds):
        cm[int(t), int(p)] += 1
    return cm


# ==========================================================
# PRECISION / RECALL / F1
# ==========================================================

def precision_recall_f1(cm):
    """
    Given confusion matrix (C, C), returns per-class and macro-averaged
    precision, recall, and F1.

    Returns dict:
        per_class: list of dicts with keys precision, recall, f1
        macro:     dict with keys precision, recall, f1
    """
    C = cm.shape[0]
    per_class = []

    for i in range(C):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp

        prec   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1     = (2 * prec * recall / (prec + recall)
                  if (prec + recall) > 0 else 0.0)

        per_class.append({"precision": prec, "recall": recall, "f1": f1})

    macro = {
        "precision": np.mean([c["precision"] for c in per_class]),
        "recall":    np.mean([c["recall"]    for c in per_class]),
        "f1":        np.mean([c["f1"]        for c in per_class]),
    }

    return {"per_class": per_class, "macro": macro}


# ==========================================================
# PRINT REPORT
# ==========================================================

def print_distribution(train_labels, test_labels, shapes):
    train_dist = Counter(train_labels.tolist())
    test_dist  = Counter(test_labels.tolist())

    print(f"\n{'Class':<14} {'Train':>6} {'Test':>6}")
    print("-" * 28)
    for i, shape in enumerate(shapes):
        print(f"{shape:<14} {train_dist[i]:>6} {test_dist[i]:>6}")
    print("-" * 28)
    print(f"{'Total':<14} {sum(train_dist.values()):>6} {sum(test_dist.values()):>6}\n")


def print_classification_report(preds, labels, shapes):
    num_classes = len(shapes)
    cm     = confusion_matrix(preds, labels, num_classes)
    report = precision_recall_f1(cm)
    acc    = accuracy(preds, labels)

    print(f"\nOverall Accuracy: {acc:.2f}%")
    print(f"\n{'Class':<14} {'Prec':>7} {'Recall':>7} {'F1':>7} {'Total':>7}")
    print("-" * 46)

    for i, shape in enumerate(shapes):
        m = report["per_class"][i]
        total = int(labels.tolist().count(i)) if hasattr(labels, 'tolist') else sum(1 for l in labels if l == i)
        print(
            f"{shape:<14}"
            f" {m['precision']*100:>6.1f}%"
            f" {m['recall']*100:>6.1f}%"
            f" {m['f1']*100:>6.1f}%"
            f" {total:>7}"
        )

    print("-" * 46)
    macro = report["macro"]
    print(
        f"{'Macro avg':<14}"
        f" {macro['precision']*100:>6.1f}%"
        f" {macro['recall']*100:>6.1f}%"
        f" {macro['f1']*100:>6.1f}%"
    )

    return cm, report