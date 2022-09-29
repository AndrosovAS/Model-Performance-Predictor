from sklearn.metrics import roc_auc_score as ras
import numpy as np


def accuracy_score(y_true, y_pred):
    TP = sum(y_true & y_pred)
    FP = sum(y_true & ~y_pred)
    FN = sum(~y_true & y_pred)
    TN = sum(~y_true & ~y_pred)

    return (TP + TN) / (TP + FP + FN + TN)

def precision_score(y_true, y_pred):
    TP = sum(y_true & y_pred)
    FP = sum(y_true & ~y_pred)

    try:
        return TP / (TP + FP) 
    except:
        return None

def recall_score(y_true, y_pred):
    TP = sum(y_true & y_pred)
    FN = sum(~y_true & y_pred)

    try:
        return TP / (TP + FN)
    except:
        return None

def f_score(y_true, y_pred, beta=1):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    return (1 + beta**2) * (precision*recall) / (beta**2 * precision + recall)

def roc_auc_score(y_true, y_pred):
    try:
        return ras(y_true, y_pred)
    except:
        return None

def MWS(y_true, y_pred):
    """
    Средневзвешанная метрика.
    out: mean(weight * (y_true == y_pred))
    """
    unique = np.unique(y_true)
    length = len(unique)
    num = np.array([sum(y_true == u) for u in unique])
    total = sum(num)
    w = {u: total / (length*n) for n,u in zip(num, unique)}

    n = len(y_true)
    weight = np.zeros(n)
    for k in range(n):
        weight[k] = w[y_true[k]]
    return np.mean(weight * (y_pred == y_true))

def MAE(y_true, y_pred):
    return np.mean(abs(y_true - y_pred))

def MRE(y_true, y_pred):
    return np.mean(abs( (y_true - y_pred)/ y_true))
