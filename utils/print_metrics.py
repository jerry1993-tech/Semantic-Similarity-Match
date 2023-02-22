import unicodedata
import numpy as np
from sklearn import metrics


def print_table(metrics, val):
    # assert len(metrics) > 0 and len(metrics) == len(val)

    def wide_chars(s):
        return sum(unicodedata.east_asian_width(x) == 'W' for x in s)

    def width(s):
        return len(s) + wide_chars(s)

    _max = max([width(str(i)) for i in metrics + val])

    for i, j in zip(metrics, val):
        print("-" * (_max * 2 + 5))
        print(f" {i:^{_max-wide_chars(str(i))}} | {str(j):^{_max-wide_chars(str(j))}} ")
    print("-" * (_max * 2 + 5))


def binary_cal_metrics(y_true, y_pred):
    labels = [0, 1]
    # 评价预测效果，计算混淆矩阵
    print(metrics.classification_report(y_true,
                                        y_pred.round(),
                                        labels=labels,
                                        zero_division=0,
                                        digits=4))

    confm = metrics.confusion_matrix(y_true,
                                     y_pred.round(),
                                     labels=labels)
    print("confusion matrix=", confm)


def multiclass_cal_metrics(y_true, y_pred):
    labels = [0, 1]  # 例如 num_classes = 2
    # 评价预测效果，计算混淆矩阵
    print(metrics.classification_report(np.argmax(y_true, axis=1),
                                        np.argmax(y_pred, axis=1),
                                        labels=labels,
                                        zero_division=0,
                                        digits=4))

    confm = metrics.confusion_matrix(np.argmax(y_true, axis=1),
                                     np.argmax(y_pred, axis=1),
                                     labels=labels)
    print("confusion matrix=", confm)

