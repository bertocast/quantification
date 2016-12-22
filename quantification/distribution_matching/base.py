from quantification.classify_and_count.base import BaseMulticlassClassifyAndCount, BaseBinaryClassifyAndCount
import numpy as np


class BinaryHDy(BaseBinaryClassifyAndCount):
    def predict(self, X, plot=False, method="hdy"):
        assert method == 'hdy'
        return self._predict_hdy(X, plot=plot)


class MulticlassHDy(BaseMulticlassClassifyAndCount):
    def predict(self, X, method="hdy"):
        assert method == "hdy"
        return self._predict_hdy(X)


class BinaryEM:
    def __init__(self):
        pass


    def fit(self, X, y):

        p_cls_s = np.bincount(y, minlength=2) / float(len(y))
        p_cls_s = p_cls_s[1]

        p_cond_pos = 0


        while True:
            p_cls = np.bincount(y, minlength=2) / float(len(y))
            p_cls = p_cls[1]
            num = p_cls / p_cls_s

