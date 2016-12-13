from abc import ABCMeta, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import six
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from quantification import BasicModel
from quantification.classify_and_count.base import BaseMulticlassClassifyAndCount, BaseBinaryClassifyAndCount


class BinaryHDy(BaseBinaryClassifyAndCount):

    def predict(self, X, plot=False, method="hdy"):
        assert method == 'hdy'
        return self._predict_hdy(X, plot=plot)



class MulticlassHDy(BaseMulticlassClassifyAndCount):
    def predict(self, X, method="hdy"):
        assert method == "hdy"
        return self._predict_hdy(X)
