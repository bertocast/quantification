import numpy as np
from sklearn.datasets.base import Bunch, load_breast_cancer, load_iris


class ModelTestCase:
    def setup(self):
        self.Xb, self.yb = load_breast_cancer(return_X_y=True)


        self.Xmc, self.ymc = load_iris(return_X_y=True)