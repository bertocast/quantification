import numpy as np
from sklearn.datasets.base import Bunch, load_breast_cancer, load_iris


class ModelTestCase:
    def setup(self):
        X, y = load_breast_cancer(return_X_y=True)
        idx = np.random.choice(range(len(y)), (10, 100), replace=True)
        self.binary_X = [X[idx_] for idx_ in idx.tolist()]
        self.binary_y = [y[idx_] for idx_ in idx.tolist()]


        X, y = load_iris(return_X_y=True)
        idx = np.random.choice(range(len(y)), (10, 100), replace=True)
        self.mc_X = [X[idx_] for idx_ in idx.tolist()]
        self.mc_y = [y[idx_] for idx_ in idx.tolist()]