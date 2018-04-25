import numpy as np
from sklearn.linear_model import LinearRegression


class OQT():

    def fit(self, X, y):

        n_class = len(np.unique(y))

        for n in range(n_class):
            pos_class = y[:n]

            mask = (y == pos_class)
            y_bin = np.ones(y.shape, dtype=np.int)
            y_bin[~mask] = 0

            clf = LinearRegression()


    def predict(self, X):
        pass