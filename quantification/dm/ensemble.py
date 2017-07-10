import numpy as np

from quantification.cc.ensemble import EnsembleMulticlassCC, EnsembleBinaryCC, BaseEnsembleCCModel
from quantification.dm.base import BinaryEM, MulticlassEM


class BinaryEnsembleHDy(EnsembleBinaryCC):
    """"""
    def predict(self, X, method='hdy'):
        assert method == "hdy"
        return self._predict_hdy(X)


class MulticlassEnsembleHDy(EnsembleMulticlassCC):
    def predict(self, X, method='hdy'):
        assert method == "hdy"
        return self._predict_hdy(X)


class BinaryEnsembleEM(BaseEnsembleCCModel):
    def __init__(self, estimator_class=None, estimator_params=None, estimator_grid=None, tol=1e-9):
        super(BinaryEnsembleEM, self).__init__(estimator_class, estimator_params, estimator_grid, b=None)
        self.tol = tol

    def fit(self, X, y):
        self.qnfs_ = []

        if len(X) != len(y):
            raise ValueError("X and y has to be the same length.")
        for n, (X_sample, y_sample) in enumerate(zip(X, y)):
            if len(np.unique(y_sample)) < 2:
                continue
            qnf = BinaryEM(estimator_class=self.estimator_class, estimator_params=self.estimator_params,
                           estimator_grid=self.estimator_grid, tol=self.tol)

            qnf.fit(X, y)

            self.qnfs_.append(qnf)

        return self

    def predict(self, X):
        predictions = np.full((len(self.qnfs_), 2), np.nan)

        for n, qnf in enumerate(self.qnfs_):
            prevs = qnf.predict()
            predictions[n] = prevs
        predictions = np.mean(predictions, axis=0)
        return predictions


class MulticlassEnsembleEM(BaseEnsembleCCModel):
    def __init__(self, estimator_class=None, estimator_params=None, estimator_grid=None, tol=1e-9):
        super(MulticlassEnsembleEM, self).__init__(estimator_class, estimator_params, estimator_grid, b=None)
        self.tol = tol

    def fit(self, X, y, verbose=False):
        if len(X) != len(y):
            raise ValueError("X and y has to be the same length.")

        self.classes_ = np.unique(np.concatenate(y)).tolist()
        self.qnfs_ = {k: [] for k in self.classes_}
        for n, (X_sample, y_sample) in enumerate(zip(X, y)):

            classes = np.unique(y_sample).tolist()
            for cls in classes:
                mask = (y_sample == cls)
                y_bin = np.ones(y_sample.shape, dtype=np.int)
                y_bin[~mask] = 0
                if len(np.unique(y_bin)) != 2 or np.any(np.bincount(y_bin) < 3):
                    continue
                if verbose:
                    print "\tFitting classifier for class {}".format(cls + 1)

                qnf = BinaryEM(estimator_class=self.estimator_class, estimator_params=self.estimator_params,
                               estimator_grid=self.estimator_grid, tol=self.tol)
                qnf = qnf.fit(X_sample, y_bin)
                self.qnfs_[cls].append(qnf)
        return self

    def predict(self, X):
        predictions = np.empty(len(self.classes_))
        for cls in self.classes_:
            preds = []
            for qnf in self.qnfs_[cls]:
                preds.append(qnf.predict(X)[1])
            predictions[cls] = np.mean(preds)

        return predictions / np.sum(predictions)
