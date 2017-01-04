from sklearn.model_selection import GridSearchCV

from quantification.classify_and_count.base import BaseMulticlassClassifyAndCount, BaseBinaryClassifyAndCount, \
    BaseClassifyAndCountModel
import numpy as np


class BinaryHDy(BaseBinaryClassifyAndCount):
    def predict(self, X, plot=False, method="hdy"):
        assert method == 'hdy'
        return self._predict_hdy(X, plot=plot)


class MulticlassHDy(BaseMulticlassClassifyAndCount):
    def predict(self, X, method="hdy"):
        assert method == "hdy"
        return self._predict_hdy(X)


class BinaryEM(BaseClassifyAndCountModel):
    def __init__(self, estimator_class=None, estimator_params=None, estimator_grid=None, tol=1e-9):
        super(BinaryEM, self).__init__(estimator_class, estimator_params, estimator_grid, b=None)
        self.tol = tol

    def fit(self, X, y):
        self.estimator_ = self._make_estimator()
        self.estimator_.fit(X, y)
        if isinstance(self.estimator_, GridSearchCV):
            self.estimator_ = self.estimator_.best_estimator_
        self.prob_tr_ = (np.bincount(y, minlength=2) / float(len(y)))[1]
        return self


    def predict(self, X):

        m = X.shape[0]
        p_cond_tr = self.estimator_.predict_proba(X)[:, 1]
        p_s = self.prob_tr_


        while True:
            p_s_last_ = p_s
            num = p_s / self.prob_tr_ * p_cond_tr
            denom = num + (1 - p_s) / (1 - self.prob_tr_) * (1 - p_cond_tr)
            p_cond_s = num / denom

            p_s = np.sum(p_cond_s) / m

            if abs(p_s - p_s_last_) < self.tol:
                break

        return np.array([1 - p_s, p_s])


class MulticlassEM(BaseClassifyAndCountModel):
    def __init__(self, estimator_class=None, estimator_params=None, estimator_grid=None, tol=1e-9):
        super(MulticlassEM, self).__init__(estimator_class, estimator_params, estimator_grid, b=None)
        self.tol = tol

    def fit(self, X, y, verbose=False):
        self.classes_ = np.unique(y).tolist()
        n_classes = len(self.classes_)
        self.qnfs_ = dict.fromkeys(self.classes_)

        for pos_class in self.classes_:
            if verbose:
                print "Fitting classifier for class {}/{}".format(pos_class + 1, n_classes)
            mask = (y == pos_class)
            y_bin = np.ones(y.shape, dtype=np.int)
            y_bin[~mask] = 0

            qnf = BinaryEM(estimator_class=self.estimator_class, estimator_params=self.estimator_params,
                           estimator_grid=self.estimator_grid, tol=self.tol)

            qnf.fit(X, y_bin)
            self.qnfs_[pos_class] = qnf
        return self

    def predict(self, X):
        prev = dict.fromkeys(self.classes_)

        for pos_class in self.classes_:
            pred = self.qnfs_[pos_class].predict(X)
            prev[pos_class] = pred[1]

        return prev.values()



