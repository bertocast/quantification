import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

from quantification.classify_and_count.base import BaseMulticlassClassifyAndCount, BaseBinaryClassifyAndCount, \
    BaseClassifyAndCountModel
from quantification.metrics import model_score


class BinaryHDy(BaseBinaryClassifyAndCount):
    def predict(self, X, plot=False, method="hdy"):
        assert method == 'hdy'
        return self._predict_hdy(X, plot=plot)


class MulticlassHDy(BaseMulticlassClassifyAndCount):
    def predict(self, X, method="hdy"):
        assert method == "hdy"
        return self._predict_hdy(X)


class BinaryEM(BaseClassifyAndCountModel):
    def __init__(self, estimator_class=None, estimator_params=None, estimator_grid=None, grid_params=None, tol=1e-9):
        super(BinaryEM, self).__init__(estimator_class, estimator_params, estimator_grid, grid_params, b=None)
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
    def __init__(self, estimator_class=None, estimator_params=None, estimator_grid=None, grid_params=None, tol=1e-9):
        super(MulticlassEM, self).__init__(estimator_class, estimator_params, estimator_grid, grid_params, b=None)
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


class BinaryCDEIter(BaseClassifyAndCountModel):
    def __init__(self, estimator_class=None, estimator_params=None, estimator_grid=None, grid_params=None, num_iter=3):
        super(BinaryCDEIter, self).__init__(estimator_class, estimator_params, estimator_grid, grid_params, b=None)
        self.num_iter = num_iter

    def fit(self, X, y):

        self.classes_ = np.unique(y)
        self.estimator_params['class_weight'] = dict(zip(self.classes_, [1, 1]))
        self.estimator_ = self._make_estimator()
        if isinstance(self.estimator_, GridSearchCV):
            if not hasattr(self.estimator_.estimator, 'class_weight'):
                raise ValueError("Classifier must have class_weight attribute in order to perform cost "
                                "sensitive classification")
        else:
            if not hasattr(self.estimator_, 'class_weight'):
                raise ValueError("Classifier must have class_weight attribute in order to perform cost "
                                 "sensitive classification")

        self.estimator_ = self.estimator_.fit(X, y)
        training_prevalences = np.bincount(y, minlength=2)
        self.pos_neg_orig = training_prevalences[1] / float(training_prevalences[0])
        self.X_train = X
        self.y_train = y
        return self

    def predict(self, X):

        n_iter = 0
        while n_iter < self.num_iter:
            pred = self.estimator_.predict(X)
            test_prevalences = np.bincount(pred, minlength=2)
            dmr = self.pos_neg_orig / (test_prevalences[1] / float(test_prevalences[0]))
            if np.isinf(dmr):
                self.estimator_params['class_weight'] = 'balanced'
            else:
                self.estimator_params['class_weight'] = dict(zip(self.classes_, [dmr, 1.0]))
            self.estimator = self._make_estimator()
            self.estimator_ = self.estimator_.fit(self.X_train, self.y_train)
            n_iter += 1

        final_preds = pred
        prevalences = np.bincount(final_preds, minlength=2) / float(len(final_preds))
        return prevalences


class MulticlassCDEIter(BaseClassifyAndCountModel):
    def __init__(self, estimator_class=None, estimator_params=None, estimator_grid=None, grid_params=None, num_iter=3):
        super(MulticlassCDEIter, self).__init__(estimator_class, estimator_params, estimator_grid, grid_params, b=None)
        self.num_iter = num_iter

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

            qnf = BinaryCDEIter(estimator_class=self.estimator_class, estimator_params=self.estimator_params,
                            estimator_grid=self.estimator_grid, num_iter=self.num_iter)

            qnf.fit(X, y_bin)
            self.qnfs_[pos_class] = qnf
        return self

    def predict(self, X):
        prev = dict.fromkeys(self.classes_)

        for pos_class in self.classes_:
            pred = self.qnfs_[pos_class].predict(X)
            prev[pos_class] = pred[1]

        return np.array(prev.values())


class BinaryCDEAC(BaseClassifyAndCountModel):
    def __init__(self, estimator_class=None, estimator_params=None, estimator_grid=None, grid_params=None, num_iter=3):
        super(BinaryCDEAC, self).__init__(estimator_class, estimator_params, estimator_grid, grid_params, b=None)
        self.num_iter = num_iter

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.estimator_params['class_weight'] = dict(zip(self.classes_, [1, 1]))
        self.estimator_ = self._make_estimator()
        if not hasattr(self.estimator_, 'class_weight'):
            raise ValueError("Classifier must have class_weight attribute in order to perform cost "
                             "sensitive classification")
        self.estimator_ = self.estimator_.fit(X, y)
        training_prevalences = np.bincount(y, minlength=2)
        self.pos_neg_orig = training_prevalences[1] / float(training_prevalences[0])
        self.X_train = X
        self.y_train = y
        cm = model_score.cv_confusion_matrix(self.estimator_, X, y, 10)
        cm = np.mean(cm, axis=0)
        self.tpr_ = cm[1, 1] / float(cm[1, 1] + cm[1, 0])
        self.fpr_ = cm[0, 1] / float(cm[0, 1] + cm[0, 0])

        return self


    def predict(self, X):

        n_iter = 0
        while n_iter < self.num_iter:
            pred = self.estimator_.predict(X)
            test_prevalences = np.bincount(pred, minlength=2)
            corrected = (test_prevalences[1] - self.fpr_) / (self.tpr_ - self.fpr_)
            test_prevalences = np.array([1 - corrected, corrected])
            dmr = self.pos_neg_orig / (test_prevalences[1] / float(test_prevalences[0]))
            if np.isinf(dmr):
                self.estimator_params['class_weight'] = 'balanced'
            else:
                self.estimator_params['class_weight'] = dict(zip(self.classes_, [dmr, 1.0]))
            self.estimator = self._make_estimator()
            self.estimator_ = self.estimator_.fit(self.X_train, self.y_train)
            n_iter += 1

        final_preds = pred
        prevalences = np.bincount(final_preds, minlength=2) / float(len(final_preds))
        return prevalences


class MulticlassCDEAC(BaseClassifyAndCountModel):
    def __init__(self, estimator_class=None, estimator_params=None, estimator_grid=None, grid_params=None, num_iter=3):
        super(MulticlassCDEAC, self).__init__(estimator_class, estimator_params, estimator_grid, grid_params, b=None)
        self.num_iter = num_iter

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

            qnf = BinaryCDEAC(estimator_class=self.estimator_class, estimator_params=self.estimator_params,
                            estimator_grid=self.estimator_grid, num_iter=self.num_iter)

            qnf.fit(X, y_bin)
            self.qnfs_[pos_class] = qnf
        return self

    def predict(self, X):
        prev = dict.fromkeys(self.classes_)

        for pos_class in self.classes_:
            pred = self.qnfs_[pos_class].predict(X)
            prev[pos_class] = pred[1]

        return np.array(prev.values())