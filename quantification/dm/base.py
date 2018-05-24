import quadprog
import warnings
from abc import ABCMeta

import numpy as np
import six
from shapely.geometry import Polygon
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV, cross_val_predict
from scipy.stats import rankdata, norm

from quantification.utils.base import gss
from quantification.cc.base import BaseCC, \
    BaseClassifyAndCountModel
from quantification.metrics import model_score
from quantification.utils.base import is_pd, nearest_pd


class BinaryEM(BaseClassifyAndCountModel):
    """
        Binary EM method.

        Parameters
        -----------
        tol : float, optional, default=1e-9
            Minimum error before stopping the learning process.

        estimator_class : object, optional
            An instance of a classifier class. It has to have fit and predict methods. It is highly advised to use one of
            the implementations in sklearn library. If it is leave as None, Logistic Regression will be used.

        estimator_params : dictionary, optional
            Additional params to initialize the classifier.

        estimator_grid : dictionary, optional
            During training phase, grid search is performed. This parameter should provided the parameters of the classifier
            that will be tested (e.g. estimator_grid={C: [0.1, 1, 10]} for Logistic Regression).

        Attributes
        ----------
        estimator_ : object
            The underlying classifier

        prob_tr_ = float
            Prevalence of the positive class in the training set.

        """

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

            if np.abs(p_s - p_s_last_) < self.tol:
                break

        return np.array([1 - p_s, p_s])


class BinaryCDEIter(BaseClassifyAndCountModel):
    """
            Binary CDE Iterate method.

            Parameters
            -----------
            num_iter : float, optional, default=3
                Number of iterations.

            estimator_class : object, optional
                An instance of a classifier class. It has to have fit and predict methods. It is highly advised to use one of
                the implementations in sklearn library. If it is leave as None, Logistic Regression will be used.

            estimator_params : dictionary, optional
                Additional params to initialize the classifier.

            estimator_grid : dictionary, optional
                During training phase, grid search is performed. This parameter should provided the parameters of the classifier
                that will be tested (e.g. estimator_grid={C: [0.1, 1, 10]} for Logistic Regression).

            Attributes
            ----------
            estimator_ : object
                The underlying classifier

            pos_neg_orig : float
                Original ratio between positive and negatives examples.

            X_train : array, shape=(num_samples, num_features)
                Training features dataset.

            y_train : array, shape=(num_samples,)
                Class of each example in the training dataset.

            """

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

    def predict(self, X, verbose=False):

        n_iter = 0
        while n_iter < self.num_iter:
            if verbose:
                print("Iteration", n_iter)
            pred = self.estimator_.predict(X)
            test_prevalences = np.bincount(pred, minlength=2)
            dmr = self.pos_neg_orig / (test_prevalences[1] / float(test_prevalences[0]))
            if np.isinf(dmr):
                self.estimator_params['class_weight'] = 'balanced'
            else:
                self.estimator_params['class_weight'] = dict(zip(self.classes_, [dmr, 1.0]))
            if isinstance(self.estimator_, GridSearchCV):
                self.estimator_.estimator.set_params(**self.estimator_params)
            else:
                self.estimator_.set_params(**self.estimator_params)
            self.estimator_ = self.estimator_.fit(self.X_train, self.y_train)
            n_iter += 1

        final_preds = pred
        prevalences = np.bincount(final_preds, minlength=2) / float(len(final_preds))
        return prevalences


class BinaryCDEAC(BaseClassifyAndCountModel):
    def __init__(self, estimator_class=None, estimator_params=None, estimator_grid=None, grid_params=None, num_iter=3):
        super(BinaryCDEAC, self).__init__(estimator_class, estimator_params, estimator_grid, grid_params, b=None)
        self.num_iter = num_iter

    def fit(self, X, y, verbose=False):
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
        cm = model_score.cv_confusion_matrix(self.estimator_, X, y, 10, verbose)
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


class HDy(BaseCC):
    """
             Binary HDy method.

             It is just a wrapper of BaseClassifyAndCountModel (see :ref:`Classify and Count <cc_ref>`) to perform HDy.
             Although HDy is a distribution matching algorithm, it needs a classifier too.

        """

    def predict(self, X, method="hdy"):
        assert method == "hdy"
        return self._predict_hdy(X)


class EM(BaseClassifyAndCountModel):
    """
            Multiclass EM method.

            Parameters
            -----------
            tol : float, optional, default=1e-9
                Minimum error before stopping the learning process.

            estimator_class : object, optional
                An instance of a classifier class. It has to have fit and predict methods. It is highly advised to use one of
                the implementations in sklearn library. If it is leave as None, Logistic Regression will be used.

            estimator_params : dictionary, optional
                Additional params to initialize the classifier.

            estimator_grid : dictionary, optional
                During training phase, grid search is performed. This parameter should provided the parameters of the classifier
                that will be tested (e.g. estimator_grid={C: [0.1, 1, 10]} for Logistic Regression).

            Attributes
            ----------
            classes_ : array
                Unique classes in the training set.

            qnfs_ : array, shape = (n_samples)
            List of quantifiers to train. There is one for each class in the training set.

            """

    def __init__(self, estimator_class=None, estimator_params=None, estimator_grid=None, grid_params=None, tol=1e-9):
        super(EM, self).__init__(estimator_class, estimator_params, estimator_grid, grid_params, b=None)
        self.tol = tol

    def fit(self, X, y, verbose=False):
        self.classes_ = np.unique(y).tolist()
        n_classes = len(self.classes_)
        self.qnfs_ = dict.fromkeys(self.classes_)

        for pos_class in self.classes_:
            if verbose:
                print("Fitting classifier for class {}/{}".format(pos_class + 1, n_classes))
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


class CDEIter(BaseClassifyAndCountModel):
    """
        Multiclass CDE Iterate method.

        Parameters
        -----------
        num_iter : float, optional, default=3
            Number of iterations.

        estimator_class : object, optional
            An instance of a classifier class. It has to have fit and predict methods. It is highly advised to use one of
            the implementations in sklearn library. If it is leave as None, Logistic Regression will be used.

        estimator_params : dictionary, optional
            Additional params to initialize the classifier.

        estimator_grid : dictionary, optional
            During training phase, grid search is performed. This parameter should provided the parameters of the classifier
            that will be tested (e.g. estimator_grid={C: [0.1, 1, 10]} for Logistic Regression).

        Attributes
        ----------
        classes_ : array
            Unique classes in the training dataset.

        qnfs_ : array, shape = (n_samples)
            List of quantifiers to train. There is one for each class in the training set.

        """

    def __init__(self, estimator_class=None, estimator_params=None, estimator_grid=None, grid_params=None, num_iter=3):
        super(CDEIter, self).__init__(estimator_class, estimator_params, estimator_grid, grid_params, b=None)
        self.num_iter = num_iter

    def fit(self, X, y, verbose=False):
        self.classes_ = np.unique(y).tolist()
        n_classes = len(self.classes_)
        self.qnfs_ = dict.fromkeys(self.classes_)

        for pos_class in self.classes_:
            if verbose:
                print("Fitting classifier for class {}/{}".format(pos_class + 1, n_classes))
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


class CDEAC(BaseClassifyAndCountModel):
    def __init__(self, estimator_class=None, estimator_params=None, estimator_grid=None, grid_params=None, num_iter=3):
        super(CDEAC, self).__init__(estimator_class, estimator_params, estimator_grid, grid_params, b=None)
        self.num_iter = num_iter

    def fit(self, X, y, verbose=False):
        self.classes_ = np.unique(y).tolist()
        n_classes = len(self.classes_)
        self.qnfs_ = dict.fromkeys(self.classes_)

        for pos_class in self.classes_:
            if verbose:
                print("Fitting classifier for class {}/{}".format(pos_class + 1, n_classes))
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


class EDx(BaseCC):
    def __init__(self, estimator_class=None, estimator_params=None, estimator_grid=None, grid_params=None):
        super(EDx, self).__init__(estimator_class=estimator_class,
                                  estimator_params=estimator_params,
                                  estimator_grid=estimator_grid,
                                  grid_params=grid_params)

    def fit(self, X, y, local=True, cv=50, plot=False, verbose=False):
        super(EDx, self).fit(X, y)

        n_classes = len(self.classes_)
        self.K = np.zeros((n_classes, n_classes))
        self.X_train = X
        self.y_train = y

        for i in range(n_classes):
            self.K[i, i] = self.distance(X[y == self.classes_[i]],
                                         X[y == self.classes_[i]])
            for j in range(i + 1, n_classes):
                self.K[i, j] = self.distance(X[y == self.classes_[i]],
                                             X[y == self.classes_[j]])
                self.K[j, i] = self.K[i, j]

    def predict(self, X, method="edx"):
        assert method == "edx"
        n_classes = len(self.classes_)

        Kt = np.zeros(n_classes)

        for i in range(n_classes):
            Kt[i] = self.distance(self.X_train[self.y_train == self.classes_[i]], X)

        train_cls = np.array(np.bincount(self.y_train))[:, np.newaxis]
        train_cls_m = np.dot(train_cls, train_cls.T)
        m = float(len(X))

        K = self.K / train_cls_m
        Kt = Kt / (train_cls.squeeze() * m)
        B = np.zeros((n_classes - 1, n_classes - 1))
        for i in range(n_classes - 1):
            B[i, i] = - K[i, i] - K[-1, -1] + 2 * K[i, -1]
            for j in range(n_classes - 1):
                if j == i:
                    continue
                B[i, j] = - K[i, j] - K[-1, -1] + K[i, -1] + K[j, -1]

        t = - Kt[:-1] + K[:-1, -1] + Kt[-1] - K[-1, -1]

        G = 2 * B
        if not is_pd(G):
            G = nearest_pd(G)

        a = 2 * t
        C = np.vstack([- np.ones((1, n_classes - 1)), np.eye(n_classes - 1)]).T
        b = np.array([-1] + [0] * (n_classes - 1), dtype=np.float)
        sol = quadprog.solve_qp(G=G,
                                a=a, C=C, b=b)

        p = sol[0]
        p = np.append(p, 1 - p.sum())

        self.final_ed = 2 * p.dot(Kt) - p.T.dot(K).dot(p) - self.distance(X, X) / (m * m)

        return p

    def distance(self, p, q):
        return np.square(p[:, None] ** 2 - q ** 2).sum()

    def _compute_distribution(self, X, y):
        pass

    def _compute_performance(self, X, y, pos_class, folds, local, verbose):
        pass


class EDy(BaseCC):
    def fit(self, X, y, local=True, cv=50, plot=False, verbose=False):
        self.X_train = X
        self.y_train = y
        super(EDy, self).fit(X, y)

    def predict(self, X, method="edx"):
        n_classes = len(self.classes_)

        K = np.zeros((n_classes, n_classes))
        Kt = np.zeros(n_classes)

        repr_train = self.train_dist_
        if n_classes == 2:
            repr_test = self.estimators_[1].predict_proba(X)[:, 1]
        else:
            repr_test = np.zeros((len(X), n_classes))

            for n, clf in self.estimators_.items():
                repr_test[..., n] = clf.predict_proba(X)[..., 1]

        for i in range(n_classes):
            K[i, i] = self.l1_norm(repr_train[self.classes_[i]],
                                   repr_train[self.classes_[i]])
            Kt[i] = self.l1_norm(repr_train[self.classes_[i]], repr_test)
            for j in range(i + 1, n_classes):
                K[i, j] = self.l1_norm(repr_train[self.classes_[i]],
                                       repr_train[self.classes_[j]])
                K[j, i] = K[i, j]

        train_cls = np.array(np.bincount(self.y_train))[:, np.newaxis]
        train_cls_m = np.dot(train_cls, train_cls.T)
        m = float(len(X))

        K = K / train_cls_m
        Kt = Kt / (train_cls.squeeze() * m)
        B = np.zeros((n_classes - 1, n_classes - 1))
        for i in range(n_classes - 1):
            B[i, i] = - K[i, i] - K[-1, -1] + 2 * K[i, -1]
            for j in range(n_classes - 1):
                if j == i:
                    continue
                B[i, j] = - K[i, j] - K[-1, -1] + K[i, -1] + K[j, -1]

        t = - Kt[:-1] + K[:-1, -1] + Kt[-1] - K[-1, -1]

        G = 2 * B
        if not is_pd(G):
            G = nearest_pd(G)

        a = 2 * t
        C = np.vstack([np.ones((1, n_classes - 1)), np.eye(n_classes - 1)]).T
        b = np.array([1] + [0] * (n_classes - 1), dtype=np.float)
        sol = quadprog.solve_qp(G=G,
                                a=a, C=C, b=b)

        p = sol[0]
        p = np.append(p, 1 - p.sum())
        return p

    def l1_norm(self, p, q):
        return np.abs(p[:, None] - q).sum()

    def _compute_distribution(self, X, y):

        n_classes = len(self.classes_)
        self.train_dist_ = dict.fromkeys(self.classes_)
        if len(self.classes_) == 2:
            pos_preds = self.estimators_[1].predict_proba(X[y == 1])[:, 1]
            neg_preds = self.estimators_[1].predict_proba(X[y == 0])[:, 1]
            self.train_dist_[0] = neg_preds
            self.train_dist_[1] = pos_preds
        else:
            for n_cls, cls in enumerate(self.classes_):

                self.train_dist_[n_cls] = np.zeros((sum(y == cls), n_classes))

                for n_clf, (clf_cls, clf) in enumerate(self.estimators_.items()):
                    preds = clf.predict_proba(X[y == cls])[:, 1]
                    self.train_dist_[n_cls][:, n_clf] = preds

    def _compute_performance(self, X, y, pos_class, folds, local, verbose):
        pass


class kEDx(EDx):
    def __init__(self, k, estimator_class=None, estimator_params=None, estimator_grid=None, grid_params=None):
        self.k = k
        super(kEDx, self).__init__(estimator_class=estimator_class,
                                   estimator_params=estimator_params,
                                   estimator_grid=estimator_grid,
                                   grid_params=grid_params)

    def fit(self, X, y, local=True, cv=50, plot=False, verbose=False):
        classes = np.unique(y)
        self.orig_n_classes = len(classes)
        class_map = {}
        y_n = np.copy(y)

        for n, cls in enumerate(classes):
            kmeans = KMeans(n_clusters=self.k)
            kmeans.fit(X[y == cls])
            yn = kmeans.predict(X[y == cls])
            y_n[y == cls] = yn + (self.k * n)
            new_classes = np.arange(self.k) + (self.k * n)
            [class_map.update({str(nc): cls}) for nc in new_classes]

        self.class_map = class_map
        super(kEDx, self).fit(X, y_n)

    def predict(self, X, method="kedx"):
        assert method == "kedx"

        probs = super(kEDx, self).predict(X)

        prevalence = np.zeros(self.orig_n_classes)

        for n, p in enumerate(probs):
            prevalence[self.class_map[str(n)]] += p

        return prevalence


class CvMy(BaseCC):

    def __init__(self, estimator_class=None, estimator_params=None, estimator_grid=None, grid_params=None):
        super(CvMy, self).__init__(estimator_class=estimator_class,
                                   estimator_params=estimator_params,
                                   estimator_grid=estimator_grid,
                                   grid_params=grid_params)

    def fit(self, X, y, local=True, cv=50, plot=False, verbose=False):
        super(CvMy, self).fit(X, y)
        self.X_train = X
        self.y_train = y

    def predict(self, X, method="cvmx"):
        n_classes = len(self.classes_)

        train_repr = self.estimators_[1].predict_proba(self.X_train)[..., 1][:, np.newaxis]
        test_repr = self.estimators_[1].predict_proba(X)[..., 1][:, np.newaxis]

        Hn = rankdata(np.concatenate(np.concatenate([train_repr, test_repr])))
        Htr = Hn[:len(self.X_train)]
        Htst = Hn[len(self.X_train):]

        K = np.zeros((n_classes, n_classes))
        Kt = np.zeros(n_classes)

        for i in range(n_classes):
            K[i, i] = self.distance(Htr[self.y_train == self.classes_[i]],
                                    Htr[self.y_train == self.classes_[i]])

            Kt[i] = self.distance(Htr[self.y_train == self.classes_[i]], Htst)
            for j in range(i + 1, n_classes):
                K[i, j] = self.distance(Htr[self.y_train == self.classes_[i]],
                                        Htr[self.y_train == self.classes_[j]])
                K[j, i] = K[i, j]

        train_cls = np.array(np.bincount(self.y_train))[:, np.newaxis]
        train_cls_m = np.dot(train_cls, train_cls.T)
        m = float(len(X))

        K = K / train_cls_m
        Kt = Kt / (train_cls.squeeze() * m)
        B = np.zeros((n_classes - 1, n_classes - 1))
        for i in range(n_classes - 1):
            B[i, i] = - K[i, i] - K[-1, -1] + 2 * K[i, -1]
            for j in range(n_classes - 1):
                if j == i:
                    continue
                B[i, j] = - K[i, j] - K[-1, -1] + K[i, -1] + K[j, -1]

        t = - Kt[:-1] + K[:-1, -1] + Kt[-1] - K[-1, -1]

        G = 2 * B
        if not is_pd(G):
            G = nearest_pd(G)

        a = 2 * t
        C = np.vstack([- np.ones((1, n_classes - 1)), np.eye(n_classes - 1)]).T
        b = np.array([-1] + [0] * (n_classes - 1), dtype=np.float)
        sol = quadprog.solve_qp(G=G,
                                a=a, C=C, b=b)

        p = sol[0]
        p = np.append(p, 1 - p.sum())

        self.final_ed = 2 * p.dot(Kt) - p.T.dot(K).dot(p) - self.distance(X, X) / (m * m)

        return p

    def distance(self, p, q):
        return np.abs(p[:, None] - q).sum()

    def _compute_distribution(self, X, y):
        pass

    def _compute_performance(self, X, y, pos_class, folds, local, verbose):
        pass


class CvMX(BaseCC):
    def __init__(self, estimator_class=None, estimator_params=None, estimator_grid=None, grid_params=None):
        super(CvMX, self).__init__(estimator_class=estimator_class,
                                   estimator_params=estimator_params,
                                   estimator_grid=estimator_grid,
                                   grid_params=grid_params)

    def fit(self, X, y, local=True, cv=50, plot=False, verbose=False):
        super(CvMX, self).fit(X, y)
        self.X_train = X
        self.y_train = y

    def predict(self, X, method="cvmx"):
        n_classes = len(self.classes_)

        z = np.vstack(np.concatenate([self.X_train, X]))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            Htr = 1 / len(z) * np.nansum(
                (self.X_train[:, None] - z) / np.linalg.norm(self.X_train[:, None] - z, axis=-1, keepdims=True),
                axis=1)
            Htst = 1 / len(z) * np.nansum(
                (X[:, None] - z) / np.linalg.norm(X[:, None] - z, axis=-1, keepdims=True),
                axis=1)

        """
        Htr = np.zeros_like(self.X_train)
        Htst = np.zeros_like(X)

        for j in range(len(self.X_train)):
            for i in range(len(z)):
                r = (self.X_train[j] - z[i]) / np.linalg.norm(self.X_train[j] - z[i])
                if np.all(np.isnan(r)):
                    r = 0
                Htr[j] += r
        Htr /= len(z)

        for j in range(len(X)):
            for i in range(len(z)):
                r = (X[j] - z[i]) / np.linalg.norm(X[j] - z[i])
                if np.all(np.isnan(r)):
                    r = 0
                Htst[j] += r
        Htst /= len(z)
        """

        K = np.zeros((n_classes, n_classes))
        Kt = np.zeros(n_classes)

        for i in range(n_classes):
            K[i, i] = self.distance(Htr[self.y_train == self.classes_[i]],
                                    Htr[self.y_train == self.classes_[i]])

            Kt[i] = self.distance(Htr[self.y_train == self.classes_[i]], Htst)
            for j in range(i + 1, n_classes):
                K[i, j] = self.distance(Htr[self.y_train == self.classes_[i]],
                                        Htr[self.y_train == self.classes_[j]])
                K[j, i] = K[i, j]

        train_cls = np.array(np.bincount(self.y_train))[:, np.newaxis]
        train_cls_m = np.dot(train_cls, train_cls.T)
        m = float(len(X))

        K = K / train_cls_m
        Kt = Kt / (train_cls.squeeze() * m)
        B = np.zeros((n_classes - 1, n_classes - 1))
        for i in range(n_classes - 1):
            B[i, i] = - K[i, i] - K[-1, -1] + 2 * K[i, -1]
            for j in range(n_classes - 1):
                if j == i:
                    continue
                B[i, j] = - K[i, j] - K[-1, -1] + K[i, -1] + K[j, -1]

        t = - Kt[:-1] + K[:-1, -1] + Kt[-1] - K[-1, -1]

        G = 2 * B
        if not is_pd(G):
            G = nearest_pd(G)

        a = 2 * t
        C = np.vstack([- np.ones((1, n_classes - 1)), np.eye(n_classes - 1)]).T
        b = np.array([-1] + [0] * (n_classes - 1), dtype=np.float)
        sol = quadprog.solve_qp(G=G,
                                a=a, C=C, b=b)

        p = sol[0]
        p = np.append(p, 1 - p.sum())

        self.final_ed = 2 * p.dot(Kt) - p.T.dot(K).dot(p) - self.distance(X, X) / (m * m)

        return p

    def distance(self, p, q):
        return np.sqrt(np.sum(np.square(p[:, None] - q)))

    def _compute_distribution(self, X, y):
        pass

    def _compute_performance(self, X, y, pos_class, folds, local, verbose):
        pass


class MMy(BaseCC):

    def fit(self, X, y, cv=50, verbose=False, local=True):
        if len(np.unique(y)) != 2:
            raise AttributeError("This is a binary method, more than two clases are not yet supported")

        super(MMy, self).fit(X, y)

    def predict(self, X, method='cc'):

        test_repr = self.estimators_[1].predict_proba(X)[:, 1]
        counts, _ = np.histogram(test_repr, bins=self.b, range=(0., 1.))
        test_dist = np.cumsum(counts)

        test_dist = np.insert(test_dist, 0, 0)
        test_dist = test_dist / test_dist.max()

        ps = np.linspace(0., 1., 1001)
        areas = np.zeros_like(ps)
        l1 = np.zeros_like(ps)

        for i in range(len(ps)):
            Du = (1 - ps[i]) * self.train_dist_[0] + ps[i] * self.train_dist_[1]
            Du = np.insert(Du, 0, 0)

            dists = np.dstack((Du, test_dist)).squeeze()
            line = np.array([[0, 0], [1, 1]])
            polygon = np.concatenate([dists, line[::-1], dists[0:1]])
            areas[i] = Polygon(polygon).area

            l1[i] = np.abs(Du - test_dist).sum()

        import matplotlib.pyplot as plt

        p = ps[areas.argmin()]

        Du_opt = (1 - p) * self.train_dist_[0] + p * self.train_dist_[1]
        Du_opt = np.insert(Du_opt, 0, 0)
        Du_opt = Du_opt / Du_opt.max()
        plt.figure()
        plt.plot(Du_opt, test_dist)
        plt.plot([0, 1], [0, 1], color='k')
        plt.title("Optimum prevalence: {}".format(p))

        p_w = 0.1

        Du_w = (1 - p_w) * self.train_dist_[0] + p_w * self.train_dist_[1]
        Du_w = np.insert(Du_w, 0, 0)
        Du_w = Du_w / Du_w.max()
        plt.figure()
        plt.plot(Du_w, test_dist)
        plt.plot([0, 1], [0, 1], color='k')
        plt.title("Prevalence: {}".format(p_w))

        return np.array([1 - p, p])

    def _compute_distribution(self, X, y):

        pred = cross_val_predict(self.estimators_[1], X, y, cv=10, method='predict_proba')
        # pred = self.estimators_[1].predict_proba(X)

        self.train_dist_ = {0: None, 1: None}

        train_repr = pred[y == self.classes_[0]][..., 1]
        counts, _ = np.histogram(train_repr, bins=self.b, range=(0., 1.))
        self.train_dist_[0] = np.cumsum(counts) / (y == self.classes_[0]).sum()

        train_repr = pred[y == self.classes_[1]][..., 1]
        counts, _ = np.histogram(train_repr, bins=self.b, range=(0., 1.))
        self.train_dist_[1] = np.cumsum(counts) / (y == self.classes_[1]).sum()

    def _compute_performance(self, X, y, pos_class, folds, local, verbose):
        pass


class FriedmanBM(BaseCC):

    def predict(self, X, method='friedman-bm'):
        Af_ = self.estimators_[1].predict_proba(X)[:, 1]
        Af = np.mean(Af_ > self.train_prevs[1])

        p = (Af - self.An) / (self.Ap - self.An)
        return np.array([1 - p, p])

    def _compute_performance(self, X, y, pos_class, folds, local, verbose):
        pass

    def _compute_distribution(self, X, y):
        self.train_prevs = np.unique(y, return_counts=True)[1] / len(X)
        Ap_ = self.estimators_[1].predict_proba(X[y == self.classes_[1]])[:, 1]
        self.Ap = np.mean(Ap_ > self.train_prevs[1])
        An_ = self.estimators_[1].predict_proba(X[y == self.classes_[0]])[:, 1]
        self.An = np.mean(An_ > self.train_prevs[1])
        pass


class FriedmanMM(BaseCC):

    def predict(self, X, method='friedman-mm'):
        n_classes = len(self.classes_)

        Up = np.zeros((len(X), n_classes))
        if n_classes == 2:
            Up = (self.estimators_[1].predict_proba(X) >= self.train_prevs)
        else:
            for n_clf, (clf_cls, clf) in enumerate(self.estimators_.items()):
                Up[:, n_clf] = (clf.predict_proba(X)[:, 1] >= self.train_prevs[n_clf])

        U = Up.mean(axis=0)

        G = self.V.T.dot(self.V)
        if not is_pd(G):
            G = nearest_pd(G)
        a = U.dot(self.V)

        C = np.vstack([np.ones((1, n_classes)), np.eye(n_classes)]).T
        b = np.array([1] + [0] * n_classes, dtype=np.float)
        sol = quadprog.solve_qp(G=G,
                                a=a, C=C, b=b)

        p = sol[0]

        return p

    def _compute_distribution(self, X, y):
        n_classes = len(self.classes_)
        Vp = np.zeros((len(X), n_classes))
        self.train_prevs = np.unique(y, return_counts=True)[1] / len(X)

        if n_classes == 2:
            Vp = (self.estimators_[1].predict_proba(X) >= self.train_prevs)
        else:
            for n_clf, (clf_cls, clf) in enumerate(self.estimators_.items()):
                Vp[:, n_clf] = (clf.predict_proba(X)[:, 1] >= self.train_prevs[n_clf]).astype(np.int)

        self.V = np.zeros((n_classes, n_classes))

        for cls in self.classes_:
            self.V[:, cls] = Vp[y == cls].mean(axis=0)

    def _compute_performance(self, X, y, pos_class, folds, local, verbose):
        pass


class FriedmanDB(BaseCC):

    def predict(self, X, method='friedman-db'):
        Af = np.mean(self.estimators_[1].predict_proba(X)[:, 1])
        p = (Af - self.train_prevs[1]) / self.Vt + self.train_prevs[1]
        return np.array([1 - p, p])

    def _compute_distribution(self, X, y):
        self.train_prevs = np.unique(y, return_counts=True)[1] / len(X)
        At = np.mean((self.estimators_[1].predict_proba(X)[:, 1] - self.train_prevs[1]) ** 2)
        self.Vt = At / (self.train_prevs[1] * self.train_prevs[0])

    def _compute_performance(self, X, y, pos_class, folds, local, verbose):
        pass


class LSDD(six.with_metaclass(ABCMeta, BaseEstimator)):

    def __init__(self, sigma=0.01, lda=0.001, tol=1e-3, sampling=True):
        self.sigma = sigma
        self.lda = lda
        self.tol = tol
        self.sampling = sampling

    def fit(self, X, y):

        if len(np.unique(y)) != 2:
            raise AttributeError("This is a binary method, more than two clases are not yet supported")

        self.X_train = X
        self.y_train = y

    def predict(self, X):

        f = lambda p: self.lsdd(X, p)[0]
        a, b = gss(f, tol=self.tol)
        p = (a + b) / 2
        return np.array([1 - p, p])

    def lsdd(self, X, prev, folds=5):
        """
        least-squares density difference estimation
        Written by M.C. du Plessis

        Least-squares density difference estimation.

        Estimates p1(x) - p2(x) from samples {x1_i}_{i=1}^{n1} and {x2_j}_{j=1}^{n2}
        drawn i.i.d. from p1(x) and p2(x), respectively.

        """

        # get the sizes
        (m, d_test) = X.shape
        (n, d_train) = self.X_train.shape

        # check input argument
        T = np.vstack((X, self.X_train))

        # set the kernel bases
        X = np.vstack((X, self.X_train))
        if m + n >= 300 and self.sampling:  # This sampling cause the method to make a lot of mistakes
            tst_idxs = np.random.choice(len(X), int(m / (m + n) * 300), replace=False)
            pos_idxs = np.random.choice(np.sum(self.y_train == 1), int(np.sum(self.y_train == 1) / (m + n) * 300),
                                        replace=False)
            neg_idxs = np.random.choice(np.sum(self.y_train == 0), int(np.sum(self.y_train == 0) / (m + n) * 300),
                                        replace=False)
            C = np.vstack([X[tst_idxs],
                           self.X_train[self.y_train == 1][pos_idxs],
                           self.X_train[self.y_train == 0][neg_idxs]])
            b = len(C)
            C = C[np.random.permutation(b)]
        else:
            C = X
            b = m + n

        # calculate the squared distances
        test_c_dist = self.distance_squared(X, C)
        train_c_dist = self.distance_squared(self.X_train, C)
        t_c_dist = self.distance_squared(T, C)
        c_c_dist = self.distance_squared(C, C)

        # setup the cross validation
        cv_fold = np.arange(folds)  # normal range behaves strange with == sign
        cv_split_test = np.floor(np.arange(m) * folds / m)
        cv_split_train = np.floor(np.arange(n) * folds / n)
        cv_index_test = cv_split_test[np.random.permutation(m)]
        cv_index_train = cv_split_train[np.random.permutation(n)]
        n_test_cv = np.array([np.sum(cv_index_test == i) for i in cv_fold])
        n_train_pos_cv = np.array([np.sum(np.logical_and(cv_index_train == i, self.y_train == 1)) for i in cv_fold])
        n_train_neg_cv = np.array([np.sum(np.logical_and(cv_index_train == i, self.y_train == 0)) for i in cv_fold])

        # calculate the new solution

        sigma = self.sigma
        lda = self.lda

        H = (np.sqrt(np.pi) * sigma) ** d_train * np.exp(-c_c_dist / (4 * sigma ** 2))
        h = np.mean(np.exp(-test_c_dist / (2 * sigma ** 2)), axis=1) \
            - prev * np.mean(np.exp(-train_c_dist[:, self.y_train == 1] / (2 * sigma ** 2)), axis=1) \
            - (1 - prev) * np.mean(np.exp(-train_c_dist[:, self.y_train == 0] / (2 * sigma ** 2)), axis=1)

        alpha = np.linalg.solve(H + lda * np.eye(b), h)
        L2dist = 2 * np.dot(alpha, h) - np.dot(alpha, np.dot(H, alpha))

        # calculate the values a
        ddh = np.dot(alpha, np.exp(-t_c_dist / (2 * sigma ** 2)))

        return L2dist, ddh

    def distance_squared(self, X, C):
        """
        Calculates the squared distance between X and C.
        """
        Xsum = np.sum(X ** 2, axis=1)
        Csum = np.sum(C ** 2, axis=1)
        XC_dist2 = Xsum[np.newaxis, :] + Csum[:, np.newaxis] - 2 * np.dot(C, X.transpose())

        return XC_dist2
