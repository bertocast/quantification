import multiprocessing
from abc import ABCMeta, abstractmethod


import numpy as np
import six
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from quantification import BasicModel
from quantification.metrics import distributed, model_score


class BaseClassifyAndCountModel(six.with_metaclass(ABCMeta, BasicModel)):
    """Base class for C&C Models"""

    def __init__(self, estimator_class, estimator_params, estimator_grid, b):
        if estimator_params is None:
            estimator_params = dict()
        if estimator_grid is None:
            estimator_grid = dict()
        self.b = b
        self.estimator_class = estimator_class
        self.estimator_params = estimator_params
        self.estimator_grid = estimator_grid

    @abstractmethod
    def fit(self, X, y):
        """Fit a sample or a set of samples and combine them"""

    @abstractmethod
    def predict(self, X):
        """Predict the prevalence"""

    def _validate_estimator(self, default, default_params, default_grid):
        """Check the estimator."""
        if self.estimator_class is not None:
            clf = self.estimator_class
            clf.set_params(**self.estimator_params)
            estimator = GridSearchCV(estimator=self.estimator_class, param_grid=self.estimator_grid, verbose=True,
                                     n_jobs=multiprocessing.cpu_count())
        else:
            clf = default
            clf.set_params(**default_params)
            estimator = GridSearchCV(estimator=clf, param_grid=default_grid, verbose=True,
                                     n_jobs=multiprocessing.cpu_count())

        if estimator is None:
            raise ValueError('estimator cannot be None')

        return estimator

    def _make_estimator(self):
        estimator = self._validate_estimator(default=LogisticRegression(), default_grid={'C': [0.1, 1, 10]},
                                             default_params=dict())
        return estimator


class BaseBinaryClassifyAndCount(BaseClassifyAndCountModel):
    """
    Binary Classify And Count method.

    It is meant to be trained once and be able to predict using the following methods:
        - Classify & Count
        - Adjusted Count
        - Probabilistic Classify & Count
        - Probabilistic Adjusted Count
        - HDy

    The idea is not to trained the classifiers more than once, due to the computational cost. In the training phase
    every single performance metric that is needed in predictions are computed, that is, FPR, TPR and so on.

    If you are only going to use one of the methods above, you can use the wrapper classes in this package.

    Parameters
    ----------
    b : integer, optional
        Number of bins to compute the distributions in the HDy method. If you are not going to use that method in the
        prediction phase, leave it as None. The training phase will be probably faster.

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
        The underlying classifier+

    confusion_matrix_ : numpy array, shape = (2, 2)
        The confusion matrix estimated by cross-validation of the underlying classifier

    tpr_ : float
        True Positive Rate of the underlying classifier from the confusion matrix

    fpr_ : float
        False Positive Rate of the underlying classifier from the confusion matrix

    tp_pa_ : float
        True Positive Probability Average of the underlying classifier if it is probabilistic.

    fp_pa_ : float
        False Positive Probability Average of the underlying classifier if it is probabilistic.

    tn_pa_ : float
        True Negative Probability Average of the underlying classifier if it is probabilistic.

    fn_pa_ : float
        False Negative Probability Average of the underlying classifier if it is probabilistic.

    train_dist_ : numpy array, shape = (bins, 2)
        Distribution of the positive and negative samples for each bin in the training data

    """

    def __init__(self, estimator_class=None, estimator_params=None, estimator_grid=None, b=None, strategy='macro'):
        super(BaseBinaryClassifyAndCount, self).__init__(b, estimator_class, estimator_params, estimator_grid)
        self.estimator_ = self._make_estimator()
        self.strategy = strategy

    def fit(self, X, y, local=True, plot=False):
        """
        Fit C&C model.

        Parameters
        ----------
        X : numpy array, shape = (n_samples, n_features)
            Training data.

        y : numpy_array, shape = (n_samples, 1)
            Target values.

        local : boolean, optional, default True
            Whether or not do the fit in local or in a cluster using dispy.

        plot : boolean, optional, default False
            Whether or not plot the training distributions.


        Returns
        -------
        self : returns an instance of self.

        """
        n_classes = len(np.unique(y))
        if n_classes != 2:
            raise ValueError("This solver is meant for binary samples "
                             "thus number of classes must be 2, but the "
                             "data contains %s", n_classes)

        if not local:
            self._persist_data(X, y)

        self.estimator_.fit(X, y)
        self.estimator_ = self.estimator_.best_estimator_
        self._compute_performance(X, y, local=local)
        if self.b:
            self._compute_distribution(X, y, plot=plot)
        return self

    def predict(self, X, method='cc', plot=False):
        """Predict using one of the available methods.

        Parameters
        ---------
        X : numpy array, shape = (n_samples, n_features)
            Samples.

        method : string, optional, default cc
            Method to use in the prediction. It can be one of:
                - 'cc' : Classify & Count predictions
                - 'ac' : Adjusted Count predictions
                - 'pcc' : Probabilistic Classify & Count predictions
                - 'pcc' : Probabilistic Adjusted Count predictions
                - 'hdy' : HDy predictions

        plot : boolean, optional, default False
            If hdy predictions are in order, testing distributions can be plotted.


        Returns
        -------
        pred : numpy array, shape = (n_classes)
            Prevalences of each of the classes. Note that pred[0] = 1 - pred[1]

        """
        if method == 'cc':
            return self._predict_cc(X)
        elif method == 'ac':
            return self._predict_ac(X)
        elif method == 'pcc':
            return self._predict_pcc(X)
        elif method == 'pac':
            return self._predict_pac(X)
        elif method == "hdy":
            return self._predict_hdy(X, plot=plot)
        else:
            raise ValueError("Invalid method %s. Choices are `cc`, `ac`, `pcc`, `pac`.", method)

    def _compute_performance(self, X, y, local):
        """Compute the performance metrics, that is, confusion matrix, FPR, TPR and the probabilities averages
        and store them. The calculus of the confusion matrix can be parallelized if 'local' parameter is set to True.
        """
        if local:
            cm = model_score.cv_confusion_matrix(self.estimator_, X, y, 50)
        else:
            cm = distributed.cv_confusion_matrix(self.estimator_, X, y, self.X_y_path_, folds=50)

        if self.strategy == 'micro':
            self.confusion_matrix_ = np.mean(cm, axis=0)
            self.tpr_ = self.confusion_matrix_[1, 1] / float(self.confusion_matrix_[1, 1] + self.confusion_matrix_[1, 0])
            self.fpr_ = self.confusion_matrix_[0, 1] / float(self.confusion_matrix_[0, 1] + self.confusion_matrix_[0, 0])
        elif self.strategy == 'macro':
            self.confusion_matrix_ = cm
            self.tpr_ = np.mean([cm_[1, 1] / float(cm_[1, 1] + cm_[1, 0]) for cm_ in cm])
            self.fpr_ = np.mean([cm_[0, 1] / float(cm_[0, 1] + cm_[0, 0]) for cm_ in cm])
        if np.isnan(self.tpr_):
            self.tpr_ = 0
        if np.isnan(self.fpr_):
            self.fpr_ = 0

        try:
            predictions = self.estimator_.predict_proba(X)
        except AttributeError:
            return
        self.tp_pa_ = np.sum(predictions[y == self.estimator_.classes_[0], 0]) / \
                      np.sum(y == self.estimator_.classes_[0])
        self.fp_pa_ = np.sum(predictions[y == self.estimator_.classes_[1], 0]) / \
                      np.sum(y == self.estimator_.classes_[1])
        self.tn_pa_ = np.sum(predictions[y == self.estimator_.classes_[1], 1]) / \
                      np.sum(y == self.estimator_.classes_[1])
        self.fn_pa_ = np.sum(predictions[y == self.estimator_.classes_[0], 1]) / \
                      np.sum(y == self.estimator_.classes_[0])

    def _compute_distribution(self, X, y, plot):
        """Compute the distributions of each of the classes and store them. If plot is set to True, both histograms
        are plotted.
        """
        pos_class = self.estimator_.classes_[1]
        neg_class = self.estimator_.classes_[0]
        pos_preds = self.estimator_.predict_proba(X[y == pos_class,])[:, 1]
        neg_preds = self.estimator_.predict_proba(X[y == neg_class,])[:, +1]

        train_pos_pdf, _ = np.histogram(pos_preds, self.b)
        train_neg_pdf, _ = np.histogram(neg_preds, self.b)
        self.train_dist_ = np.full((self.b, 2), np.nan)
        for i in range(self.b):
            self.train_dist_[i] = [train_pos_pdf[i] / float(sum(y == pos_class)),
                                   train_neg_pdf[i] / float(sum(y == neg_class))]

        if plot:
            import matplotlib.pyplot as plt
            plt.subplot(121)
            plt.hist(neg_preds, self.b)
            plt.title('Negative PDF')

            plt.figure(1)
            plt.subplot(122)
            plt.hist(pos_preds, self.b)
            plt.title('Positive PDF')

            plt.show()

    def _predict_cc(self, X):
        predictions = self.estimator_.predict(X)
        freq = np.bincount(predictions, minlength=2)
        relative_freq = freq / float(np.sum(freq))
        return relative_freq

    def _predict_ac(self, X):
        probabilities = self._predict_cc(X)
        adjusted = np.clip((probabilities[1] - self.fpr_) / float(self.tpr_ - self.fpr_), 0, 1)
        return np.array([1 - adjusted, adjusted])

    def _predict_pcc(self, X):
        try:
            predictions = self.estimator_.predict_proba(X)
        except AttributeError:
            raise ValueError("Probabilistic methods like PCC or PAC cannot be used "
                             "with hard (crisp) classifiers like %s", self.estimator_.__class__.__name__)

        p = np.mean(predictions, axis=0)
        return np.array(p)

    def _predict_pac(self, X):
        predictions = self._predict_pcc(X)
        neg = np.clip((predictions[0] - self.fp_pa_) / float(self.tp_pa_ - self.fp_pa_), 0, 1)
        pos = np.clip((predictions[1] - self.fn_pa_) / float(self.tn_pa_ - self.fn_pa_), 0, 1)
        return np.array([neg, pos])

    def _predict_hdy(self, X, plot):
        if not self.b:
            raise ValueError("If HDy predictions are in order, the quantifier must be trained with the parameter `b`")
        preds = self.estimator_.predict_proba(X)[:, 1]
        test_pdf, _ = np.histogram(preds, self.b)

        if plot:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.hist(preds, self.b)
            plt.title('Test PDF')

            plt.show()

        probas = [p / 100.0 for p in range(0, 100)]
        hd = np.full(len(probas), np.nan)
        for p in range(len(probas)):
            diff = np.full(self.b, np.nan)
            for i in range(self.b):
                di = np.sqrt(self.train_dist_[i, 0] * probas[p] + self.train_dist_[i, 1] * (1 - probas[p]))
                ti = np.sqrt(test_pdf[i] / float(X.shape[0]))
                diff[i] = np.power(di - ti, 2)
            hd[p] = np.sqrt(np.sum(diff))

        p_min = np.argmin(hd)
        prevalence = probas[p_min]
        return np.array([1 - prevalence, prevalence])


class BaseMulticlassClassifyAndCount(BaseClassifyAndCountModel):
    def __init__(self, estimator_class=None, estimator_params=None, estimator_grid=None, b=None, strategy='macro'):
        super(BaseMulticlassClassifyAndCount, self).__init__(estimator_class, estimator_params, estimator_grid, b)
        self.strategy = strategy

    def fit(self, X, y, cv=50, verbose=False, local=True):
        self.classes_ = np.unique(y).tolist()
        n_classes = len(self.classes_)
        self.estimators_ = dict.fromkeys(self.classes_)
        self.confusion_matrix_ = dict.fromkeys(self.classes_)
        self.fpr_ = dict.fromkeys(self.classes_)
        self.tpr_ = dict.fromkeys(self.classes_)
        self.tp_pa_ = dict.fromkeys(self.classes_)
        self.fp_pa_ = dict.fromkeys(self.classes_)
        self.train_dist_ = dict.fromkeys(self.classes_)

        if not local:
            self._persist_data(X, y)

        for pos_class in self.classes_:
            if verbose:
                print "Fitting classifier for class {}/{}".format(pos_class + 1, n_classes)
            mask = (y == pos_class)
            y_bin = np.ones(y.shape, dtype=np.int)
            y_bin[~mask] = 0
            clf = self._make_estimator()
            clf = clf.fit(X, y_bin)
            clf = clf.best_estimator_
            self.estimators_[pos_class] = clf
            if verbose:
                print "Computing performance for classifier of class {}/{}".format(pos_class + 1, n_classes)
            self._compute_performance(X, y_bin, pos_class, folds=cv, local=local, verbose=verbose)
            if self.b:
                if verbose:
                    print "Computing distribution for classifier of class {}/{}".format(pos_class + 1, n_classes)
                self._compute_distribution(clf, X, y_bin, pos_class)

        return self

    def _compute_performance(self, X, y, pos_class, folds, local, verbose):

        if local:
            cm = model_score.cv_confusion_matrix(self.estimators_[pos_class], X, y, folds)
        else:
            cm = distributed.cv_confusion_matrix(self.estimators_[pos_class], X, y, self.X_y_path_, pos_class=pos_class, folds=folds,
                                                 verbose=verbose)

        if self.strategy == 'micro':
            self.confusion_matrix_[pos_class]= np.mean(cm, axis=0)
            self.tpr_[pos_class] = self.confusion_matrix_[pos_class][1, 1] / float(
                self.confusion_matrix_[pos_class][1, 1] + self.confusion_matrix_[pos_class][1, 0])
            self.fpr_[pos_class] = self.confusion_matrix_[pos_class][0, 1] / float(
                self.confusion_matrix_[pos_class][0, 1] + self.confusion_matrix_[pos_class][0, 0])
        elif self.strategy == 'macro':
            self.confusion_matrix_[pos_class] = cm
            self.tpr_[pos_class] = np.mean([cm_[1, 1] / float(cm_[1, 1] + cm_[1, 0]) for cm_ in cm])
            self.fpr_[pos_class] = np.mean([cm_[0, 1] / float(cm_[0, 1] + cm_[0, 0]) for cm_ in cm])

        try:
            predictions = self.estimators_[pos_class].predict_proba(X)
        except AttributeError:
            return

        self.tp_pa_[pos_class] = np.sum(predictions[y == self.estimators_[pos_class].classes_[1], 1]) / \
                                 np.sum(y == self.estimators_[pos_class].classes_[1])
        self.fp_pa_[pos_class] = np.sum(predictions[y == self.estimators_[pos_class].classes_[0], 1]) / \
                                 np.sum(y == self.estimators_[pos_class].classes_[0])

    def _compute_distribution(self, clf, X, y_bin, cls):
        pos_class = clf.classes_[1]
        neg_class = clf.classes_[0]
        pos_preds = clf.predict_proba(X[y_bin == pos_class,])[:, 1]
        neg_preds = clf.predict_proba(X[y_bin == neg_class,])[:, +1]

        train_pos_pdf, _ = np.histogram(pos_preds, self.b)
        train_neg_pdf, _ = np.histogram(neg_preds, self.b)
        self.train_dist_[cls] = np.full((self.b, 2), np.nan)
        for i in range(self.b):
            self.train_dist_[cls][i] = [train_pos_pdf[i] / float(sum(y_bin == pos_class)),
                                        train_neg_pdf[i] / float(sum(y_bin == neg_class))]

    def predict(self, X, method='cc'):
        if method == 'cc':
            return self._predict_cc(X)
        elif method == 'ac':
            return self._predict_ac(X)
        elif method == 'pcc':
            return self._predict_pcc(X)
        elif method == 'pac':
            return self._predict_pac(X)
        elif method == "hdy":
            return self._predict_hdy(X)
        else:
            raise ValueError("Invalid method %s. Choices are `cc`, `ac`, `pcc`, `pac`.", method)

    def _predict_cc(self, X):
        n_classes = len(self.classes_)
        probabilities = np.full(n_classes, np.nan)
        for n, (cls, clf) in enumerate(self.estimators_.iteritems()):
            predictions = clf.predict(X)
            freq = np.bincount(predictions, minlength=2)
            relative_freq = freq / float(np.sum(freq))
            probabilities[n] = relative_freq[1]
        return probabilities / np.sum(probabilities)

    def _predict_ac(self, X):
        n_classes = len(self.classes_)
        probabilities = np.full(n_classes, np.nan)
        for n, (cls, clf) in enumerate(self.estimators_.iteritems()):
            predictions = clf.predict(X)
            freq = np.bincount(predictions, minlength=2)
            relative_freq = freq / float(np.sum(freq))
            adjusted = (relative_freq - self.fpr_[cls]) / float(self.tpr_[cls] - self.fpr_[cls])
            probabilities[n] = np.clip(adjusted[1], 0, 1)
        return probabilities / np.sum(probabilities)

    def _predict_pcc(self, X):
        n_classes = len(self.classes_)
        probabilities = np.full(n_classes, np.nan)
        for n, (cls, clf) in enumerate(self.estimators_.iteritems()):
            try:
                predictions = clf.predict_proba(X)
            except AttributeError:
                raise ValueError("Probabilistic methods like PCC or PAC cannot be used "
                                 "with hard (crisp) classifiers like %s", clf.__class__.__name__)

            p = np.mean(predictions, axis=0)
            probabilities[n] = p[1]
        return probabilities / np.sum(probabilities)

    def _predict_pac(self, X):
        n_classes = len(self.classes_)
        probabilities = np.full(n_classes, np.nan)
        for n, (cls, clf) in enumerate(self.estimators_.iteritems()):
            try:
                predictions = clf.predict_proba(X)
            except AttributeError:
                raise ValueError("Probabilistic methods like PCC or PAC cannot be used "
                                 "with hard (crisp) classifiers like %s", clf.__class__.__name__)

            p = np.mean(predictions, axis=0)
            probabilities[n] = np.clip((p[1] - self.fp_pa_[cls]) / float(self.tp_pa_[cls] - self.fp_pa_[cls]), 0, 1)

        return probabilities / np.sum(probabilities)

    def _predict_hdy(self, X):
        if not self.b:
            raise ValueError("If HDy predictions are in order, the quantifier must be trained with the parameter `b`")
        n_classes = len(self.classes_)
        probabilities = np.full(n_classes, np.nan)
        for n, (cls, clf) in enumerate(self.estimators_.iteritems()):
            preds = clf.predict_proba(X)[:, 1]
            test_pdf, _ = np.histogram(preds, self.b)
            probas = [p / 100.0 for p in range(0, 100)]
            hd = np.full(len(probas), np.nan)
            for p in range(len(probas)):
                diff = np.full(self.b, np.nan)
                for i in range(self.b):
                    di = np.sqrt(
                        self.train_dist_[cls][i, 0] * probas[p] + self.train_dist_[cls][i, 1] * (1 - probas[p]))
                    ti = np.sqrt(test_pdf[i] / float(X.shape[0]))
                    diff[i] = np.power(di - ti, 2)
                hd[p] = np.sqrt(np.sum(diff))

            p_min = np.argmin(hd)
            probabilities[n] = probas[p_min]

        return probabilities / np.sum(probabilities)


class BinaryClassifyAndCount(BaseBinaryClassifyAndCount):
    def predict(self, X, method='cc', plot=False):
        assert method == 'cc'
        return self._predict_cc(X)


class BinaryAdjustedCount(BaseBinaryClassifyAndCount):
    def predict(self, X, method='ac', plot=False):
        assert method == 'ac'
        return self._predict_ac(X)


class BinaryProbabilisticCC(BaseBinaryClassifyAndCount):
    def predict(self, X, method='pcc', plot=False):
        assert method == 'pcc'
        return self._predict_pcc(X)


class BinaryProbabilisticAC(BaseBinaryClassifyAndCount):
    def predict(self, X, method='pac', plot=False):
        assert method == 'pac'
        return self._predict_pac(X)


class MulticlassClassifyAndCount(BaseMulticlassClassifyAndCount):
    def predict(self, X, method='cc'):
        assert method == 'cc'
        return self._predict_cc(X)


class MulticlassAdjustedCount(BaseMulticlassClassifyAndCount):
    def predict(self, X, method='ac'):
        assert method == 'ac'
        return self._predict_ac(X)


class MulticlassProbabilisticCC(BaseMulticlassClassifyAndCount):
    def predict(self, X, method='pcc'):
        assert method == 'pcc'
        return self._predict_pcc(X)


class MulticlassProbabilisticAC(BaseMulticlassClassifyAndCount):
    def predict(self, X, method='pac'):
        assert method == 'pac'
        return self._predict_pac(X)
