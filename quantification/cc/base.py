from abc import ABCMeta, abstractmethod

import numpy as np
from matplotlib import pyplot as plt
import six
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsOneClassifier
from copy import deepcopy
from quantification.base import BasicModel
from quantification.metrics import distributed, model_score


class BaseClassifyAndCountModel(six.with_metaclass(ABCMeta, BasicModel)):
    def __init__(self, estimator_class, estimator_params, estimator_grid, grid_params, b):
        if estimator_params is None:
            estimator_params = dict()
        if estimator_grid is None:
            estimator_grid = dict()
        if grid_params is None:
            grid_params = dict()
        self.b = b
        self.estimator_class = estimator_class
        self.estimator_params = estimator_params
        self.estimator_grid = estimator_grid
        self.grid_params = grid_params

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
        else:
            clf = default
        if self.estimator_params is not None:
            clf.set_params(**self.estimator_params)
            if not self.estimator_grid:
                estimator = clf
            else:
                estimator = GridSearchCV(estimator=self.estimator_class, param_grid=self.estimator_grid,
                                         **self.grid_params)
        else:
            clf.set_params(**default_params)
            if not self.estimator_grid:
                estimator = clf
            else:
                estimator = GridSearchCV(estimator=clf, param_grid=default_grid, verbose=11)

        if estimator is None:
            raise ValueError('estimator cannot be None')

        return estimator

    def _make_estimator(self):
        """Build the estimator"""
        estimator = self._validate_estimator(default=LogisticRegression(), default_grid={'C': [0.1, 1, 10]},
                                             default_params=dict())
        return estimator


class BaseBinaryCC(BaseClassifyAndCountModel):
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
    -----------
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

    def __init__(self, estimator_class=None, estimator_params=None, estimator_grid=None, grid_params=None, b=None,
                 strategy='macro'):
        super(BaseBinaryCC, self).__init__(estimator_class, estimator_params, estimator_grid, grid_params,
                                           b)
        self.estimator_ = self._make_estimator()
        self.strategy = strategy

    def fit(self, X, y, local=True, cv=50, plot=False, verbose=False):
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
        if isinstance(self.estimator_, GridSearchCV):
            self.estimator_ = self.estimator_.best_estimator_
        self._compute_performance(X, y, local=local, verbose=verbose, cv=cv)
        if self.b:
            if self.b == 'piramidal':
                self._compute_distribution_piramidal(X, y, plot=False)
            else:
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

    def _compute_performance(self, X, y, local, verbose, cv=50):
        """Compute the performance metrics, that is, confusion matrix, FPR, TPR and the probabilities averages
        and store them. The calculus of the confusion matrix can be parallelized if 'local' parameter is set to False.
        """
        if local:
            cm = model_score.cv_confusion_matrix(self.estimator_, X, y, folds=cv, verbose=verbose)
        else:
            cm = distributed.cv_confusion_matrix(self.estimator_, X, y, self.X_y_path_, folds=50)

        if self.strategy == 'micro':
            self.confusion_matrix_ = np.mean(cm, axis=0)
            self.tpr_ = self.confusion_matrix_[1, 1] / float(
                self.confusion_matrix_[1, 1] + self.confusion_matrix_[1, 0])
            self.fpr_ = self.confusion_matrix_[0, 1] / float(
                self.confusion_matrix_[0, 1] + self.confusion_matrix_[0, 0])
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
            plt.subplot(121)
            plt.hist(neg_preds, self.b)
            plt.title('Negative PDF')

            plt.figure(1)
            plt.subplot(122)
            plt.hist(pos_preds, self.b)
            plt.title('Positive PDF')

            plt.show()

    def _compute_distribution_piramidal(self, X, y, plot=False):
        """Compute the distributions in a piramidal fashion, that is, for each number of bins from 2 to 8, one histogram
        is calculated. Then, at predicting time, they are combined.
        """
        pos_class = self.estimator_.classes_[1]
        neg_class = self.estimator_.classes_[0]
        pos_preds = self.estimator_.predict_proba(X[y == pos_class,])[:, 1]
        neg_preds = self.estimator_.predict_proba(X[y == neg_class,])[:, +1]

        self.train_dist_ = {}
        for b in range(2, 9):
            train_pos_pdf, _ = np.histogram(pos_preds, b)
            train_neg_pdf, _ = np.histogram(neg_preds, b)
            self.train_dist_[b] = np.full((b, 2), np.nan)
            for i in range(b):
                self.train_dist_[b][i] = [train_pos_pdf[i] / float(sum(y == pos_class)),
                                          train_neg_pdf[i] / float(sum(y == neg_class))]

    def _predict_cc(self, X):
        """Compute the prevalence following the Classify and Count strategy"""
        predictions = self.estimator_.predict(X)
        freq = np.bincount(predictions, minlength=2)
        relative_freq = freq / float(np.sum(freq))
        return relative_freq[1]

    def _predict_ac(self, X):
        """Compute the prevalence following the Adjusted Count strategy"""
        prevalence = self._predict_cc(X)
        adjusted = np.clip((prevalence - self.fpr_) / float(self.tpr_ - self.fpr_), 0, 1)
        return adjusted

    def _predict_pcc(self, X):
        """Compute the prevalence following the Probabilistic Classify and Count strategy"""
        try:
            predictions = self.estimator_.predict_proba(X)
        except AttributeError:
            raise ValueError("Probabilistic methods like PCC or PAC cannot be used "
                             "with hard (crisp) classifiers like %s", self.estimator_.__class__.__name__)

        p = np.mean(predictions, axis=0)
        return p[1]

    def _predict_pac(self, X):
        """Compute the prevalence following the Probabilistic Adjusted Count strategy"""
        prevalence = self._predict_pcc(X)
        pos = np.clip((prevalence - self.fn_pa_) / float(self.tn_pa_ - self.fn_pa_), 0, 1)
        return pos

    def _predict_hdy(self, X, plot):
        """Compute the prevalence by applying HDy algorithm."""
        if self.b == 'piramidal':
            return self._predict_hdy_piramidal(X, False)

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
        return prevalence

    def _predict_hdy_piramidal(self, X, plot):
        """compute the prevalence by applying HDy algorithm using a piramidal estimationg of the distributions."""
        preds = self.estimator_.predict_proba(X)[:, 1]
        probas = [p / 100.0 for p in range(0, 100)]
        hd = np.full((len(probas), 7), np.nan)

        for b in range(2, 9):
            test_pdf, _ = np.histogram(preds, b)
            for p in range(len(probas)):
                diff = np.full(b, np.nan)
                for i in range(b):
                    di = np.sqrt(self.train_dist_[b][i, 0] * probas[p] + self.train_dist_[b][i, 1] * (1 - probas[p]))
                    ti = np.sqrt(test_pdf[i] / float(X.shape[0]))
                    diff[i] = np.power(di - ti, 2)
                hd[p, b - 2] = np.sqrt(np.sum(diff))
        hd = hd.mean(axis=1)
        p_min = hd.argmin()
        prevalence = probas[p_min]
        return np.array([1 - prevalence, prevalence])


class BaseMulticlassCC(BaseClassifyAndCountModel):
    """
        Multiclass Classify And Count method.

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

        strategy : string, optional
            Strategy to follow when aggregating.

        multiclass : string, optional
            One versus all or one vs one

        Attributes
        ----------
        estimator_class : object
            The underlying classifier+

        confusion_matrix_ : dictionary
            The confusion matrix estimated by cross-validation of the underlying classifier for each class

        tpr_ : dictionary
            True Positive Rate of the underlying classifier from the confusion matrix for each class

        fpr_ : dictionary
            False Positive Rate of the underlying classifier from the confusion matrix for each class

        tp_pa_ : dictionary
            True Positive Probability Average of the underlying classifier if it is probabilistic for each class

        fp_pa_ : dictionary
            False Positive Probability Average of the underlying classifier if it is probabilistic for each class

        train_dist_ : dictionary
            Distribution of the positive and negative samples for each bin in the training data for each class

        """

    def __init__(self, estimator_class=None, estimator_params=None, estimator_grid=None, grid_params=None, b=None,
                 strategy='macro', multiclass='ova'):
        super(BaseMulticlassCC, self).__init__(estimator_class, estimator_params, estimator_grid,
                                               grid_params, b)
        self.strategy = strategy
        self.multiclass = multiclass

    def fit(self, X, y, cv=50, verbose=False, local=True):
        self.classes_ = np.unique(y).tolist()
        n_classes = len(self.classes_)
        self.fpr_ = dict.fromkeys(self.classes_)
        self.tpr_ = dict.fromkeys(self.classes_)

        if not local:
            self._persist_data(X, y)

        if self.multiclass == 'ova':
            self.estimators_ = dict.fromkeys(self.classes_)
            self.confusion_matrix_ = dict.fromkeys(self.classes_)

            self.tp_pa_ = dict.fromkeys(self.classes_)
            self.fp_pa_ = dict.fromkeys(self.classes_)
            self.train_dist_ = dict.fromkeys(self.classes_)
            # TODO: Use sklearn's OneVsAllClassifier
            for pos_class in self.classes_:
                if verbose:
                    print "Class {}/{}".format(pos_class + 1, n_classes)
                    print "\tFitting  classifier..."
                mask = (y == pos_class)
                y_bin = np.ones(y.shape, dtype=np.int)
                y_bin[~mask] = 0
                clf = self._make_estimator()
                clf = clf.fit(X, y_bin)
                if isinstance(clf, GridSearchCV):
                    clf = clf.best_estimator_
                self.estimators_[pos_class] = deepcopy(clf)
                if verbose:
                    print "\tComputing performance..."
                self._compute_performance_ova(X, y_bin, pos_class, folds=cv, local=local, verbose=verbose)
                if self.b:
                    if verbose:
                        print "\tComputing distribution..."
                    self._compute_distribution_ova(clf, X, y_bin, pos_class)
        elif self.multiclass == 'ovo':
            clf = self._make_estimator()
            model = OneVsOneClassifier(clf)

            if verbose:
                print "Fitting classifiers..."
            model.fit(X, y)
            self.clf = model
            if verbose:
                print "Computing performance..."
            self._compute_performance_ovo(X, y, folds=cv, local=local, verbose=verbose)
            if self.b:
                if verbose:
                    print "\tComputing distribution..."
                self._compute_distribution_ovo(clf, X, y)

        return self

    def _compute_performance_ova(self, X, y, pos_class, folds, local, verbose):

        if local:
            cm = model_score.cv_confusion_matrix(self.estimators_[pos_class], X, y, folds, verbose)
        else:
            cm = distributed.cv_confusion_matrix(self.estimators_[pos_class], X, y, self.X_y_path_, pos_class=pos_class,
                                                 folds=folds,
                                                 verbose=verbose)

        if self.strategy == 'micro':
            self.confusion_matrix_[pos_class] = np.mean(cm, axis=0)
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

    def _compute_performance_ovo(self, X, y, folds, local, verbose):
        if local:
            cm = model_score.cv_confusion_matrix(self.clf, X, y, folds, verbose)
        else:
            cm = distributed.cv_confusion_matrix(self.clf, X, y, self.X_y_path_,
                                                 folds=folds,
                                                 verbose=verbose)
        if self.strategy == 'micro':
            self.confusion_matrix_ = np.mean(cm, axis=0)
            for n, pos_class in enumerate(self.classes_):
                self.tpr_[pos_class] = self.confusion_matrix_[n, n] / float(self.confusion_matrix_[:, n].sum())
                self.fpr_[pos_class] = np.delete(self.confusion_matrix_[:, n], n).sum() / float(
                    np.delete(self.confusion_matrix_, n, axis=0).sum())
        elif self.strategy == 'macro':
            self.confusion_matrix_ = cm
            for n, pos_class in enumerate(self.classes_):
                self.tpr_[pos_class] = np.mean([cm_[n, n] / float(cm_[:, n].sum()) for cm_ in cm])
                self.fpr_[pos_class] = np.mean(
                    [np.delete(cm_[:, n], n).sum() / float(np.delete(cm, n, axis=0).sum()) for cm_ in cm])

    def _compute_distribution_ova(self, clf, X, y_bin, cls):
        pos_class = clf.classes_[1]
        neg_class = clf.classes_[0]
        pos_preds = clf.predict_proba(X[y_bin == pos_class,])[:, 1]
        neg_preds = clf.predict_proba(X[y_bin == neg_class,])[:, 1]

        train_pos_pdf, _ = np.histogram(pos_preds, bins=self.b)
        train_neg_pdf, _ = np.histogram(neg_preds, bins=self.b)
        self.train_dist_[cls] = np.full((self.b, 2), np.nan)
        for i in range(self.b):
            self.train_dist_[cls][i] = [train_pos_pdf[i] / float(sum(y_bin == pos_class)),
                                        train_neg_pdf[i] / float(sum(y_bin == neg_class))]

    def _compute_distribution_ovo(self, X, y):
        raise NotImplementedError

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
        if self.multiclass == 'ovo':
            predictions = self.clf.predict(X)
            freq = np.bincount(predictions, minlength=len(self.classes_))
            probabilities = freq / float(np.sum(freq))
        else:
            for n, (cls, clf) in enumerate(self.estimators_.iteritems()):
                predictions = clf.predict(X)
                freq = np.bincount(predictions, minlength=2)
                relative_freq = freq / float(np.sum(freq))
                probabilities[n] = relative_freq[1]
        if np.sum(probabilities) == 0:
            return probabilities
        return probabilities / np.sum(probabilities)

    def _predict_ac(self, X):
        n_classes = len(self.classes_)
        probabilities = np.full(n_classes, np.nan)
        if self.multiclass == 'ovo':
            for n, clf in enumerate(self.clf.estimators_):
                predictions = clf.predict(X)
                freq = np.bincount(predictions, minlength=2)
                relative_freq = freq / float(np.sum(freq))
                adjusted = (relative_freq - self.fpr_[n]) / float(self.tpr_[n] - self.fpr_[n])
                adjusted = np.nan_to_num(adjusted)
                probabilities[n] = np.clip(adjusted[1], 0, 1)
        else:
            for n, (cls, clf) in enumerate(self.estimators_.iteritems()):
                predictions = clf.predict(X)
                freq = np.bincount(predictions, minlength=2)
                relative_freq = freq / float(np.sum(freq))
                adjusted = (relative_freq - self.fpr_[cls]) / float(self.tpr_[cls] - self.fpr_[cls])
                adjusted = np.nan_to_num(adjusted)
                probabilities[n] = np.clip(adjusted[1], 0, 1)
        if np.sum(probabilities) == 0:
            return probabilities
        return probabilities / np.sum(probabilities)

    def _predict_pcc(self, X):
        n_classes = len(self.classes_)
        probabilities = np.full(n_classes, np.nan)
        if self.multiclass == 'ovo':
            for n, clf in enumerate(self.clf.estimators_):
                try:
                    predictions = clf.predict_proba(X)
                except AttributeError:
                    raise ValueError("Probabilistic methods like PCC or PAC cannot be used "
                                     "with hard (crisp) classifiers like %s", clf.__class__.__name__)
                p = np.mean(predictions, axis=0)
                probabilities[n] = p[1]
        else:
            for n, (cls, clf) in enumerate(self.estimators_.iteritems()):
                try:
                    predictions = clf.predict_proba(X)
                except AttributeError:
                    raise ValueError("Probabilistic methods like PCC or PAC cannot be used "
                                     "with hard (crisp) classifiers like %s", clf.__class__.__name__)

                p = np.mean(predictions, axis=0)
                probabilities[n] = p[1]
        if np.sum(probabilities) == 0:
            return probabilities
        return probabilities / np.sum(probabilities)

    def _predict_pac(self, X):
        n_classes = len(self.classes_)
        probabilities = np.full(n_classes, np.nan)

        if self.multiclass == 'ovo':
            for n, clf in enumerate(self.clf.estimators_):
                try:
                    predictions = clf.predict_proba(X)
                except AttributeError:
                    raise ValueError("Probabilistic methods like PCC or PAC cannot be used "
                                     "with hard (crisp) classifiers like %s", clf.__class__.__name__)

                p = np.mean(predictions, axis=0)
                probabilities[n] = np.clip((p[1] - self.fp_pa_[n]) / float(self.tp_pa_[n] - self.fp_pa_[n]), 0, 1)
        else:
            for n, (cls, clf) in enumerate(self.estimators_.iteritems()):
                try:
                    predictions = clf.predict_proba(X)
                except AttributeError:
                    raise ValueError("Probabilistic methods like PCC or PAC cannot be used "
                                     "with hard (crisp) classifiers like %s", clf.__class__.__name__)

                p = np.mean(predictions, axis=0)
                probabilities[n] = np.clip((p[1] - self.fp_pa_[cls]) / float(self.tp_pa_[cls] - self.fp_pa_[cls]), 0, 1)

        if np.sum(probabilities) == 0:
            return probabilities
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
        if np.sum(probabilities) == 0:
            return probabilities
        return probabilities / np.sum(probabilities)


class BinaryCC(BaseBinaryCC):
    """
        Binary Classify And Count method.

        Just a wrapper to perform adjusted count without the need of every other single methods.
        The main difference with the general class is the `predict` method that enforces to use CC.

        """

    def predict(self, X, method='cc', plot=False):
        assert method == 'cc'
        return self._predict_cc(X)

    def _compute_performance(self, X, y, local, verbose, cv=50):
        self.confusion_matrix_ = np.full((2, 2), np.nan)
        self.tpr_ = np.nan
        self.fpr_ = np.nan

    def _compute_distribution(self, X, y, plot):
        self.train_dist_ = np.full((self.b, 2), np.nan)

    def _compute_distribution_piramidal(self, X, y, plot=False):
        pass


class BinaryAC(BaseBinaryCC):
    """
        Binary Adjusted Count method.

        Just a wrapper to perform adjusted count without the need of every other single methods.
        The main difference with the general class is the `predict` method that enforces to use AC.
        """

    def predict(self, X, method='ac', plot=False):
        assert method == 'ac'
        return self._predict_ac(X)


class BinaryPCC(BaseBinaryCC):
    """
        Binary Probabilistic Classify And Count method.

        Just a wrapper to perform adjusted count without the need of every other single methods.
        The main difference with the general class is the `predict` method that enforces to use PCC.
        Moreover, tihs class will not compute some parameters that are not needed and only will slow the process.
        """

    def predict(self, X, method='pcc', plot=False):
        assert method == 'pcc'
        return self._predict_pcc(X)

    def _compute_performance(self, X, y, local, verbose, cv=50):
        self.confusion_matrix_ = np.full((2, 2), np.nan)
        self.tpr_ = np.nan
        self.fpr_ = np.nan

    def _compute_distribution(self, X, y, plot):
        self.train_dist_ = np.full((self.b, 2), np.nan)

    def _compute_distribution_piramidal(self, X, y, plot=False):
        pass

class BinaryPAC(BaseBinaryCC):
    """
        Binary Probabilistic Adjusted Count method.

        Just a wrapper to perform adjusted count without the need of every other single methods.
        The main difference with the general class is the `predict` method that enforces to use PAC.
        """

    def predict(self, X, method='pac', plot=False):
        assert method == 'pac'
        return self._predict_pac(X)


class MulticlassCC(BaseMulticlassCC):
    """
        Multiclass Classify And Count method.

        Just a wrapper to perform adjusted count without the need of every other single methods.
        The main difference with the general class is the `predict` method that enforces to use CC.
        """

    def predict(self, X, method='cc'):
        assert method == 'cc'
        return self._predict_cc(X)


class MulticlassAC(BaseMulticlassCC):
    """
        Multiclass Adjusted Count method.

        Just a wrapper to perform adjusted count without the need of every other single methods.
        The main difference with the general class is the `predict` method that enforces to use AC.

        """

    def predict(self, X, method='ac'):
        assert method == 'ac'
        return self._predict_ac(X)


class MulticlassPCC(BaseMulticlassCC):
    """
        Multiclass Probabilistic Classify And Count method.

        Just a wrapper to perform adjusted count without the need of every other single methods.
        The main difference with the general class is the `predict` method that enforces to use PCC."""

    def predict(self, X, method='pcc'):
        assert method == 'pcc'
        return self._predict_pcc(X)


class MulticlassPAC(BaseMulticlassCC):
    """
        Multiclass Probabilistic Adjusted Count method.

        Just a wrapper to perform adjusted count without the need of every other single methods.
        The main difference with the general class is the `predict` method that enforces to use PAC.
        """

    def predict(self, X, method='pac'):
        assert method == 'pac'
        return self._predict_pac(X)
