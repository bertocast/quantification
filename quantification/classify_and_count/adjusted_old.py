from copy import deepcopy
import pickle
from tempfile import NamedTemporaryFile
from os.path import basename
import numpy as np

from quantification.classify_and_count.base import BaseClassifyAndCountModel, predict_wrapper_per_clf, fit_wrapper
from quantification.utils.base import merge, mean_of_non_zero
from quantification.utils.parallelism import ClusterParallel
from quantification.utils.validation import split, cross_validation_score, create_partitions


# TODO: Make a common class and extend it

class BinaryAdjustedCount(BaseClassifyAndCountModel):
    def __init__(self):
        super(BinaryAdjustedCount, self).__init__()
        self.fpr_, self.tpr_ = None, None

    def _predict(self, X):
        parallel = ClusterParallel(predict_wrapper_per_clf, self.estimators_, {'X': X}, local=True)  # TODO: Fix this
        predictions = parallel.retrieve()
        maj = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)),
                                  axis=0,
                                  arr=predictions.astype('int'))
        freq = np.bincount(maj, minlength=len(self.classes_))
        relative_freq = freq / float(np.sum(freq))
        adjusted = self._adjust(relative_freq)
        return adjusted

    def fit(self, X, y, local=False):
        if not isinstance(X, list):
            clf = self._fit(X, y)
            self.tpr_, self.fpr_ = self._performance(X, y, clf, local)
            self.estimators_.append(clf)
        else:
            X, y = np.array(X), np.array(y)
            for sample in y:
                if len(np.unique(sample)) != 2:
                    raise ValueError('Number of classes must be 2 for a binary quantification problem')
            split_iter = split(X, len(X))
            if local:
                results = []
                for val, train in split_iter:
                    results.append(self.fit_and_performance(val, train, X, y, local=True))
            else:

                parallel = ClusterParallel(fit_and_performance_wrapper, split_iter, {'X': X, 'y': y,
                                                                                 'quantifier': self, 'local': local},
                                       local=True)  # TODO: Fix this
                results = parallel.retrieve()
            clfs, tpr, fpr = zip(*results)
            self.estimators_.extend(clfs)
            self.tpr_ = np.mean(tpr)
            self.fpr_ = np.mean(fpr)
        self.classes_ = set(label for clf in self.estimators_ for label in clf.classes_)
        return self

    def _performance(self, X, y, clf, local, cv=3):
        confusion_matrix = np.mean(
            cross_validation_score(clf, X, y, cv, score="confusion_matrix", local=local), 0)
        tpr = confusion_matrix[0, 0] / float(confusion_matrix[0, 0] + confusion_matrix[1, 0])
        fpr = confusion_matrix[0, 1] / float(confusion_matrix[0, 1] + confusion_matrix[1, 1])
        return tpr, fpr

    def fit_and_performance(self, perf, train, X, y, local):
        clf = self._fit(X[train[0]], y[train[0]])
        tpr, fpr = self._performance(np.concatenate(X[perf]), np.concatenate(y[perf]), clf, local)
        return clf, tpr, fpr

    def _adjust(self, prob):
        return (prob - self.fpr_) / float(self.tpr_ - self.fpr_)


class MulticlassAdjustedCount(BaseClassifyAndCountModel):
    def __init__(self):
        super(MulticlassAdjustedCount, self).__init__()
        self.clf_ = None
        self.estimators_ = None
        self.conditional_prob_ = None
        self.fpr_ = None
        self.tpr_ = None
        self.ensemble_ = False

    def _predict(self, X):
        if not self.ensemble_:
            predictions = self.clf_.predict(X)
            freq = np.bincount(predictions, minlength=len(self.classes_))
            relative_freq = freq / float(np.sum(freq))
            adjusted = np.linalg.solve(np.matrix.transpose(self.conditional_prob_), relative_freq)
            return adjusted

        class_freqs = []
        for cls in self.classes_:
            parallel = ClusterParallel(predict_wrapper_per_clf, self.estimators_[cls], {'X': X},
                                       local=True)  # TODO: Fix this
            predictions = parallel.retrieve()
            maj = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)),
                                      axis=0,
                                      arr=predictions.astype('int'))
            freq = np.bincount(maj, minlength=2)
            relative_freq = freq / float(np.sum(freq))
            adjusted = self._adjust(relative_freq, cls)
            class_freqs.append(adjusted[1])

        return class_freqs

    def fit(self, X, y, local=False):

        if not isinstance(X, list):  # Standard AC
            self.classes_ = np.unique(y).tolist()
            clf = self._fit(X, y)
            self.clf_ = clf
            self.conditional_prob_ = self._performance(X, y, clf, local)
        else:  # Ensemble AC
            self.ensemble_ = True
            self.classes_ = np.unique(np.concatenate(y)).tolist()
            self.estimators_ = dict.fromkeys(self.classes_)
            self.fpr_ = dict.fromkeys(self.classes_)
            self.tpr_ = dict.fromkeys(self.classes_)
            cls_smp = {k: [] for k in self.classes_}

            for n, sample in enumerate(y):
                for label in np.unique(sample):
                    cls_smp[label].append(n)

            for pos_class in self.classes_:
                X_samples, y_samples = np.array([X[s] for s in cls_smp[pos_class]]), \
                                       np.array([y[s] for s in cls_smp[pos_class]])
                split_iter = split(X_samples, len(X_samples))
                if local:
                    results = []
                    for val, train in split_iter:
                        results.append(self._fit_performance(X_samples[train[0]], y_samples[train[0]],
                                                             np.concatenate(X_samples[val]),
                                                             np.concatenate(y_samples[val]),
                                                             pos_class, local=True))
                    clfs, tpr, fpr = zip(*results)
                else:
                    f_x, f_y = NamedTemporaryFile(delete=False), NamedTemporaryFile(delete=False)
                    pickle.dump(X_samples, f_x)
                    pickle.dump(y_samples, f_y)
                    parallel = ClusterParallel(fit_ova_wrapper, split_iter,
                                           {'X_path': basename(f_x.name), 'y_path': basename(f_y.name),
                                            'quantifier': self, 'pos_class': pos_class, 'local': True},
                                           local=local, verbose=True, dependencies=[f_x.name, f_y.name])
                    f_x.close()
                    f_y.close()
                    clfs, tpr, fpr = zip(*parallel.retrieve())

                self.estimators_[pos_class] = clfs
                self.fpr_[pos_class] = np.mean(fpr)
                self.tpr_[pos_class] = np.mean(tpr)

        return self

    def _fit_performance(self, X_train, y_train, X_val, y_val, pos_class, local):
        mask = (y_train == pos_class)
        y_bin = np.ones(y_train.shape, dtype=np.float64)
        y_bin[~mask] = 0.
        clf = self._fit(X_train, y_bin)

        mask_val = (y_val == pos_class)
        y_bin_val = np.ones(y_val.shape, dtype=np.float64)
        y_bin_val[~mask_val] = 0.
        tpr, fpr = self._ensemble_performance(X_val, y_bin_val, clf, local)
        return clf, tpr, fpr

    def _performance(self, X, y, clf, local, cv=3):
        n_classes = len(self.classes_)
        confusion_matrix = np.mean(
            cross_validation_score(clf, X, y, cv, score="confusion_matrix", local=local, labels=self.classes_),
            0)
        conditional_prob = np.empty((n_classes, n_classes))
        for i in range(n_classes):
            if np.all(confusion_matrix[i] == 0.0):
                continue
            conditional_prob[i] = confusion_matrix[i] / np.sum(confusion_matrix[i])
        return conditional_prob

    def _ensemble_performance(self, X, y, clf, local, cv=3):
        confusion_matrix = np.mean(
            cross_validation_score(clf, X, y, cv, score="confusion_matrix", local=local), axis=0)
        tpr = confusion_matrix[1, 1] / float(confusion_matrix[1, 1] + confusion_matrix[1, 0])
        fpr = confusion_matrix[0, 1] / float(confusion_matrix[0, 1] + confusion_matrix[0, 0])
        return tpr, fpr

    def _adjust(self, prob, pos_class):
        adjusted = (prob - self.fpr_[pos_class]) / float(self.tpr_[pos_class] - self.fpr_[pos_class])
        return adjusted


def fit_and_performance_wrapper(val, train, X, y, quantifier, local):
    return quantifier._fit_performance(val, train, X, y, local)

def fit_ova_wrapper(val, train, X_path, y_path, quantifier, pos_class, local):
    import pickle, numpy as np
    with open(X_path, 'rb') as f_x:
        X = pickle.load(f_x)
    with open(y_path, 'rb') as f_y:
        y = pickle.load(f_y)
    return quantifier._fit_performance(X[train[0]], y[train[0]],
                                       np.concatenate(X[val]), np.concatenate(y[val]),
                                       pos_class, local)
