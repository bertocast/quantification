from copy import deepcopy

from quantification.distribution_matching.base import BaseDistributionMatchingModel, BinaryHDy, MulticlassHDy

import numpy as np


class BaseEnsembleDMModel(BaseDistributionMatchingModel):
    def __init__(self, b, estimator_class=None, estimator_params=tuple()):
        super(BaseEnsembleDMModel, self).__init__(estimator_class, estimator_params)
        self.b = b
        self.qnfs_ = None


class BinaryEnsembleHDy(BaseEnsembleDMModel):
    def fit(self, X, y, plot=False):
        if len(X) != len(y):
            raise ValueError("X and y has to be the same length.")
        self.qnfs_ = [None for _ in y]
        for n, (X_sample, y_sample) in enumerate(zip(X, y)):
            qnf = BinaryHDy(self.b, self.estimator_class, self.estimator_params)
            qnf.fit(X_sample, y_sample, plot=plot)
            self.qnfs_[n] = qnf

        return self

    def predict(self, X, plot=False):
        predictions = np.array([qnf.predict(X) for qnf in self.qnfs_])
        return np.mean(predictions, axis=0)


class MulticlassEnsembleHDy(BaseEnsembleDMModel):
    def __init__(self, b):
        super(MulticlassEnsembleHDy, self).__init__(b)
        self.classes_ = None
        self.qnfs_ = None

    def fit(self, X, y, verbose=False, plot=False):
        if len(X) != len(y):
            raise ValueError("X and y has to be the same length.")

        self.classes_ = np.unique(np.concatenate(y)).tolist()
        self.qnfs_ = [None for _ in y]

        for n, (X_sample, y_sample) in enumerate(zip(X, y)):
            if verbose:
                print "Processing sample {}/{}".format(n, len(y))
            qnf = MulticlassHDy(self.b, self.estimator_class, self.estimator_params)
            classes = np.unique(y_sample).tolist()
            n_classes = len(classes)
            qnf.classes_ = classes
            qnf.estimators_ = dict.fromkeys(classes)
            qnf.train_dist_ = dict.fromkeys(classes)
            for cls in classes:
                if verbose:
                    print "\tFitting classifier for class {}/{}".format(cls, n_classes)
                mask = (y == cls)
                y_bin = np.ones(y_sample.shape, dtype=np.int)
                y_bin[~mask] = 0
                clf = qnf._make_estimator()
                if verbose:
                    print "Fitting classifier for class {}/{}".format(cls + 1, n_classes)
                clf.fit(X_sample, y_bin)
                qnf.estimators_[cls] = clf
                pos_class = clf.classes_[1]
                neg_class = clf.classes_[0]
                pos_preds = clf.predict_proba(X_sample[y_bin == pos_class,])[:, 1]
                neg_preds = clf.predict_proba(X_sample[y_bin == neg_class,])[:, +1]

                train_pos_pdf, _ = np.histogram(pos_preds, self.b)
                train_neg_pdf, _ = np.histogram(neg_preds, self.b)
                qnf.train_dist_[cls] = np.full((self.b, 2), np.nan)
                for i in range(self.b):
                    qnf.train_dist_[cls][i] = [train_pos_pdf[i] / float(sum(y_bin == pos_class)),
                                                train_neg_pdf[i] / float(sum(y_bin == neg_class))]
            self.qnfs_[n] = deepcopy(qnf)

        return self

    def predict(self, X, plot=False):
        cls_prevalences = {k: [] for k in self.classes_}
        for qnf in self.qnfs_:
            probs = qnf.predict(X)
            for n, cls in enumerate(qnf.classes_):
                cls_prevalences[cls].append(probs[n])
        for cls in cls_prevalences.keys():
            cls_prevalences[cls] = np.mean(cls_prevalences[cls])

        probs = np.array(cls_prevalences.values())
        return probs / np.sum(probs)