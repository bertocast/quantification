from copy import deepcopy
from os.path import basename

import dispy
import functools

import logging

from quantification.distribution_matching.base import BaseDistributionMatchingModel, BinaryHDy, MulticlassHDy

import numpy as np

from quantification.utils.errors import ClusterException


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
    def __init__(self, b, estimator_class=None, estimator_params=tuple()):
        super(MulticlassEnsembleHDy, self).__init__(b, estimator_class, estimator_params)
        self.classes_ = None
        self.qnfs_ = None

    def fit(self, X, y, verbose=False, plot=False, local=True):
        if len(X) != len(y):
            raise ValueError("X and y has to be the same length.")

        self.classes_ = np.unique(np.concatenate(y)).tolist()
        self.qnfs_ = [None for _ in y]

        if not local:
            self._persist_data(X, y)
            self.parallel_fit(X, y, verbose=verbose)
        else:
            for n, (X_sample, y_sample) in enumerate(zip(X, y)):
                if verbose:
                    print "\rProcessing sample {}/{}".format(n + 1, len(y)),
                qnf = self.fit_and_get_distributions(X_sample, y_sample, verbose)
                self.qnfs_[n] = deepcopy(qnf)

        return self

    def predict(self, X, plot=False, verbose=False, local=True):

        if not local:
            cls_prevalences = self._parallel_predict(X, verbose=verbose)

        else:
            cls_prevalences = {k: [] for k in self.classes_}
            for n, qnf in enumerate(self.qnfs_):
                print "\rPredicting by quantifier {}/{}".format(n, len(self.qnfs_)),
                probs = qnf.predict(X)
                for m, cls in enumerate(qnf.classes_):
                    cls_prevalences[cls].append(probs[m])
        for cls in cls_prevalences.keys():
            cls_prevalences[cls] = np.mean(cls_prevalences[cls])

        probs = np.array(cls_prevalences.values())
        return probs / np.sum(probs)

    def fit_and_get_distributions(self, X_sample, y_sample, verbose):
        qnf = MulticlassHDy(self.b, self.estimator_class, self.estimator_params)
        classes = np.unique(y_sample).tolist()
        n_classes = len(classes)
        qnf.classes_ = classes
        qnf.estimators_ = dict.fromkeys(classes)
        qnf.train_dist_ = dict.fromkeys(classes)
        for cls in classes:
            mask = (y_sample == cls)
            y_bin = np.ones(y_sample.shape, dtype=np.int)
            y_bin[~mask] = 0
            clf = qnf._make_estimator()
            if verbose:
                print "\rFitting classifier for class {}/{}".format(cls + 1, n_classes),
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
        return qnf

    def parallel_fit(self, X, y, verbose=False):
        def setup(data_file):
            global X, y
            import numpy as np
            with open(data_file, 'rb') as fh:
                data = np.load(fh)
                X = data['X']
                y = data['y']
            return 0

        def cleanup():
            global X, y
            del X, y

        def wrapper(qnf, n):
            X_sample = X[n]
            y_sample = y[n]
            return qnf.fit_and_get_distributions(X_sample, y_sample, True)

        cluster = dispy.SharedJobCluster(wrapper,
                                         depends=[self.X_y_path_],
                                         reentrant=True,
                                         setup=functools.partial(setup, basename(self.X_y_path_)),
                                         cleanup=cleanup,
                                         scheduler_node='dhcp015.aic.uniovi.es',
                                         loglevel=logging.ERROR)
        try:
            jobs = []
            for n in range(len(y)):
                job = cluster.submit(self, n)
                job.id = n
                jobs.append(job)
            for job in jobs:
                job()
                if job.exception:
                    raise ClusterException(job.exception + job.ip_addr)
                self.qnfs_[job.id] = deepcopy(job.result)
                if verbose:
                    print "\rSample {}/{} processed".format(job.id + 1, len(y)),
        except KeyboardInterrupt:
            cluster.close()
        if verbose:
            cluster.print_status()
        cluster.close()

    def _parallel_predict(self, X, verbose=False):
        def wrapper(qnf, X, classes):
            import numpy as np
            cls_prevalences = dict.fromkeys(classes)
            probs = qnf.predict(X)
            for m, cls in enumerate(qnf.classes_):
                cls_prevalences[cls] = probs[m]
            return cls_prevalences

        cluster = dispy.SharedJobCluster(wrapper,
                                         reentrant=True,
                                         scheduler_node='dhcp015.aic.uniovi.es',
                                         loglevel=logging.ERROR)
        prevalences = []
        try:
            jobs = []
            for n, qnf in enumerate(self.qnfs_):
                job = cluster.submit(qnf, X, self.classes_)
                job.id = n
                jobs.append(job)
            for job in jobs:
                job()
                if job.exception:
                    raise ClusterException(job.exception + job.ip_addr)
                prevalences.append(job.result)
                if verbose:
                    print "\rPredicted by quantifier {}/{}".format(job.id + 1, len(self.qnfs_)),
        except KeyboardInterrupt:
            cluster.close()
        if verbose:
            cluster.print_status()
        cluster.close()

        cls_prevalences = {k: [] for k in self.classes_}
        for d in prevalences:
            for k, v in d.iteritems():
                if v is not None:
                    cls_prevalences[k].append(v)
        return cls_prevalences
