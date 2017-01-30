import functools
import logging
from copy import deepcopy
from os.path import basename

import dispy

from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.model_selection import GridSearchCV

from quantification.classify_and_count.base import BaseClassifyAndCountModel, BaseBinaryClassifyAndCount, \
    BaseMulticlassClassifyAndCount
from quantification.utils.errors import ClusterException


class BaseEnsembleCCModel(BaseClassifyAndCountModel):
    def __init__(self, estimator_class, estimator_params, estimator_grid, grid_params, b, strategy='macro'):
        super(BaseEnsembleCCModel, self).__init__(estimator_class, estimator_params, estimator_grid, grid_params, b)
        self.strategy = strategy


class EnsembleBinaryCC(BaseEnsembleCCModel):
    def __init__(self, estimator_class=None, estimator_params=None, estimator_grid=None, grid_params=None, b=None, strategy='macro'):
        super(EnsembleBinaryCC, self).__init__(estimator_class, estimator_params, estimator_grid, grid_params, b, strategy)

    def fit(self, X, y):

        self.qnfs_ = []

        if len(X) != len(y):
            raise ValueError("X and y has to be the same length.")
        for n, (X_sample, y_sample) in enumerate(zip(X, y)):
            if len(np.unique(y_sample)) < 2:
                continue
            qnf = BaseBinaryClassifyAndCount(b=self.b,
                                             estimator_class=self.estimator_class,
                                             estimator_params=self.estimator_params,
                                             estimator_grid=self.estimator_grid)
            try:
                qnf.estimator_.fit(X_sample, y_sample)
            except ValueError:  # Positive classes has not enough samples to perform cv
                continue
            if isinstance(qnf.estimator_, GridSearchCV):
                qnf.estimator_ = qnf.estimator_.best_estimator_
            qnf = self._performance(qnf, n, X, y)
            self.qnfs_.append(qnf)

        return self

    def _performance(self, qnf, n, X, y):
        X_val = np.concatenate([X[:n], X[(n + 1):]])
        y_val = np.concatenate([y[:n], y[(n + 1):]])
        cm = []
        for X_, y_ in zip(X_val, y_val):
            cm.append(confusion_matrix(y_, qnf.estimator_.predict(X_)))

        if self.strategy == 'micro':
            qnf.confusion_matrix_ = np.mean(cm, axis=0)
            qnf.tpr_ = qnf.confusion_matrix_[1, 1] / float(qnf.confusion_matrix_[1, 1] + qnf.confusion_matrix_[1, 0])
            qnf.fpr_ = qnf.confusion_matrix_[0, 1] / float(qnf.confusion_matrix_[0, 1] + qnf.confusion_matrix_[0, 0])
        elif self.strategy == 'macro':
            qnf.confusion_matrix_ = cm
            qnf.tpr_ = np.mean([cm_[1, 1] / float(cm_[1, 1] + cm_[1, 0]) for cm_ in cm])
            qnf.fpr_ = np.mean([cm_[0, 1] / float(cm_[0, 1] + cm_[0, 0]) for cm_ in cm])

        if np.isnan(qnf.tpr_):
            qnf.tpr_ = 0
        if np.isnan(qnf.fpr_):
            qnf.fpr_ = 0

        tp_pa, fp_pa, tn_pa, fn_pa = [], [], [], []
        for X_, y_ in zip(X_val, y_val):
            try:
                predictions = qnf.estimator_.predict_proba(X_)
            except AttributeError:
                return
            tp_pa.append(np.sum(predictions[y_ == qnf.estimator_.classes_[0], 0]) / \
                         np.sum(y_ == qnf.estimator_.classes_[0]))
            fp_pa.append(np.sum(predictions[y_ == qnf.estimator_.classes_[1], 0]) / \
                         np.sum(y_ == qnf.estimator_.classes_[1]))
            tn_pa.append(np.sum(predictions[y_ == qnf.estimator_.classes_[1], 1]) / \
                         np.sum(y_ == qnf.estimator_.classes_[1]))
            fn_pa.append(np.sum(predictions[y_ == qnf.estimator_.classes_[0], 1]) / \
                         np.sum(y_ == qnf.estimator_.classes_[0]))
        qnf.tp_pa_ = np.mean(tp_pa)
        qnf.fp_pa_ = np.mean(fp_pa)
        qnf.tn_pa_ = np.mean(tn_pa)
        qnf.fn_pa_ = np.mean(fn_pa)
        return qnf

    def predict(self, X, method='cc'):
        if method == 'cc':
            return self._predict_cc(X)
        elif method == 'ac':
            return self._predict_ac(X)
        elif method == 'pcc':
            return self._predict_pcc(X)
        elif method == 'pac':
            return self._predict_pac(X)
        elif method == 'hdy':
            return self._predict_hdy(X)
        else:
            raise ValueError("Invalid method %s. Choices are `cc`, `ac`, `pcc`, `pac`.", method)

    def _predict_cc(self, X):
        predictions = np.array([qnf.estimator_.predict(X) for qnf in self.qnfs_])
        maj = np.apply_along_axis(lambda x: np.argmax(np.bincount(x, minlength=2)),
                                  axis=0,
                                  arr=predictions.astype('int'))
        freq = np.bincount(maj, minlength=2)
        relative_freq = freq / float(np.sum(freq))
        return relative_freq

    def _predict_ac(self, X):
        predictions = np.full((len(self.qnfs_), 2), np.nan)
        for n, qnf in enumerate(self.qnfs_):
            probabilities = qnf._predict_cc(X)
            adjusted = (probabilities[1] - qnf.fpr_) / float(qnf.tpr_ - qnf.fpr_)
            predictions[n] = np.clip(np.array([1 - adjusted, adjusted]), 0, 1)
        predictions = np.mean(predictions, axis=0)
        return predictions / np.sum(predictions)

    def _predict_pcc(self, X):
        predictions = np.array([qnf.estimator_.predict_proba(X) for qnf in self.qnfs_])
        probabilities = np.mean(predictions, axis=1)
        maj = np.mean(probabilities, axis=0)
        return maj

    def _predict_pac(self, X):
        predictions = np.full((len(self.qnfs_), 2), np.nan)
        for n, qnf in enumerate(self.qnfs_):
            probabilities = qnf._predict_cc(X)
            pos = np.clip((probabilities[0] - qnf.fp_pa_) / float(qnf.tp_pa_ - qnf.fp_pa_), 0, 1)
            neg = np.clip((probabilities[1] - qnf.fn_pa_) / float(qnf.tn_pa_ - qnf.fn_pa_), 0, 1)
            predictions[n] = np.clip(np.array([pos, neg]), 0, 1)
        predictions = np.mean(predictions, axis=0)
        return predictions / np.sum(predictions)

    def _predict_hdy(self, X):
        if not self.b:
            raise ValueError("If HDy predictions are in order, the quantifier must be trained with the parameter `b`")
        predictions = np.array([qnf.predict(X) for qnf in self.qnfs_])
        return np.mean(predictions, axis=0)


class EnsembleMulticlassCC(BaseEnsembleCCModel):
    def __init__(self, estimator_class=None, estimator_params=None, estimator_grid=None, grid_params=None, b=None, strategy='macro'):
        super(EnsembleMulticlassCC, self).__init__(estimator_class, estimator_params, estimator_grid, grid_params, b, strategy)

    def fit(self, X, y, verbose=False, local=True):

        if len(X) != len(y):
            raise ValueError("X and y has to be the same length.")

        self.classes_ = np.unique(np.concatenate(y)).tolist()
        self.qnfs_ = [None for _ in y]
        self.cls_smp_ = {k: [] for k in self.classes_}

        if not local:
            self._persist_data(X, y)
            self._parallel_fit(X, y, verbose)
        else:
            for n, (X_sample, y_sample) in enumerate(zip(X, y)):
                qnf, classes = self._fit_and_get_distributions(X_sample, y_sample, verbose)
                for cls in classes:
                    self.cls_smp_[cls].append(n)
                self.qnfs_[n] = deepcopy(qnf)
                if verbose:
                    print "Sample {}/{} processed".format(n + 1, len(y))

        for pos_class in self.classes_:
            if verbose:
                print "Computing performance for classifiers of class {}".format(pos_class + 1)
            for sample in self.cls_smp_[pos_class]:
                self.qnfs_[sample] = self._performance(self.cls_smp_[pos_class], sample, pos_class, X, y)

        return self

    def _fit_and_get_distributions(self, X_sample, y_sample, verbose):
        # TODO: Refactor this to a dicts of classes and binary quantifiers
        qnf = BaseMulticlassClassifyAndCount(estimator_class=self.estimator_class,
                                             estimator_params=self.estimator_params,
                                             estimator_grid=self.estimator_grid)
        classes = np.unique(y_sample).tolist()
        invalid_classes = []
        qnf.estimators_ = dict()
        qnf.confusion_matrix_ = dict.fromkeys(classes)
        qnf.fpr_ = dict.fromkeys(self.classes_)
        qnf.tpr_ = dict.fromkeys(self.classes_)
        qnf.tp_pa_ = dict.fromkeys(classes)
        qnf.fp_pa_ = dict.fromkeys(classes)
        qnf.train_dist_ = dict.fromkeys(classes)
        for cls in classes:
            mask = (y_sample == cls)
            y_bin = np.ones(y_sample.shape, dtype=np.int)
            y_bin[~mask] = 0
            if len(np.unique(y_bin)) != 2 or np.any(np.bincount(y_bin) < 3):
                invalid_classes.append(cls)
                continue
            if verbose:
                print "\tFitting classifier for class {}".format(cls + 1)
            clf = qnf._make_estimator()
            clf = clf.fit(X_sample, y_bin)
            if isinstance(clf, GridSearchCV):
                clf = clf.best_estimator_
            qnf.estimators_[cls] = clf
            if self.b:
                if verbose:
                    print "\tComputing distribution for classifier of class {}".format(cls + 1)
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
        valid_classes = filter(lambda x: x not in invalid_classes, classes)
        qnf.classes_ = valid_classes
        return qnf, filter(lambda x: x not in invalid_classes, classes)

    def _parallel_fit(self, X, y, verbose):
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
            return qnf._fit_and_get_distributions(X_sample, y_sample, True)

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
                self.qnfs_[job.id], classes = deepcopy(job.result)
                for cls in classes:
                    self.cls_smp_[cls].append(job.id)


        except KeyboardInterrupt:
            cluster.close()
        if verbose:
            cluster.print_status()
        cluster.close()

    def _performance(self, samples, n, label, X, y):
        samples_val = samples[:n] + samples[(n + 1):]
        X_val = [X[i] for i in samples_val]
        y_val = [y[i] for i in samples_val]
        qnf = self.qnfs_[n]

        cm = []
        for X_, y_ in zip(X_val, y_val):
            mask = (y_ == label)
            y_bin = np.ones(y_.shape, dtype=np.int)
            y_bin[~mask] = 0
            cm.append(confusion_matrix(y_bin, qnf.estimators_[label].predict(X_)))

            try:
                predictions = qnf.estimators_[label].predict_proba(X_)
            except AttributeError:
                return

            qnf.tp_pa_[label] = np.sum(predictions[y_bin == qnf.estimators_[label].classes_[1], 1]) / \
                                np.sum(y_bin == qnf.estimators_[label].classes_[1])
            qnf.fp_pa_[label] = np.sum(predictions[y_bin == qnf.estimators_[label].classes_[0], 1]) / \
                                np.sum(y_bin == qnf.estimators_[label].classes_[0])

        if self.strategy == 'micro':
            qnf.confusion_matrix_[label] = np.mean(cm, axis=0)

            qnf.tpr_[label] = qnf.confusion_matrix_[label][1, 1] / float(qnf.confusion_matrix_[label][1, 1]
                                                                     + qnf.confusion_matrix_[label][1, 0])
            qnf.fpr_[label] = qnf.confusion_matrix_[label][0, 1] / float(qnf.confusion_matrix_[label][0, 1]
                                                                     + qnf.confusion_matrix_[label][0, 0])
        elif self.strategy == 'macro':
            qnf.confusion_matrix_[label] = cm
            qnf.tpr_[label] = np.mean([cm_[1, 1] / float(cm_[1, 1] + cm_[1, 0]) for cm_ in cm])
            qnf.fpr_[label] = np.mean([cm_[0, 1] / float(cm_[0, 1] + cm_[0, 0]) for cm_ in cm])

        if np.isnan(qnf.tpr_[label]):
            qnf.tpr_[label] = 0
        if np.isnan(qnf.fpr_[label]):
            qnf.fpr_[label] = 0

        return qnf

    def predict(self, X, method='cc'):
        if method == 'cc':
            return self._predict_cc(X)
        elif method == 'ac':
            return self._predict_ac(X)
        elif method == 'pcc':
            return self._predict_pcc(X)
        elif method == 'pac':
            return self._predict_pac(X)
        elif method == 'hdy':
            return self._predict_hdy(X)
        else:
            raise ValueError("Invalid method %s. Choices are `cc`, `ac`, `pcc`, `pac`.", method)

    def _predict_cc(self, X):
        probabilities = dict.fromkeys(self.classes_)
        predictions = {k: [] for k in self.classes_}
        for qnf in self.qnfs_:
            for cls, clf in qnf.estimators_.iteritems():
                pred = clf.predict(X)
                predictions[cls].append(pred)

        for cls, preds in predictions.iteritems():
            preds = np.array(preds)
            maj = np.apply_along_axis(lambda x: np.argmax(np.bincount(x, minlength=2)),
                                      axis=0,
                                      arr=preds.astype('int'))
            freq = np.bincount(maj, minlength=2)
            relative_freq = freq / float(np.sum(freq))
            probabilities[cls] = relative_freq[1]

        probs = np.array(probabilities.values())
        return probs / np.sum(probs)

    def _predict_ac(self, X):
        probabilities = dict.fromkeys(self.classes_)
        binary_freqs = {k: [] for k in self.classes_}
        for qnf in self.qnfs_:
            for cls, clf in qnf.estimators_.iteritems():
                pred = clf.predict(X)
                freq = np.bincount(pred, minlength=2)
                relative_freq = freq / float(np.sum(freq))

                adjusted = (relative_freq[1] - qnf.fpr_[cls]) / float(qnf.tpr_[cls] - qnf.fpr_[cls])
                binary_freqs[cls].append(np.clip(adjusted, 0, 1))

        for cls, freqs in binary_freqs.iteritems():
            freqs = np.array(freqs)
            probabilities[cls] = np.mean(freqs)
        probs = np.array(probabilities.values())
        return probs / np.sum(probs)

    def _predict_pcc(self, X):
        probabilities = dict.fromkeys(self.classes_)
        predictions = {k: [] for k in self.classes_}
        for qnf in self.qnfs_:
            for cls, clf in qnf.estimators_.iteritems():
                pred = clf.predict_proba(X)
                predictions[cls].append(pred)

        for cls, preds in predictions.iteritems():
            preds = np.array(preds)
            maj = np.mean(preds, axis=0)
            relative_freq = np.mean(maj, axis=0)
            probabilities[cls] = relative_freq[1]

        probs = np.array(probabilities.values())
        return probs / np.sum(probs)

    def _predict_pac(self, X):
        probabilities = dict.fromkeys(self.classes_)
        binary_freqs = {k: [] for k in self.classes_}
        for qnf in self.qnfs_:
            for (cls, clf) in qnf.estimators_.iteritems():
                pred = clf.predict_proba(X)
                relative_freq = np.mean(pred, axis=0)
                adjusted = (relative_freq[1] - qnf.fp_pa_[cls]) / float(qnf.tp_pa_[cls] - qnf.fp_pa_[cls])
                binary_freqs[cls].append(np.clip(adjusted, 0, 1))

        for cls, freqs in binary_freqs.iteritems():
            freqs = np.array(freqs)
            probabilities[cls] = np.mean(freqs)
        probs = np.array(probabilities.values())
        return probs / np.sum(probs)

    def _predict_hdy(self, X):

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
