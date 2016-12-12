from copy import deepcopy

from sklearn.metrics import confusion_matrix

from quantification.classify_and_count.base import BaseClassifyAndCountModel, BaseBinaryClassifyAndCount, \
    BaseMulticlassClassifyAndCount
import numpy as np


class BaseEnsembleCCModel(BaseClassifyAndCountModel):
    def __init__(self, estimator_class=None, estimator_params=dict(), estimator_grid=dict()):
        super(BaseEnsembleCCModel, self).__init__(estimator_class, estimator_params, estimator_grid)
        self.qnfs_ = []


class EnsembleBinaryCC(BaseEnsembleCCModel):
    def fit(self, X, y):
        if len(X) != len(y):
            raise ValueError("X and y has to be the same length.")
        for n, (X_sample, y_sample) in enumerate(zip(X, y)):
            if len(np.unique(y_sample)) != 2:
                continue
            qnf = BaseBinaryClassifyAndCount(self.estimator_class, self.estimator_params)
            try:
                qnf.estimator_.fit(X_sample, y_sample)
            except ValueError: # Positive classes has not enough samples to perform cv
                continue
            qnf.estimator_ = qnf.estimator_.best_estimator_
            qnf = self._performance(qnf, n, X, y)
            self.qnfs_.append(qnf)

        return self

    def _performance(self, qnf, n, X, y):
        X_val = np.concatenate(np.concatenate([X[:n], X[(n + 1):]]))
        y_val = np.concatenate(np.concatenate([y[:n], y[(n + 1):]]))
        qnf.confusion_matrix_ = confusion_matrix(y_val, qnf.estimator_.predict(X_val))

        qnf.tpr_ = qnf.confusion_matrix_[1, 1] / float(qnf.confusion_matrix_[1, 1] + qnf.confusion_matrix_[1, 0])
        qnf.fpr_ = qnf.confusion_matrix_[0, 1] / float(qnf.confusion_matrix_[0, 1] + qnf.confusion_matrix_[0, 0])

        if np.isnan(qnf.tpr_):
            qnf.tpr_ = 0
        if np.isnan(qnf.fpr_):
            qnf.fpr_ = 0

        try:
            predictions = qnf.estimator_.predict_proba(X_val)
        except AttributeError:
            return
        qnf.tp_pa_ = np.sum(predictions[y_val == qnf.estimator_.classes_[0], 0]) / \
                     np.sum(y_val == qnf.estimator_.classes_[0])
        qnf.fp_pa_ = np.sum(predictions[y_val == qnf.estimator_.classes_[1], 0]) / \
                     np.sum(y_val == qnf.estimator_.classes_[1])
        qnf.tn_pa_ = np.sum(predictions[y_val == qnf.estimator_.classes_[1], 1]) / \
                     np.sum(y_val == qnf.estimator_.classes_[1])
        qnf.fn_pa_ = np.sum(predictions[y_val == qnf.estimator_.classes_[0], 1]) / \
                     np.sum(y_val == qnf.estimator_.classes_[0])
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
            predictions[n] = np.array([1 - adjusted, adjusted])
        predictions = np.clip(np.mean(predictions, axis=0), 0, 1)
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
            predictions[n] = np.array([pos, neg])
        predictions = np.clip(np.mean(predictions, axis=0), 0, 1)
        return predictions / np.sum(predictions)


class EnsembleMulticlassCC(BaseEnsembleCCModel):
    def __init__(self, estimator_class=None, estimator_params=dict(), estimator_grid=dict()):
        super(EnsembleMulticlassCC, self).__init__(estimator_class, estimator_params, estimator_grid)
        self.classes_ = None
        self.qnfs_ = None

    def fit(self, X, y, verbose=False):

        if len(X) != len(y):
            raise ValueError("X and y has to be the same length.")

        self.classes_ = np.unique(np.concatenate(y)).tolist()
        self.qnfs_ = [None for _ in y]
        cls_smp = {k: [] for k in self.classes_}

        for n, (X_sample, y_sample) in enumerate(zip(X, y)):
            if verbose:
                print "\rProcessing sample {}/{}".format(n+1, len(y))
            qnf = BaseMulticlassClassifyAndCount(self.estimator_class, self.estimator_params)
            classes = np.unique(y_sample).tolist()
            n_classes = len(classes)
            qnf.classes_ = classes
            qnf.estimators_ = dict.fromkeys(classes)
            qnf.confusion_matrix_ = dict.fromkeys(classes)
            qnf.tp_pa_ = dict.fromkeys(classes)
            qnf.fp_pa_ = dict.fromkeys(classes)
            for pos_class in classes:
                if verbose:
                    print "\rFitting classifier for class {}".format(pos_class+1)
                mask = (y_sample == pos_class)
                y_bin = np.ones(y_sample.shape, dtype=np.int)
                y_bin[~mask] = 0
                if np.unique(y_bin) != 2:
                    continue
                clf = qnf._make_estimator()
                clf = clf.fit(X_sample, y_bin)
                clf = clf.best_estimator_
                qnf.estimators_[pos_class] = clf
                cls_smp[pos_class].append(n)
            self.qnfs_[n] = deepcopy(qnf)

        for pos_class in self.classes_:
            for sample in cls_smp[pos_class]:
                self.qnfs_[sample] = self._performance(cls_smp[pos_class], sample, pos_class, X, y)

        return self

    def _performance(self, samples, n, label, X, y):
        samples_val = samples[:n] + samples[(n + 1):]
        X_val = np.concatenate([X[i] for i in samples_val])
        y_val = np.concatenate([y[i] for i in samples_val])
        mask = (y_val == label)
        y_bin = np.ones(y_val.shape, dtype=np.int)
        y_bin[~mask] = 0

        qnf = self.qnfs_[n]
        qnf.confusion_matrix_[label] = confusion_matrix(y_bin, qnf.estimators_[label].predict(X_val))



        try:
            predictions = qnf.estimators_[label].predict_proba(X_val)
        except AttributeError:
            return

        qnf.tp_pa_[label] = np.sum(predictions[y_bin == qnf.estimators_[label].classes_[1], 1]) / \
                            np.sum(y_bin == qnf.estimators_[label].classes_[1])
        qnf.fp_pa_[label] = np.sum(predictions[y_bin == qnf.estimators_[label].classes_[0], 1]) / \
                            np.sum(y_bin == qnf.estimators_[label].classes_[0])

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
            for (cls, clf) in qnf.estimators_.iteritems():
                pred = clf.predict(X)
                freq = np.bincount(pred, minlength=2)
                relative_freq = freq / float(np.sum(freq))
                # TODO: Pasar esto al training
                tpr = qnf.confusion_matrix_[cls][1, 1] / float(qnf.confusion_matrix_[cls][1, 1]
                                                                + qnf.confusion_matrix_[cls][1, 0])
                fpr = qnf.confusion_matrix_[cls][0, 1] / float(qnf.confusion_matrix_[cls][0, 1]
                                                                + qnf.confusion_matrix_[cls][0, 0])
                if np.isnan(tpr):
                    tpr = 0
                if np.isnan(fpr):
                    fpr = 0

                adjusted = (relative_freq[1] - fpr) / float(tpr - fpr)
                binary_freqs[cls].append(np.clip(adjusted,0,1))

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
                binary_freqs[cls].append(np.clip(adjusted,0,1))

        for cls, freqs in binary_freqs.iteritems():
            freqs = np.array(freqs)
            probabilities[cls] = np.mean(freqs)
        probs = np.array(probabilities.values())
        return probs / np.sum(probs)


