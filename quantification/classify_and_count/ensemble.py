from sklearn.metrics import confusion_matrix

from quantification.classify_and_count.base import BaseClassifyAndCountModel, BinaryClassifyAndCount, \
    MulticlassClassifyAndCount
import numpy as np


class BaseEnsembleCCModel(BaseClassifyAndCountModel):
    def __init__(self, estimator_class=None, estimator_params=tuple()):
        super(BaseEnsembleCCModel, self).__init__(estimator_class, estimator_params)
        self.qnfs_ = None


class EnsembleBinaryCC(BaseEnsembleCCModel):
    def fit(self, X, y):
        if len(X) != len(y):
            raise ValueError("X and y has to be the same length.")
        self.qnfs_ = [None for _ in y]
        for n, (X_sample, y_sample) in enumerate(zip(X, y)):
            qnf = BinaryClassifyAndCount(self.estimator_class, self.estimator_params)
            qnf.estimator_.fit(X_sample, y_sample)
            qnf = self._performance(qnf, n, X, y)
            self.qnfs_[n] = qnf

        return self

    def _performance(self, qnf, n, X, y):
        X_val = np.concatenate(X[:n] + X[(n + 1):])
        y_val = np.concatenate(y[:n] + y[(n + 1):])
        qnf.confusion_matrix_ = confusion_matrix(y_val, qnf.estimator_.predict(X_val))

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
            tpr = qnf.confusion_matrix_[0, 0] / float(qnf.confusion_matrix_[0, 0] + qnf.confusion_matrix_[1, 0])
            fpr = qnf.confusion_matrix_[0, 1] / float(qnf.confusion_matrix_[0, 1] + qnf.confusion_matrix_[1, 1])
            adjusted = (probabilities - fpr) / float(tpr - fpr)
            predictions[n] = adjusted
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
    def fit(self, X, y):

        if len(X) != len(y):
            raise ValueError("X and y has to be the same length.")

        self.classes_ = np.unique(np.concatenate(y)).tolist()
        cls_smp = {k: [] for k in self.classes_}
        for n, (X_sample, y_sample) in enumerate(zip(X, y)):
            qnf = MulticlassClassifyAndCount(self.estimator_class, self.estimator_params)
            classes = np.unique(y_sample).tolist()
            qnf.estimators_ = dict.fromkeys(self.classes_)
            for pos_class in classes:
                mask = (y_sample == pos_class)
                y_bin = np.ones(y_sample.shape, dtype=np.int)
                y_bin[~mask] = 0
                clf = qnf._make_estimator()
                clf = clf.fit(X_sample, y_bin)
                qnf.estimators_[pos_class] = clf
                cls_smp[pos_class].append(n)

        for pos_class in self.classes_:
            for sample in cls_smp[pos_class]:
                self._perfomance(n_para_hacer_el_slice_and_concatenate_anterior)

        return self

    def predict(self, X, method):
        pass
