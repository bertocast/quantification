from copy import deepcopy

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_score
from sklearn.utils import check_X_y
import multiprocessing

from quantification.classify_and_count.base import BaseMulticlassClassifyAndCount


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


class KLR(BaseEstimator, ClassifierMixin):
    def __init__(self, p=5, gamma=1, kernel='rbf', pos_class_weight=None):
        self.p = p
        self.gamma = gamma
        self.kernel = kernel
        self.pos_class_weight = pos_class_weight

    def fit(self, X, y):
        X, y = check_X_y(X, y, accept_sparse='csr', dtype=np.float64,
                         order="C")
        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape

        counts = np.bincount(y)
        self.prior_ = counts[1] / float(len(y))

        self.weights, self.bias = self._get_weights(n_features)
        Q = self._get_kernel(X)

        pos_class_weight = self.pos_class_weight
        if pos_class_weight == 'balanced':
            counts = np.bincount(y)
            pos_class_weight = counts[0] / float(counts[1])

        y_ = deepcopy(y)
        y_[y_==self.classes_[1]] = y_[y_==self.classes_[1]] * pos_class_weight

        self.beta, self.b = self.kernel_trick(Q, y_)
        return self

    def kernel_trick(self, Q, y):
        s = 4.0 / self.gamma
        f = np.dot(4.0 * Q.T, y - 0.5)
        B = np.sum(Q, axis=0)
        V = np.dot(Q.T, Q)
        C = np.dot(Q, np.linalg.pinv(V))
        C = np.sum(C, axis=0)
        V[np.diag_indices_from(V)] += s
        V_inv = np.linalg.pinv(V)
        Z = np.dot(V_inv, f)
        S_i = 1 / np.dot(np.dot(C, V_inv).T, B)
        b = np.dot(S_i * C, Z)
        beta = Z - np.dot(V_inv, B) * b

        return beta, b

    def predict_proba(self, X):
        Q = self._get_kernel(X)
        p = np.dot(Q, self.beta) + self.b
        prob = sigmoid(p)
        return np.column_stack((1 - prob, prob))

    def predict(self, X):
        prob = self.predict_proba(X)
        preds = (prob[:, 1] > self.prior_).astype(int)
        preds[preds==0] = self.classes_[0]
        preds[preds==1] = self.classes_[1]
        return preds

    def _get_weights(self, n_feat):
        if self.kernel in ['sigmoid', 'HypTan', 'Fourier', 'HardLimit']:
            w = np.random.uniform(-1 / np.sqrt(n_feat), 1 / np.sqrt(n_feat), (self.p, n_feat))
            b = np.apply_along_axis(lambda x: np.random.uniform(min(x), abs(max(x))), 1, w)
        elif self.kernel in ['rbf', 'MultiQuad', 'hybrid']:
            w = np.random.uniform(-1, 1, (self.p, n_feat))
            b = np.random.uniform(-1, 1, self.p)

        return w, b

    def _get_kernel(self, X):

        if self.kernel == 'sigmoid':
            W = np.column_stack((self.weights, self.bias))
            X_ = np.column_stack((X, np.ones(X.shape[0])))
            Q = np.dot(X_, W.T)
            Q = sigmoid(Q)
            return Q

        elif self.kernel == 'rbf':
            Q = np.empty((X.shape[0], self.p))
            for i in range(self.p):
                Q[:, i] = np.apply_along_axis(lambda x: self.bias[i] * sum((x - self.weights[i,]) ** 2), 1, X)
            return Q

    def score(self, X, y, sample_weight=None):
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(y, self.predict(X), sample_weight=sample_weight)

        fpr = cm[0, 1] / float(cm[0, 1] + cm[0, 0])
        tpr = cm[1, 1] / float(cm[1, 1] + cm[1, 0])

        return np.sqrt((1 - fpr) * tpr)


if __name__ == '__main__':
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
    from sklearn.datasets.base import Bunch
    def load_plankton_file(path, sample_col="Sample", target_col="class"):
        data_file = pd.read_csv(path, delimiter=' ')
        le = LabelEncoder()
        data_file[target_col] = le.fit_transform(data_file[target_col])
        data = data_file.groupby(sample_col)
        target = [sample[1].values for sample in data[target_col]]
        features = [sample[1].drop([sample_col, target_col], axis=1, inplace=False).values for sample in data]
        return Bunch(data=features, target=target,
                     target_names=le.classes_), le

    from sklearn.datasets import load_breast_cancer
    from sklearn.metrics import confusion_matrix

    pl, le = load_plankton_file('../../examples/plancton.csv')
    X = np.concatenate(pl.data)
    y = np.concatenate(pl.target)


    pos_ind = np.where(y==1)[0]
    neg_ind = np.where(y==0)[0]
    pos_ind = pos_ind[:20]
    ind = np.concatenate((pos_ind, neg_ind))
    X = X[ind, :]
    y = y[ind]

    klr = KLR(p=30)
    klr.fit(X, y)
    print confusion_matrix(y_true=y, y_pred=klr.predict(X))