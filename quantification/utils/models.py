import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_score
from sklearn.utils import check_X_y


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


class KLR(BaseEstimator, ClassifierMixin):
    def __init__(self, p=5, gamma=1, kernel='rbf'):
        self.p = p
        self.gamma = gamma
        self.kernel = kernel

    def fit(self, X, y):
        X, y = check_X_y(X, y, accept_sparse='csr', dtype=np.float64,
                         order="C")
        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape
        self.weights, self.bias = self._get_weights(n_features)
        Q = self._get_kernel(X)

        if isinstance(self.gamma, list):
            best_score = 0
            for g in self.gamma:
                self.beta, self.b = self.kernel_trick(Q, g, y)
                self.gamma = g
                score = np.mean(cross_val_score(self, X, y))
                if score > best_score:
                    best_score = score
                    best_beta = self.beta
                    best_b = self.b
            self.b = best_b
            self.beta = best_beta
        else:
            self.beta, self.b = self.kernel_trick(Q, self.gamma, y)
        return self

    def kernel_trick(self, Q, g, y):
        s = 4.0 / g
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
        return (prob[:, 1] > 0.5).astype(int)

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


if __name__ == '__main__':
    X = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
         [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
         [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
         [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
         [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
         [4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
         [4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
         [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
         [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
         [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
         [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
         [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
         [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
         [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
         [4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
         [4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
         [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
         [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
         [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
         ]
    X = np.array(X)
    y = np.array([1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1])

    klr = KLR(3, [10 ** i for i in range(-6,10)], 'rbf')
    klr.fit(X, y)
    print klr.gamma
    print klr.predict_proba(X)
