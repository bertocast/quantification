import FukuML.Utility as utility
import numpy as np
from FukuML.KernelLogisticRegression import KernelLogisticRegression


class KLR(KernelLogisticRegression):
    def fit(self, X, y):

        self.train_X = X
        self.train_Y = y
        self.status = 'load_train_data'
        self.init_W()
        self.train()
        return self

    def predict_proba(self, X):
        preds = []
        for i in range(X.shape[0]):
            input_data_x = utility.DatasetLoader.feature_transform(
                X[i,].reshape(1, -1),
                self.feature_transform_mode,
                self.feature_transform_degree
            )
            input_data_x = np.ravel(input_data_x)
            preds.append(self.score_function(input_data_x, self.W))
        return np.array(preds)

    def predict(self, X):
        probs = self.predict_proba(X)
        preds = (probs > 0.5).astype(int)
        return preds

    def calculate_gradient(self, X, Y, beta):

        if type(Y) is np.ndarray:
            data_num = len(Y)
            original_X = X[:, :]
            K = utility.Kernel.kernel_matrix(self, original_X)
        else:
            data_num = 1
            original_x = X[1:]
            original_X = self.train_X[:, 1:]
            K = utility.Kernel.kernel_matrix_xX(self, original_x, original_X)

        gradient_average = ((2 * self.lambda_p) / data_num) * np.dot(beta, K) + \
                           np.dot(self.theta((-1) * Y * np.dot(beta, K)) * ((-1) * Y), K) / data_num

        return gradient_average

    def score_function(self, x, W):

        x = x[1:]
        original_X = self.train_X[:, :]

        score = np.sum(self.beta * utility.Kernel.kernel_matrix_xX(self, x, original_X))
        score = self.theta(score)

        return score



