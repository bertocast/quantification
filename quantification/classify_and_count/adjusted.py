from base import BaseClassifyAndCountModel, ClassifyAndCount
from quantification.utils.validation import cross_validation_score


class AdjustedCount(BaseClassifyAndCountModel):

    def __init__(self, estimator_class=None, copy_X=True, estimator_params=tuple()):
        self.estimator_class = estimator_class
        self.copy_X = copy_X
        self.estimator_params = estimator_params

    def fit(self, X, y):

        cc = ClassifyAndCount(estimator_class=self.estimator_class, copy_X=self.copy_X,
                              estimator_params=self.estimator_params)
        cc = cc.fit(X, y)

        tpr, fpr, _ = self._performance(cc, y)

        return self


    def predict(self, X):
        pass

    def _performance(self, estimator, y, cv=3):
        return cross_validation_score(estimator, y, cv, score="roc")


if __name__ == '__main__':

    ac = AdjustedCount()
    from sklearn.datasets import load_iris

    X, y = load_iris(return_X_y=True)

    ac.fit(X, y)
