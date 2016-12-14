from quantification.classify_and_count.ensemble import EnsembleMulticlassCC, EnsembleBinaryCC


class BinaryEnsembleHDy(EnsembleBinaryCC):
    def predict(self, X, method='hdy'):
        assert method == "hdy"
        return self._predict_hdy(X)


class MulticlassEnsembleHDy(EnsembleMulticlassCC):
    def predict(self, X, method='hdy'):
        assert method == "hdy"
        return self._predict_hdy(X)
