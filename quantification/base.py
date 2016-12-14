from tempfile import mkstemp

import numpy as np


class BasicModel(object):
    def _persist_data(self, X, y):
        f, path = mkstemp()
        self.X_y_path_ = path + '.npz'
        np.savez(path, X=X, y=y)
