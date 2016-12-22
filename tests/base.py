from os.path import join, dirname

import numpy as np

from quantification.datasets.base import load_folder


class ModelTestCase:
    def setup(self):
        self.binary_data = load_folder(join(dirname(__file__), "../quantification/datasets/data/cancer"))
        for i in range(len(self.binary_data.target)):
            p = np.random.permutation(len(self.binary_data.target[i]))
            self.binary_data.data[i] = self.binary_data.data[i][p]
            self.binary_data.target[i] = self.binary_data.target[i][p]

        self.multiclass_data = load_folder(join(dirname(__file__), "../quantification/datasets/data/iris"))
        for i in range(len(self.multiclass_data.target)):
            p = np.random.permutation(len(self.multiclass_data.target[i]))
            self.multiclass_data.data[i] = self.multiclass_data.data[i][p]
            self.multiclass_data.target[i] = self.multiclass_data.target[i][p]
