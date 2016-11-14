import csv
from os import listdir
from os.path import join

import numpy as np

from sklearn.datasets.base import Bunch


def load_folder(path):
    total_data = []
    total_target = []
    for label, file in enumerate(listdir(path)):
        with open(join(path, file)) as csv_file:
            data_file = csv.reader(csv_file)
            temp = next(data_file)
            n_samples = int(temp[0])
            n_features = int(temp[1])
            target_names = np.array(temp[2:])
            data = np.empty((n_samples, n_features))
            target = np.empty((n_samples,), dtype=np.int)

            for i, ir in enumerate(data_file):
                data[i] = np.asarray(ir[:-1], dtype=np.float64)
                target[i] = np.asarray(ir[-1], dtype=np.int)
            total_data.append(data)
            total_target.append(target)

    return Bunch(data=total_data, target=total_target,
                 target_names=target_names)



