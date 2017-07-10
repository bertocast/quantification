from os.path import basename
from tempfile import mkstemp

import dispy
import functools

import logging
import pandas as pd
import numpy as np
from sklearn.datasets.base import Bunch
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

from quantification.cc.base import BaseMulticlassCC
from quantification.metrics.multiclass import bray_curtis, absolute_error
from quantification.utils.errors import ClusterException


def load_plankton_file(path, sample_col="Sample", target_col="class"):
    data_file = pd.read_csv(path, delimiter=' ')
    le = LabelEncoder()
    data_file[target_col] = le.fit_transform(data_file[target_col])
    data = data_file.groupby(sample_col)
    target = [sample[1].values for sample in data[target_col]]
    features = [sample[1].drop([sample_col, target_col], axis=1, inplace=False).values for sample in data]
    return Bunch(data=features, target=target,
                 target_names=le.classes_), le

def g_mean(estimator, X, y):
    from sklearn.metrics import confusion_matrix
    import numpy as np

    cm = confusion_matrix(y, estimator.predict(X), labels=estimator.classes_)

    fpr = cm[0, 1] / float(cm[0, 1] + cm[0, 0])
    tpr = cm[1, 1] / float(cm[1, 1] + cm[1, 0])

    return np.sqrt((1 - fpr) * tpr)

def print_and_write(text):
    global file
    print text
    file.write(text + "\n")

def cc(X, y):

    f, path = mkstemp()
    X_y_path = path + '.npz'
    np.savez(path, X=X, y=y)
    loo = LeaveOneOut()


    def setup(data_file):
        global X, y
        import numpy as np
        with open(data_file, 'rb') as fh:
            data = np.load(fh)
            X = data['X']
            y = data['y']
        return 0

    def cleanup():
        global X, y
        del X, y

    def train_fold(train_index, test_index):
        import numpy as np
        from quantification.cc.base import BaseMulticlassCC
        from sklearn.svm import SVC
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        X_test = X_test[0]
        y_test = y_test[0]

        X_train = np.concatenate(X_train)
        y_train = np.concatenate(y_train)

        grid = dict(C=[10 ** i for i in xrange(1, 4)], gamma=[10 ** i for i in xrange(-6, 0)])

        cc = BaseMulticlassCC(b=8,
                              estimator_class=SVC(class_weight='balanced', kernel='rbf', probability=True,
                                                                tol=0.1),
                              estimator_grid=grid,
                              grid_params=dict(scoring=g_mean, verbose=11),
                              strategy='macro')

        cc.fit(X_train, y_train, local=True, verbose=True)

        pred_cc = cc.predict(X_test, method='cc')
        pred_ac = cc.predict(X_test, method='ac')
        pred_pcc = cc.predict(X_test, method='pcc')
        pred_pac = cc.predict(X_test, method='pac')
        pred_hdy = cc.predict(X_test, method='hdy')

        freq = np.bincount(y_test, minlength=len(cc.classes_))
        true = freq / float(np.sum(freq))

        return true, pred_cc, pred_ac, pred_pcc, pred_pac, pred_hdy

    cluster = dispy.SharedJobCluster(train_fold,
                                     depends=[X_y_path, g_mean],
                                     reentrant=True,
                                     setup=functools.partial(setup, basename(X_y_path)),
                                     cleanup=cleanup,
                                     scheduler_node='dhcp015.aic.uniovi.es')
    true_and_preds = []

    jobs = []
    for n_fold, (train_index, test_index) in enumerate(loo.split(X)):
        job = cluster.submit(train_index, test_index)
        job.id = n_fold
        jobs.append(job)
        print "Fold {} submitted".format(n_fold)

    for job in jobs:
        job()
        print "Fold {} trained".format(job.id)
        result = job.result
        true_and_preds.append(result)

        if job.exception:
            raise ClusterException(job.exception + job.ip_addr)

    return true_and_preds




path = 'plancton.csv'
global file
file = open('{}.txt'.format('plancton_results_macro_avg_svc_linear'), 'wb')
plankton, le = load_plankton_file(path)
X = np.array(plankton.data)
y = np.array(plankton.target)

true_and_preds = cc(X, y)
print_and_write(true_and_preds)
file.close()


