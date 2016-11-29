# coding=utf-8
import pandas as pd
from sklearn.datasets.base import Bunch
from sklearn.preprocessing import LabelEncoder

import numpy as np

from quantification.classify_and_count.base import MulticlassClassifyAndCount
from quantification.classify_and_count.ensemble import EnsembleMulticlassCC
from quantification.distribution_matching.base import MulticlassHDy
from quantification.metrics.multiclass import absolute_error


def load_plankton_file(path, sample_col="Sample", target_col="class"):

    data_file = pd.read_csv(path, delimiter=' ')
    le = LabelEncoder()
    data_file[target_col] = le.fit_transform(data_file[target_col])
    data = data_file.groupby(sample_col)
    target = [sample[1].values for sample in data[target_col]]
    features = [sample[1].drop([sample_col, target_col], axis=1, inplace=False).values for sample in data]
    return Bunch(data=features, target=target,
                 target_names=le.classes_), le


def cc():
    plankton,le = load_plankton_file('/Users/albertocastano/Dropbox/PlataformaCuantificación/plancton.csv')
    cc = MulticlassClassifyAndCount()
    X = plankton.data
    y = plankton.target
    cc.fit(np.concatenate(X), np.concatenate(y), cv=50, verbose=True, local=False)
    print "Fitted"
    pred_cc = []
    pred_ac = []
    pred_pcc = []
    pred_pac = []
    for X_s in X:
        predictions = cc.predict(X_s, method='cc')
        pred_cc.append(predictions)

        predictions = cc.predict(X_s, method='ac')
        pred_ac.append(predictions)

        predictions = cc.predict(X_s, method='pcc')
        pred_pcc.append(predictions)

        predictions = cc.predict(X_s, method='pac')
        pred_pac.append(predictions)
    true = []
    for y_s in y:
        freq = np.bincount(y_s, minlength=len(cc.classes_))
        true.append(freq / float(np.sum(freq)))

    for (cc, ac, pcc, pac, tr) in zip(pred_cc, pred_ac, pred_pcc, pred_pac, true):
        print "CC:\t\t", ["{0:0.2f}".format(i) for i in cc]
        print "AC:\t\t", ["{0:0.2f}".format(i) for i in ac]
        print "PCC:\t", ["{0:0.2f}".format(i) for i in pcc]
        print "PAC:\t", ["{0:0.2f}".format(i) for i in pac]
        print "True:\t", ["{0:0.2f}".format(i) for i in tr]
        print ""


def cc_ensemble():
    plankton, le = load_plankton_file('/Users/albertocastano/Dropbox/PlataformaCuantificación/plancton.csv')
    cc = EnsembleMulticlassCC()
    X = plankton.data
    y = plankton.target
    cc.fit(X, y, verbose=True)
    print "Fitted"
    pred_cc = []
    pred_ac = []
    pred_pcc = []
    pred_pac = []
    for X_s in X:
        predictions = cc.predict(X_s, method='cc')
        pred_cc.append(predictions)

        predictions = cc.predict(X_s, method='ac')
        pred_ac.append(predictions)

        predictions = cc.predict(X_s, method='pcc')
        pred_pcc.append(predictions)

        predictions = cc.predict(X_s, method='pac')
        pred_pac.append(predictions)
    true = []
    for y_s in y:
        freq = np.bincount(y_s, minlength=len(cc.classes_))
        true.append(freq / float(np.sum(freq)))

    for (cc, ac, pcc, pac, tr) in zip(pred_cc, pred_ac, pred_pcc, pred_pac, true):
        print "CC:\t\t", ["{0:0.2f}".format(i) for i in cc]
        print "AC:\t\t", ["{0:0.2f}".format(i) for i in ac]
        print "PCC:\t", ["{0:0.2f}".format(i) for i in pcc]
        print "PAC:\t", ["{0:0.2f}".format(i) for i in pac]
        print "True:\t", ["{0:0.2f}".format(i) for i in tr]


def hdy():
    plankton, le = load_plankton_file('/Users/albertocastano/Dropbox/PlataformaCuantificación/plancton.csv')
    cc = MulticlassHDy(b=100)
    X = plankton.data
    y = plankton.target
    cc.fit(np.concatenate(X), np.concatenate(y), verbose=True, plot=False)
    print "Fitted"
    preds = []
    for X_s in X:
        predictions = cc.predict(X_s)
        preds.append(predictions)
    true = []
    for y_s in y:
        freq = np.bincount(y_s, minlength=len(cc.classes_))
        true.append(freq / float(np.sum(freq)))

    aes = []
    for pr, tr in zip(preds, true):
        ae = absolute_error(tr, pr)
        aes.append(ae)
        print "Absolute error:", ae

    print "Mean absolute error:", np.mean(aes)


if __name__ == '__main__':
    hdy()