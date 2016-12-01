# coding=utf-8
import warnings

import pandas as pd
from sklearn.datasets.base import Bunch
from sklearn.preprocessing import LabelEncoder

import numpy as np

from quantification.classify_and_count.base import BaseMulticlassClassifyAndCount
from quantification.classify_and_count.ensemble import EnsembleMulticlassCC
from quantification.distribution_matching.base import MulticlassHDy
from quantification.distribution_matching.ensemble import MulticlassEnsembleHDy
from quantification.metrics.multiclass import absolute_error, bray_curtis


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
    plankton, le = load_plankton_file('/Users/albertocastano/Dropbox/PlataformaCuantificación/plancton.csv')
    cc = BaseMulticlassClassifyAndCount(estimator_params={'class_weight': 'balanced'})
    X = plankton.data
    y = plankton.target
    X_train = X[:40]
    y_train = y[:40]
    X_test = X[40:]
    y_test = y[40:]

    cc.fit(np.concatenate(X_train), np.concatenate(y_train), verbose=False, local=False)
    pred_cc = []
    pred_ac = []
    pred_pcc = []
    pred_pac = []
    for X_s in X_test:
        predictions = cc.predict(X_s, method='cc')
        pred_cc.append(predictions)

        predictions = cc.predict(X_s, method='ac')
        pred_ac.append(predictions)

        predictions = cc.predict(X_s, method='pcc')
        pred_pcc.append(predictions)

        predictions = cc.predict(X_s, method='pac')
        pred_pac.append(predictions)
    true = []
    for y_s in y_test:
        freq = np.bincount(y_s, minlength=len(cc.classes_))
        true.append(freq / float(np.sum(freq)))

    cc_bcs = []
    cc_aes = []
    ac_bcs = []
    ac_aes = []
    pcc_bcs = []
    pcc_aes = []
    pac_bcs = []
    pac_aes = []
    for (cc, ac, pcc, pac, tr) in zip(pred_cc, pred_ac, pred_pcc, pred_pac, true):
        cc_bcs.append(bray_curtis(tr, cc))
        cc_aes.append(absolute_error(tr, cc))
        ac_bcs.append(bray_curtis(tr, ac))
        ac_aes.append(absolute_error(tr, ac))
        pcc_bcs.append(bray_curtis(tr, pcc))
        pcc_aes.append(absolute_error(tr, pcc))
        pac_bcs.append(bray_curtis(tr, pac))
        pac_aes.append(absolute_error(tr, pac))

    print "CC Mean Bray-Curtis Dissimilarity:", np.mean(cc_bcs)
    print "AC Mean Bray-Curtis Dissimilarity:", np.mean(ac_bcs)
    print "PCC Mean Bray-Curtis Dissimilarity:", np.mean(pcc_bcs)
    print "PAC Mean Bray-Curtis Dissimilarity:", np.mean(pac_bcs)


def cc_ensemble():
    plankton, le = load_plankton_file('/Users/albertocastano/Dropbox/PlataformaCuantificación/plancton.csv')
    cc = EnsembleMulticlassCC(estimator_params={'class_weight': 'balanced'})
    X = plankton.data
    y = plankton.target
    X_train = X[:40]
    y_train = y[:40]
    X_test = X[40:]
    y_test = y[40:]

    cc.fit(X_train, y_train, verbose=False)
    pred_cc = []
    pred_ac = []
    pred_pcc = []
    pred_pac = []
    for X_s in X_test:
        predictions = cc.predict(X_s, method='cc')
        pred_cc.append(predictions)

        predictions = cc.predict(X_s, method='ac')
        pred_ac.append(predictions)

        predictions = cc.predict(X_s, method='pcc')
        pred_pcc.append(predictions)

        predictions = cc.predict(X_s, method='pac')
        pred_pac.append(predictions)
    true = []
    for y_s in y_test:
        freq = np.bincount(y_s, minlength=len(cc.classes_))
        true.append(freq / float(np.sum(freq)))

    cc_bcs = []
    cc_aes = []
    ac_bcs = []
    ac_aes = []
    pcc_bcs = []
    pcc_aes = []
    pac_bcs = []
    pac_aes = []
    for (cc, ac, pcc, pac, tr) in zip(pred_cc, pred_ac, pred_pcc, pred_pac, true):
        cc_bcs.append(bray_curtis(tr, cc))
        cc_aes.append(absolute_error(tr, cc))
        ac_bcs.append(bray_curtis(tr, ac))
        ac_aes.append(absolute_error(tr, ac))
        pcc_bcs.append(bray_curtis(tr, pcc))
        pcc_aes.append(absolute_error(tr, pcc))
        pac_bcs.append(bray_curtis(tr, pac))
        pac_aes.append(absolute_error(tr, pac))

    print "CC-Ensemble Mean Bray-Curtis Dissimilarity:", np.mean(cc_bcs)
    print "AC-Ensemble Mean Bray-Curtis Dissimilarity:", np.mean(ac_bcs)
    print "PCC-Ensemble Mean Bray-Curtis Dissimilarity:", np.mean(pcc_bcs)
    print "PAC-Ensemble Mean Bray-Curtis Dissimilarity:", np.mean(pac_bcs)


def hdy():
    plankton, le = load_plankton_file('/Users/albertocastano/Dropbox/PlataformaCuantificación/plancton.csv')
    cc = MulticlassHDy(b=100, estimator_params={'class_weight': 'balanced'})
    X = plankton.data
    y = plankton.target
    X_train = X[:40]
    y_train = y[:40]
    X_test = X[40:]
    y_test = y[40:]

    cc.fit(np.concatenate(X_train), np.concatenate(y_train), verbose=False, plot=False)
    preds = []
    for X_s in X_test:
        predictions = cc.predict(X_s)
        preds.append(predictions)
    true = []
    for y_s in y_test:
        freq = np.bincount(y_s, minlength=len(cc.classes_))
        true.append(freq / float(np.sum(freq)))

    bcs = []
    aes = []
    for pr, tr in zip(preds, true):
        bcs.append(bray_curtis(tr, pr))
        aes.append(absolute_error(tr, pr))

    print "HDy Mean Bray-Curtis Dissimilarity:", np.mean(bcs)


def hdy_ensemble():
    plankton, le = load_plankton_file('/Users/albertocastano/Dropbox/PlataformaCuantificación/plancton.csv')
    cc = MulticlassEnsembleHDy(b=100, estimator_params={'class_weight': 'balanced'})
    X = plankton.data
    y = plankton.target
    X_train = X[:40]
    y_train = y[:40]
    X_test = X[40:]
    y_test = y[40:]

    cc.fit(X_train, y_train, verbose=False, plot=False, local=False)
    preds = []
    for n, X_s in enumerate(X_test):
        # print "\rPredicting sample {}/{}".format(n+1, len(X_test))
        predictions = cc.predict(X_s, local=False)
        preds.append(predictions)
    true = []
    for y_s in y_test:
        freq = np.bincount(y_s, minlength=len(cc.classes_))
        true.append(freq / float(np.sum(freq)))

    bcs = []
    aes = []
    for pr, tr in zip(preds, true):
        bcs.append(bray_curtis(tr, pr))
        aes.append(absolute_error(tr, pr))

    print "HDy-Ensemble Mean Bray-Curtis Dissimilarity:", np.mean(bcs)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    cc()
    print ""
    cc_ensemble()
    print ""
    hdy()
    print ""
    hdy_ensemble()
