# coding=utf-8
import warnings

import pandas as pd
from sklearn.datasets.base import Bunch
from sklearn.model_selection import LeaveOneOut
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
    X = np.array(plankton.data)
    y = np.array(plankton.target)
    loo = LeaveOneOut()

    cc_bcs = []
    cc_aes = []
    ac_bcs = []
    ac_aes = []
    pcc_bcs = []
    pcc_aes = []
    pac_bcs = []
    pac_aes = []

    print "CC MONOCLASE"
    for n_fold, (train_index, test_index) in enumerate(loo.split(X)):
        print "\rTraining fold {}/{}".format(n_fold, 60),
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        cc = BaseMulticlassClassifyAndCount(estimator_params={'class_weight': 'balanced'})
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


        for (cc_pr, ac_pr, pcc_pr, pac_pr, tr) in zip(pred_cc, pred_ac, pred_pcc, pred_pac, true):
            cc_bcs.append(bray_curtis(tr, cc_pr))
            cc_aes.append(absolute_error(tr, cc_pr))
            ac_bcs.append(bray_curtis(tr, ac_pr))
            ac_aes.append(absolute_error(tr, ac_pr))
            pcc_bcs.append(bray_curtis(tr, pcc_pr))
            pcc_aes.append(absolute_error(tr, pcc_pr))
            pac_bcs.append(bray_curtis(tr, pac_pr))
            pac_aes.append(absolute_error(tr, pac_pr))

    head = "{:>15}" * 3
    row_format = '{:>15}{:>15.2f}{:>15.2f}'
    print head.format("Method", "Bray-Curtis", "AE")
    print row_format.format("CC", np.mean(cc_bcs), np.mean(cc_aes))
    print row_format.format("AC", np.mean(ac_bcs), np.mean(ac_aes))
    print row_format.format("PCC", np.mean(pcc_bcs), np.mean(pcc_aes))
    print row_format.format("PAC", np.mean(pac_bcs), np.mean(pac_aes))


def cc_ensemble():
    plankton, le = load_plankton_file('/Users/albertocastano/Dropbox/PlataformaCuantificación/plancton.csv')
    X = np.array(plankton.data)
    y = np.array(plankton.target)
    loo = LeaveOneOut()

    cc_bcs = []
    cc_aes = []
    ac_bcs = []
    ac_aes = []
    pcc_bcs = []
    pcc_aes = []
    pac_bcs = []
    pac_aes = []

    print "CC ENSEMBLE"
    for n_fold, (train_index, test_index) in enumerate(loo.split(X)):
        print "\rTraining fold {}/{}".format(n_fold, 60),
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        cc = EnsembleMulticlassCC(estimator_params={'class_weight': 'balanced'})
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


        for (cc_pr, ac_pr, pcc_pr, pac_pr, tr) in zip(pred_cc, pred_ac, pred_pcc, pred_pac, true):
            cc_bcs.append(bray_curtis(tr, cc))
            cc_aes.append(absolute_error(tr, cc))
            ac_bcs.append(bray_curtis(tr, ac_pr))
            ac_aes.append(absolute_error(tr, ac_pr))
            pcc_bcs.append(bray_curtis(tr, pcc_pr))
            pcc_aes.append(absolute_error(tr, pcc_pr))
            pac_bcs.append(bray_curtis(tr, pac_pr))
            pac_aes.append(absolute_error(tr, pac_pr))

    row_format = '{:>15}{:>15.2f}{:>15.2f}'
    print row_format.format("CC-Ensemble", np.mean(cc_bcs), np.mean(cc_aes))
    print row_format.format("AC-Ensemble", np.mean(ac_bcs), np.mean(ac_aes))
    print row_format.format("PCC-Ensemble", np.mean(pcc_bcs), np.mean(pcc_aes))
    print row_format.format("PAC-Ensemble", np.mean(pac_bcs), np.mean(pac_aes))


def hdy():
    plankton, le = load_plankton_file('/Users/albertocastano/Dropbox/PlataformaCuantificación/plancton.csv')
    X = np.array(plankton.data)
    y = np.array(plankton.target)
    loo = LeaveOneOut()

    bcs = []
    aes = []

    print "HDy MONOCLASE"
    for n_fold, (train_index, test_index) in enumerate(loo.split(X)):
        print "\rTraining fold {}/{}".format(n_fold, 60),
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        hdy = MulticlassHDy(b=100, estimator_params={'class_weight': 'balanced'})
        hdy.fit(np.concatenate(X_train), np.concatenate(y_train), verbose=False, plot=False)
        preds = []
        for X_s in X_test:
            predictions = hdy.predict(X_s)
            preds.append(predictions)
        true = []
        for y_s in y_test:
            freq = np.bincount(y_s, minlength=len(hdy.classes_))
            true.append(freq / float(np.sum(freq)))


        for pr, tr in zip(preds, true):
            bcs.append(bray_curtis(tr, pr))
            aes.append(absolute_error(tr, pr))

    row_format = '{:>15}{:>15.2f}{:>15.2f}'
    print row_format.format("HDy", np.mean(bcs), np.mean(aes))


def hdy_ensemble():
    plankton, le = load_plankton_file('/Users/albertocastano/Dropbox/PlataformaCuantificación/plancton.csv')
    
    X = np.array(plankton.data)
    y = np.array(plankton.target)
    loo = LeaveOneOut()

    bcs = []
    aes = []

    print "HDy ENSEMBLE"
    for n_fold, (train_index, test_index) in enumerate(loo.split(X)):
        print "\rTraining fold {}/{}".format(n_fold, 60),
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        hdy = MulticlassEnsembleHDy(b=100, estimator_params={'class_weight': 'balanced'})
        hdy.fit(X_train, y_train, verbose=False, plot=False, local=False)
        preds = []
        for n, X_s in enumerate(X_test):
            # print "\rPredicting sample {}/{}".format(n+1, len(X_test))
            predictions = hdy.predict(X_s, local=False)
            preds.append(predictions)
        true = []
        for y_s in y_test:
            freq = np.bincount(y_s, minlength=len(hdy.classes_))
            true.append(freq / float(np.sum(freq)))


        for pr, tr in zip(preds, true):
            bcs.append(bray_curtis(tr, pr))
            aes.append(absolute_error(tr, pr))

    row_format = '{:>15}{:>15.2f}{:>15.2f}'
    print row_format.format("HDy-Ensemble", np.mean(bcs), np.mean(aes))


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    cc()
    cc_ensemble()
    hdy()
    hdy_ensemble()
