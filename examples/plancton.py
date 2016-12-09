# coding=utf-8
import warnings

import pandas as pd
from sklearn.datasets.base import Bunch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import LabelEncoder

import numpy as np

from quantification.classify_and_count.base import BaseMulticlassClassifyAndCount, BaseBinaryClassifyAndCount
from quantification.classify_and_count.ensemble import EnsembleMulticlassCC, EnsembleBinaryCC
from quantification.distribution_matching.base import MulticlassHDy, BinaryHDy
from quantification.distribution_matching.ensemble import MulticlassEnsembleHDy, BinaryEnsembleHDy
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


def cc(X, y):
    f = open('cc.txt', 'wb')

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
        print "Training fold {}/{}".format(n_fold + 1, 60)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        cc = BaseBinaryClassifyAndCount(estimator_class=LogisticRegression(),
                                        estimator_params={'class_weight': 'balanced'},
                                        estimator_grid={'C': [10 ** i for i in xrange(-3, 2)]})
        cc.fit(np.concatenate(X_train), np.concatenate(y_train), local=False)

        predictions = cc.predict(X_test[0], method='cc')
        pred_cc = predictions

        predictions = cc.predict(X_test[0], method='ac')
        pred_ac = predictions

        predictions = cc.predict(X_test[0], method='pcc')
        pred_pcc= predictions

        predictions = cc.predict(X_test[0], method='pac')
        pred_pac = predictions

        freq = np.bincount(y_test[0], minlength=2)
        true = freq / float(np.sum(freq))

        cc_bcs.append(bray_curtis(true, pred_cc))
        cc_aes.append(absolute_error(true, pred_cc))
        ac_bcs.append(bray_curtis(true, pred_ac))
        ac_aes.append(absolute_error(true, pred_ac))
        pcc_bcs.append(bray_curtis(true, pred_pcc))
        pcc_aes.append(absolute_error(true, pred_pcc))
        pac_bcs.append(bray_curtis(true, pred_pac))
        pac_aes.append(absolute_error(true, pred_pac))
        f.write("Fold {}/{}\n".format(n_fold + 1, 60))
        f.write("\tRegularization C = {}\n".format(cc.estimator_.C))
        f.write("\tPredictions CC: {}\n".format(pred_cc.tolist()))
        f.write("\tPredictions AC: {}\n".format(pred_ac.tolist()))
        f.write("\tPredictions PCC: {}\n".format(pred_pcc.tolist()))
        f.write("\tPredictions PAC: {}\n".format(pred_pac.tolist()))

        print cc.confusion_matrix_

    head = "{:>15}" * 3 + '\n'
    row_format = '{:>15}{:>15.2f}{:>15.2f}\n'
    f.write(head.format("Method", "Bray-Curtis", "AE"))
    f.write(row_format.format("CC", np.mean(cc_bcs), np.mean(cc_aes)))
    f.write(row_format.format("AC", np.mean(ac_bcs), np.mean(ac_aes)))
    f.write(row_format.format("PCC", np.mean(pcc_bcs), np.mean(pcc_aes)))
    f.write(row_format.format("PAC", np.mean(pac_bcs), np.mean(pac_aes)))

    f.close()


def cc_ensemble(X, y):
    f = open('cc_ensemble.txt', 'wb')
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
        print "Training fold {}/{}".format(n_fold+1, 60),
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        cc = EnsembleBinaryCC(estimator_class=LogisticRegression(),
                              estimator_params={'class_weight': 'balanced'},
                              estimator_grid={'C': [10 ** i for i in xrange(-3, 2)]})
        cc.fit(X_train, y_train)
        pred_cc = []
        pred_ac = []
        pred_pcc = []
        pred_pac = []
        for X_s in X_test:
            predictions = cc.predict(X_s, method='cc')
            pred_cc.append(predictions.tolist())

            predictions = cc.predict(X_s, method='ac')
            pred_ac.append(predictions.tolist())

            predictions = cc.predict(X_s, method='pcc')
            pred_pcc.append(predictions.tolist())

            predictions = cc.predict(X_s, method='pac')
            pred_pac.append(predictions.tolist())
        true = []
        for y_s in y_test:
            freq = np.bincount(y_s, minlength=2)
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

        f.write("Fold {}/{}\n".format(n_fold + 1, 60))
        f.write("\tRegularization C's = {}\n".format([qnf.estimator_.C for qnf in cc.qnfs_]))
        f.write("\tPredictions CC: {}\n".format(pred_cc))
        f.write("\tPredictions AC: {}\n".format(pred_ac))
        f.write("\tPredictions PCC: {}\n".format(pred_pcc))
        f.write("\tPredictions PAC: {}\n".format(pred_pac))
        print ""

    head = "{:>15}" * 3 + '\n'
    row_format = '{:>15}{:>15.2f}{:>15.2f}\n'
    f.write(head.format("Method", "Bray-Curtis", "AE"))
    f.write(row_format.format("CC-Ensemble", np.mean(cc_bcs), np.mean(cc_aes)))
    f.write(row_format.format("AC-Ensemble", np.mean(ac_bcs), np.mean(ac_aes)))
    f.write(row_format.format("PCC-Ensemble", np.mean(pcc_bcs), np.mean(pcc_aes)))
    f.write(row_format.format("PAC-Ensemble", np.mean(pac_bcs), np.mean(pac_aes)))

    f.close()


def hdy(X, y):
    f = open('hdy.txt', 'wb')
    loo = LeaveOneOut()

    bcs = []
    aes = []

    print "HDy MONOCLASE"
    for n_fold, (train_index, test_index) in enumerate(loo.split(X)):
        print "Training fold {}/{}".format(n_fold, 60)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        hdy = BinaryHDy(b=100, estimator_class=LogisticRegression(),
                                        estimator_params={'class_weight': 'balanced'},
                                        estimator_grid={'C': [10 ** i for i in xrange(-3, 2)]})
        hdy.fit(np.concatenate(X_train), np.concatenate(y_train), plot=False)
        predictions = hdy.predict(X_test[0])

        freq = np.bincount(y_test[0], minlength=2)
        true = freq / float(np.sum(freq))

        bcs.append(bray_curtis(true, predictions))
        aes.append(absolute_error(true, predictions))
        f.write("Fold {}/{}\n".format(n_fold + 1, 60))
        f.write("\tRegularization C = {}\n".format(hdy.estimator_.C))
        f.write("\tPredictions CC: {}\n".format(predictions.tolist()))
        print""

    head = "{:>15}" * 3 + '\n'
    row_format = '{:>15}{:>15.2f}{:>15.2f}\n'
    f.write(head.format("Method", "Bray-Curtis", "AE"))
    f.write(row_format.format("HDy", np.mean(bcs), np.mean(aes)))


def hdy_ensemble(X, y):
    f = open('hdy_ensemble.txt', 'wb')
    loo = LeaveOneOut()

    bcs = []
    aes = []

    print "HDy ENSEMBLE"
    for n_fold, (train_index, test_index) in enumerate(loo.split(X)):
        print "\rTraining fold {}/{}".format(n_fold + 1, 60),
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        hdy = BinaryEnsembleHDy(b=100, estimator_class=LogisticRegression(),
                              estimator_params={'class_weight': 'balanced'},
                              estimator_grid={'C': [10 ** i for i in xrange(-3, 2)]})
        hdy.fit(X_train, y_train, plot=False)
        preds = []
        for n, X_s in enumerate(X_test):
            # print "\rPredicting sample {}/{}".format(n+1, len(X_test))
            predictions = hdy.predict(X_s)
            preds.append(predictions)
        true = []
        for y_s in y_test:
            freq = np.bincount(y_s, minlength=2)
            true.append(freq / float(np.sum(freq)))

        for pr, tr in zip(preds, true):
            bcs.append(bray_curtis(tr, pr))
            aes.append(absolute_error(tr, pr))

        f.write("Fold {}/{}\n".format(n_fold + 1, 60))
        f.write("\tRegularization C's = {}\n".format([qnf.estimator_.C for qnf in hdy.qnfs_]))
        f.write("\tPredictions CC: {}\n".format(preds))
        print ""

    head = "{:>15}" * 3 + '\n'
    row_format = '{:>15}{:>15.2f}{:>15.2f}\n'
    f.write(head.format("Method", "Bray-Curtis", "AE"))
    f.write(row_format.format("HDy-Ensemble", np.mean(bcs), np.mean(aes)))

    f.close()


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    plankton, le = load_plankton_file('/Users/albertocastano/Dropbox/PlataformaCuantificación/plancton.csv')
    X = np.array(plankton.data)
    y = np.array(plankton.target)
    for n, y_sample in enumerate(y):
        mask = (y_sample == 4)  # Diatomeas
        y_bin = np.ones(y_sample.shape, dtype=np.int)
        y_bin[~mask] = 0
        y[n] = y_bin
    cc(X, y)
    cc_ensemble(X, y)
    hdy(X, y)
    hdy_ensemble(X, y)
