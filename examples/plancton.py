# coding=utf-8
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.datasets.base import Bunch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import LabelEncoder

from quantification.classify_and_count.base import BaseMulticlassClassifyAndCount
from quantification.classify_and_count.ensemble import EnsembleMulticlassCC
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


def print_and_write(text):
    global file
    print text
    file.write(text + "\n")


def cc(X, y):
    loo = LeaveOneOut()

    cc_bcs = []
    cc_aes = []
    ac_bcs = []
    ac_aes = []
    pcc_bcs = []
    pcc_aes = []
    pac_bcs = []
    pac_aes = []
    hdy_bcs = []
    hdy_aes = []

    print_and_write("CC MONOCLASE")
    for n_fold, (train_index, test_index) in enumerate(loo.split(X)):
        print "Training fold {}/{}".format(n_fold + 1, 60)
        time_init = time.time()
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        cc = BaseMulticlassClassifyAndCount(b=100,
                                            estimator_class=LogisticRegression(),
                                            estimator_params={'class_weight': 'balanced'},
                                            estimator_grid={'C': [10 ** i for i in xrange(-3, 2)]}, strategy='micro')
        cc.fit(np.concatenate(X_train), np.concatenate(y_train), local=False, verbose=True, cv=3)

        predictions = cc.predict(X_test[0], method='cc')
        pred_cc = predictions

        predictions = cc.predict(X_test[0], method='ac')
        pred_ac = predictions

        predictions = cc.predict(X_test[0], method='pcc')
        pred_pcc = predictions

        predictions = cc.predict(X_test[0], method='pac')
        pred_pac = predictions

        predictions = cc.predict(X_test[0], method="hdy")
        pred_hdy = predictions

        freq = np.bincount(y_test[0], minlength=len(cc.classes_))
        true = freq / float(np.sum(freq))

        cc_bcs.append(bray_curtis(true, pred_cc))
        cc_aes.append(absolute_error(true, pred_cc))
        ac_bcs.append(bray_curtis(true, pred_ac))
        ac_aes.append(absolute_error(true, pred_ac))
        pcc_bcs.append(bray_curtis(true, pred_pcc))
        pcc_aes.append(absolute_error(true, pred_pcc))
        pac_bcs.append(bray_curtis(true, pred_pac))
        pac_aes.append(absolute_error(true, pred_pac))
        hdy_bcs.append(bray_curtis(true, pred_pac))
        hdy_aes.append(absolute_error(true, pred_hdy))

        print_and_write("Fold {}/{}".format(n_fold + 1, 60))
        print_and_write("\tRegularization C's = {}".format([clf.C for clf in cc.estimators_.values()]))
        print_and_write("\tTrue:            {}".format(true.tolist()))
        print_and_write("\tPredictions CC:  {}".format(pred_cc.tolist()))
        print_and_write("\tPredictions AC:  {}".format(pred_ac.tolist()))
        print_and_write("\tPredictions PCC: {}".format(pred_pcc.tolist()))
        print_and_write("\tPredictions PAC: {}".format(pred_pac.tolist()))
        print_and_write("\tPredictions HDy: {}".format(pred_hdy.tolist()))

        timed = time.time() - time_init
        print "Fold trained in {}".format(timed)

        """
        head = "{:>15}" * 3
        row_format = '{:>15}{:>15.2f}{:>15.2f}'
        print_and_write(head.format("Method", "Bray-Curtis", "AE"))
        print_and_write(row_format.format("CC", np.mean(cc_bcs), np.mean(cc_aes)))
        print_and_write(row_format.format("AC", np.mean(ac_bcs), np.mean(ac_aes)))
        print_and_write(row_format.format("PCC", np.mean(pcc_bcs), np.mean(pcc_aes)))
        print_and_write(row_format.format("PAC", np.mean(pac_bcs), np.mean(pac_aes)))
        """

    return np.mean(cc_aes), np.mean(ac_aes), np.mean(pcc_aes), np.mean(pac_aes), np.mean(hdy_aes), np.mean(
        cc_bcs), np.mean(ac_bcs), np.mean(pcc_bcs), np.mean(pac_bcs), np.mean(hdy_bcs)


def cc_ensemble(X, y):
    loo = LeaveOneOut()

    cc_bcs = []
    cc_aes = []
    ac_bcs = []
    ac_aes = []
    pcc_bcs = []
    pcc_aes = []
    pac_bcs = []
    pac_aes = []
    hdy_bcs = []
    hdy_aes = []

    print_and_write("CC ENSEMBLE")
    for n_fold, (train_index, test_index) in enumerate(loo.split(X)):
        print "Training fold {}/{}".format(n_fold + 1, 60)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        cc = EnsembleMulticlassCC(b=100,
                                  estimator_class=LogisticRegression(),
                                  estimator_params={'class_weight': 'balanced'},
                                  estimator_grid={'C': [10 ** i for i in xrange(-3, 2)]}, strategy='micro')
        cc.fit(X_train, y_train, verbose=True, local=False)
        pred_cc = []
        pred_ac = []
        pred_pcc = []
        pred_pac = []
        pred_hdy = []
        for X_s in X_test:
            predictions = cc.predict(X_s, method='cc')
            pred_cc.append(predictions.tolist())

            predictions = cc.predict(X_s, method='ac')
            pred_ac.append(predictions.tolist())

            predictions = cc.predict(X_s, method='pcc')
            pred_pcc.append(predictions.tolist())

            predictions = cc.predict(X_s, method='pac')
            pred_pac.append(predictions.tolist())

            predictions = cc.predict(X_s, method='hdy')
            pred_hdy.append(predictions.tolist())

        true = []
        for y_s in y_test:
            freq = np.bincount(y_s, minlength=len(cc.classes_))
            true.append(freq / float(np.sum(freq)))

        for (cc_pr, ac_pr, pcc_pr, pac_pr, hdy_pr, tr) in zip(pred_cc, pred_ac, pred_pcc, pred_pac, pred_hdy, true):
            cc_bcs.append(bray_curtis(tr, cc_pr))
            cc_aes.append(absolute_error(tr, cc_pr))
            ac_bcs.append(bray_curtis(tr, ac_pr))
            ac_aes.append(absolute_error(tr, ac_pr))
            pcc_bcs.append(bray_curtis(tr, pcc_pr))
            pcc_aes.append(absolute_error(tr, pcc_pr))
            pac_bcs.append(bray_curtis(tr, pac_pr))
            pac_aes.append(absolute_error(tr, pac_pr))
            hdy_bcs.append(bray_curtis(tr, hdy_pr))
            hdy_aes.append(absolute_error(tr, hdy_pr))

        print_and_write("Fold {}/{}".format(n_fold + 1, 60))
        print_and_write("\tTrue:            {}".format(true))
        print_and_write("\tPredictions CC:  {}".format(pred_cc))
        print_and_write("\tPredictions AC:  {}".format(pred_ac))
        print_and_write("\tPredictions PCC: {}".format(pred_pcc))
        print_and_write("\tPredictions PAC: {}".format(pred_pac))
        print_and_write("\tPredictions HDy: {}".format(pred_hdy))
        print_and_write("")

    """
    head = "{:>15}" * 3
    row_format = '{:>15}{:>15.2f}{:>15.2f}'
    print_and_write(head.format("Method", "Bray-Curtis", "AE"))
    print_and_write(row_format.format("CC-Ensemble", np.mean(cc_bcs), np.mean(cc_aes)))
    print_and_write(row_format.format("AC-Ensemble", np.mean(ac_bcs), np.mean(ac_aes)))
    print_and_write(row_format.format("PCC-Ensemble", np.mean(pcc_bcs), np.mean(pcc_aes)))
    print_and_write(row_format.format("PAC-Ensemble", np.mean(pac_bcs), np.mean(pac_aes)))
    """
    return np.mean(cc_aes), np.mean(ac_aes), np.mean(pcc_aes), np.mean(pac_aes), np.mean(hdy_aes), np.mean(
        cc_bcs), np.mean(ac_bcs), np.mean(pcc_bcs), np.mean(pac_bcs), np.mean(hdy_bcs)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    plankton, le = load_plankton_file('plancton.csv')

    X = np.array(plankton.data)
    y = np.array(plankton.target)
    global file
    file = open('{}.txt'.format('plancton_results_micro_avg'), 'wb')
    cc_err, ac_err, pcc_err, pac_err, hdy_err, cc_bc, ac_bc, pcc_bc, pac_bc, hdy_bc = cc(X, y)
    ecc_err, eac_err, epcc_err, epac_err, ehdy_err, ecc_bc, eac_bc, epcc_bc, epac_bc, ehdy_bc = cc_ensemble(X, y)

    head = "{:>15}" * 3
    row_format = '{:>15}{:>15.4f}{:>15.4f}'
    print_and_write(head.format("Method", "Error", "Bray-Curtis"))
    print_and_write(row_format.format("CC", cc_err, cc_bc))
    print_and_write(row_format.format("AC", ac_err, ac_bc))
    print_and_write(row_format.format("PCC", pcc_err, pcc_bc))
    print_and_write(row_format.format("PAC", pac_err, pac_bc))
    print_and_write(row_format.format("HDy", hdy_err, hdy_bc))
    print_and_write(row_format.format("Ensemble-CC", ecc_err, ecc_bc))
    print_and_write(row_format.format("Ensemble-AC", eac_err, eac_bc))
    print_and_write(row_format.format("Ensemble-PCC", epcc_err, epcc_bc))
    print_and_write(row_format.format("Ensemble-PAC", epac_err, epac_bc))
    print_and_write(row_format.format("Ensemble-HDy", ehdy_err, ehdy_bc))
    file.close()
