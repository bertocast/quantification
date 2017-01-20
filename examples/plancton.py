# coding=utf-8
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.datasets.base import Bunch
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import LabelEncoder

from quantification.classify_and_count.base import BaseMulticlassClassifyAndCount
from quantification.classify_and_count.ensemble import EnsembleMulticlassCC
from quantification.metrics.multiclass import absolute_error, bray_curtis
from quantification.utils.models import LSPC, pair_distance_centiles


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

    cm = confusion_matrix(y, estimator.predict(X), labels=estimator.classes_)

    fpr = cm[0, 1] / float(cm[0, 1] + cm[0, 0])
    tpr = cm[1, 1] / float(cm[1, 1] + cm[1, 0])

    return np.sqrt((1 - fpr) * tpr)


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

    print_and_write("CC MONOSAMPLE")
    for n_fold, (train_index, test_index) in enumerate(loo.split(X)):
        print "Training fold {}/{}".format(n_fold + 1, 60)
        time_init = time.time()
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        X_train = np.concatenate(X_train)[:500,]
        y_train = np.concatenate(y_train)[:500]

        sigma_candidates = pair_distance_centiles(X_train, centiles=[50, 90])
        rho_candidates = [.01, .1, 1.]
        param_grid = dict(sigma=sigma_candidates, rho=rho_candidates)
        grid_options = {'scoring': g_mean, 'verbose': 11}

        cc = BaseMulticlassClassifyAndCount(b=8,
                                            estimator_class=LSPC(),
                                            estimator_params=dict(rho=.01, sigma=1000.0),
                                            strategy='macro')
        cc.fit(X_train, y_train, local=False, verbose=True)

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

        timed = time.time() - time_init
        print "Fold trained in {}".format(timed)

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

        sigma_candidates = pair_distance_centiles(np.concatenate(X_train), centiles=[10, 50, 90])
        rho_candidates = [.01, .1, .5, 1.]
        param_grid = dict(sigma=sigma_candidates, rho=rho_candidates)
        grid_options = {'scoring': g_mean, 'verbose': 11}

        cc = EnsembleMulticlassCC(b=8,
                                  estimator_class=LSPC(),
                                  estimator_params=dict(),
                                  estimator_grid=param_grid,
                                  grid_params=grid_options,
                                  strategy='macro')
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

    return np.mean(cc_aes), np.mean(ac_aes), np.mean(pcc_aes), np.mean(pac_aes), np.mean(hdy_aes), np.mean(
        cc_bcs), np.mean(ac_bcs), np.mean(pcc_bcs), np.mean(pac_bcs), np.mean(hdy_bcs)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    plankton, le = load_plankton_file('plancton.csv')
    global file
    file = open('{}.txt'.format('plancton_results_macro_avg_klr'), 'wb')
    X = np.array(plankton.data)
    y = np.array(plankton.target)

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
