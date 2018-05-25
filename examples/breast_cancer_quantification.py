from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from quantification.cc import BaseCC
from quantification.dm.base import EDx, EDy, CvMy, CvMX, MMy, FriedmanBM, FriedmanMM, FriedmanDB, LSDD, HDX


def main():
    # Load the data
    X, y = load_breast_cancer(return_X_y=True)
    scaler = StandardScaler()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    scaler.fit(X)
    # X_train = scaler.transform(X_train)
    # X_test = scaler.transform(X_test)

    # X_train, X_test, y_train, y_test = X, X, y, y

    # Create de quantifier. We will use a simple logistic regression as the underlying classifier.
    # Number of bins to perform HDy will be set to 8.
    qnf = BaseCC(estimator_class=LogisticRegression(random_state=42), b=8)

    # Fit the quantifier
    qnf.fit(X_train, y_train)

    # n_neg = (y_train == 0).sum()
    #
    # pos = np.where(y_train == 1)[0][:n_neg]
    # neg = np.where(y_train == 0)[0]
    #
    # idxs = np.concatenate([pos, neg])
    #
    # X_train = X_train[idxs]
    # y_train = y_train[idxs]

    # Get the true binary prevalence, i.e., percentage of positives samples.
    prev_true = np.bincount(y_test, minlength=2) / float(len(y_test))
    prev_true = prev_true[1]

    # Get the predicted prevalences using every of the methods available in the BaseBinaryCC
    prev_cc = qnf.predict(X_test, method='cc')[1]
    prev_ac = qnf.predict(X_test, method='ac')[1]
    prev_pcc = qnf.predict(X_test, method='pcc')[1]
    prev_pac = qnf.predict(X_test, method='pac')[1]
    prev_hdy = qnf.predict(X_test, method='hdy')[1]

    hdx = HDX(b=8)
    hdx.fit(X_train, y_train)
    prev_hdx = hdx.predict(X_test)[1]

    edx = EDx()
    edx.fit(X_train, y_train)
    prev_edx = edx.predict(X_test)[1]

    edy = EDy()
    edy.fit(X_train, y_train)
    prev_edy = edy.predict(X_test)[1]

    formatter = "{:<4}{:>15.4f}"
    print(formatter.format("True", prev_true))
    print(formatter.format("CC", prev_cc))
    print(formatter.format("AC", prev_ac))
    print(formatter.format("PCC", prev_pcc))
    print(formatter.format("PAC", prev_pac))
    print(formatter.format("HDy", prev_hdy))
    print(formatter.format("HDX", prev_hdx))
    print(formatter.format("EDx", prev_edx))
    print(formatter.format("EDy", prev_edy))

    cvmy = CvMy()
    cvmy.fit(X_train, y_train)
    prev_cvmy = cvmy.predict(X_test)[1]
    print(formatter.format("CvMy", prev_cvmy))

    cvmx = CvMX()
    cvmx.fit(X_train, y_train)
    prev_cvmx = cvmx.predict(X_test)[1]
    print(formatter.format("CvMX", prev_cvmx))

    mmy = MMy(b=8)
    mmy.fit(X_train, y_train)
    prev_mmy = mmy.predict(X_test)[1]
    print(formatter.format("MMy", prev_mmy))

    fr = FriedmanMM()
    fr.fit(X_train, y_train)
    prev_fr = fr.predict(X_test)[1]
    print(formatter.format("Friedman-MM", prev_fr))

    fr = FriedmanBM()
    fr.fit(X_train, y_train)
    prev_fr = fr.predict(X_test)[1]
    print(formatter.format("Friedman-BM", prev_fr))

    fr = FriedmanDB()
    fr.fit(X_train, y_train)
    prev_fr = fr.predict(X_test)[1]
    print(formatter.format("Friedman-DB", prev_fr))

    lsdd = LSDD(sampling=False, tol=1e-5, lda=0)
    lsdd.fit(X_train, y_train)
    prev_lsdd = lsdd.predict(X_test)[1]
    print(formatter.format("LSDD", prev_lsdd))


if __name__ == '__main__':
    main()
