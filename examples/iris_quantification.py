from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from quantification.cc import BaseCC
from quantification.cc.base import FriedmanAC
from quantification.dm.base import EDx, kEDx, EDy, FriedmanBM, FriedmanMM, HDX


def main():
    # Load the data
    X, y = load_iris(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=1234)

    # X=X[25:-25]
    # y=y[25:-25]

    # Create de quantifier. We will use a simple logistic regression as the underlying classifier.
    # Number of bins to perform HDy will be set to 8.
    qnf = BaseCC(estimator_class=LogisticRegression(), b=8)

    # Fit the quantifier
    qnf.fit(X_train, y_train, cv=3)

    # Get the true prevalence, i.e., percentage of positives samples in each of the classes.
    prev_true = np.bincount(y_test, minlength=3) / float(len(y_test))

    # Get the predicted prevalences using every of the methods available in the BaseMulticlassCC
    prev_cc = qnf.predict(X_test, method='cc')
    prev_ac = qnf.predict(X_test, method='ac')
    prev_pcc = qnf.predict(X_test, method='pcc')
    prev_pac = qnf.predict(X_test, method='pac')
    prev_hdy = qnf.predict(X_test, method='hdy')

    formatter = "{:<15}{:>15.2f}{:>15.2f}{:>15.2f}"
    print("{:<15}{:>15}{:>15}{:>15}".format('', 'setosa', 'veriscolor', 'virginica'))
    print(formatter.format("True", *prev_true))
    print(formatter.format("CC", *prev_cc))
    print(formatter.format("AC", *prev_ac))
    print(formatter.format("PCC", *prev_pcc))
    print(formatter.format("PAC", *prev_pac))
    print(formatter.format("HDy", *prev_hdy))

    hdx = HDX(b=8)
    hdx.fit(X_train, y_train)
    prev_hdx = hdx.predict(X_test)
    print(formatter.format("HDx", *prev_hdx))

    edx = EDx()
    edx.fit(X_train, y_train)
    prev_edx = edx.predict(X_test)
    print(formatter.format("EDx", *prev_edx))

    edy = EDy()
    edy.fit(X_train, y_train)
    prev_edy = edy.predict(X_test)
    print(formatter.format("EDy", *prev_edy))

    kedx = kEDx(k=3)
    kedx.fit(X_train, y_train)
    prev_kedx = kedx.predict(X_test)
    print(formatter.format("kEDx", *prev_kedx))

    fr = FriedmanMM()
    fr.fit(X_train, y_train)
    prev_fr = fr.predict(X_test)
    print(formatter.format("Friedman-MM", *prev_fr))

    fr = FriedmanAC()
    fr.fit(X_train, y_train)
    prev_fr = fr.predict(X_test)
    print(formatter.format("Friedman-AC", *prev_fr))


if __name__ == '__main__':
    main()