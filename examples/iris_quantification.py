from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

import numpy as np

from quantification.cc import BaseMulticlassCC


def main():
    # Load the data
    X, y = load_iris(return_X_y=True)

    # Create de quantifier. We will use a simple logistic regression as the underlying classifier.
    # Number of bins to perform HDy will be set to 8.
    qnf = BaseMulticlassCC(estimator_class=LogisticRegression(), b=8)

    # Fit the quantifier
    qnf.fit(X, y)

    # Get the true prevalence, i.e., percentage of positives samples in each of the classes.
    prev_true = np.bincount(y, minlength=3) / float(len(y))

    # Get the predicted prevalences using every of the methods available in the BaseMulticlassCC
    prev_cc = qnf.predict(X, method='cc')
    prev_ac = qnf.predict(X, method='ac')
    prev_pcc = qnf.predict(X, method='pcc')
    prev_pac = qnf.predict(X, method='pac')
    prev_hdy= qnf.predict(X, method='hdy')

    formatter = "{:<4}{:>15.2f}{:>15.2f}{:>15.2f}"
    print("{:<4}{:>15}{:>15}{:>15}".format('', 'setosa', 'veriscolor', 'virginica'))
    print(formatter.format("True", *prev_true))
    print(formatter.format("CC", *prev_cc))
    print(formatter.format("AC", *prev_ac))
    print(formatter.format("PCC", *prev_pcc))
    print(formatter.format("PAC", *prev_pac))
    print(formatter.format("HDy", *prev_hdy))


if __name__ == '__main__':
    main()
