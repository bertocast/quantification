from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression

import numpy as np

from quantification.cc import BaseBinaryCC


def main():
    # Load the data
    X, y = load_breast_cancer(return_X_y=True)

    # Create de quantifier. We will use a simple logistic regression as the underlying classifier.
    # Number of bins to perform HDy will be set to 8.
    qnf = BaseBinaryCC(estimator_class=LogisticRegression(), b=8)

    # Fit the quantifier
    qnf.fit(X, y)

    # Get the true binary prevalence, i.e., percentage of positives samples.
    prev_true = np.bincount(y, minlength=2) / float(len(y))
    prev_true = prev_true[1]

    # Get the predicted prevalences using every of the methods available in the BaseBinaryCC
    prev_cc = qnf.predict(X, method='cc')[1]
    prev_ac = qnf.predict(X, method='ac')[1]
    prev_pcc = qnf.predict(X, method='pcc')[1]
    prev_pac = qnf.predict(X, method='pac')[1]
    prev_hdy= qnf.predict(X, method='hdy')[1]

    formatter = "{:<4}{:>15.2f}"
    print(formatter.format("True", prev_true))
    print(formatter.format("CC", prev_cc))
    print(formatter.format("AC", prev_ac))
    print(formatter.format("PCC", prev_pcc))
    print(formatter.format("PAC", prev_pac))
    print(formatter.format("HDy", prev_hdy))


if __name__ == '__main__':
    main()
