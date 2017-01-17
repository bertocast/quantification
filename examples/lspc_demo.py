# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 13:34:28 2013

Example usage of the LSPC classifier.

This is a modified version of the scikit-learn classification demo at:
http://scikit-learn.org/dev/auto_examples/plot_classification_probability.html

@author: John Quinn <jquinn@cit.ac.ug>
"""
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import brier_score_loss


from quantification.utils.models import LSPC, pair_distance_centiles
from sklearn import datasets
import numpy as np
import seaborn as sns
import matplotlib .pyplot as plt
import pandas as pd
from scipy.io import loadmat

df = loadmat('/Users/albertocastano/Dropbox/Tesis-Alberto/datasets/mammographic')

X = df['x']
y = np.array(df['y'].T.tolist()[0])


# We choose candidates for the kernel length scale based on distances
# between pairs of instances in the training data
sigma_candidates = pair_distance_centiles(X, centiles=[10, 50, 90])
rho_candidates = [.01, .1, .5, 1.]

# Now cross-validate to find the best parameters, and train on all data
# using those parameters. Uses multi-threading by default, change to
# n_jobs=1 to disable this.
param_grid = {'sigma':sigma_candidates, 'rho':rho_candidates}
cv_folds = 5
classifier = GridSearchCV(LSPC(),
                               param_grid, cv=cv_folds,
                               scoring='roc_auc', verbose=11)
classifier.fit(X, y)

print classifier.best_score_

clf = classifier.best_estimator_

from sklearn.base import clone
clf2 = clone(clf)

probs = classifier.predict_proba(X)


sns.distplot(probs[:,1])

plt.show()