# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 13:34:28 2013

Least Squares Probabilistic Classifier

Reference:
Sugiyama, M. 
"Superfast-trainable multi-class probabilistic classifier by 
least-squares posterior fitting", IEICE Transactions on Information and
Systems, vol.E93-D, no.10, pp.2690-2701, 2010. 
http://sugiyama-www.cs.titech.ac.jp/~sugi/2010/LSPC.pdf

@author: John Quinn <jquinn@cit.ac.ug>
"""

from sklearn import metrics, base, neighbors
import numpy as np


def median_kneighbour_distance(X, k=5):
    """Calculate the median distance between a set of random datapoints and
    their kth nearest neighbours. This is a heuristic for setting the
    kernel length scale."""
    N_all = X.shape[0]
    N_subset = min(N_all, 2000)
    sample_idx_train = np.random.permutation(N_all)[:N_subset]
    nn = neighbors.NearestNeighbors(k, warn_on_equidistant=False)
    nn.fit(X[sample_idx_train, :])
    d, idx = nn.kneighbors(X[sample_idx_train, :])
    return np.median(d[:, -1])


def pair_distance_centiles(X, centiles, max_pairs=2000):
    """Helper function to find the centiles of pairwise distances in a
    dataset. This can be useful information to help choose candidates for the
    kernel parameter sigma.

    Parameters
    ----------

    X : array 
        Should have dimension N times d (rows are examples, columns are data
        dimensions)
    centiles : array
        List of centiles to calculate, where all values are between 0 and 100
    max_pairs : int
        Maximum number of random pairs to evaluate.

    Returns
    -------
    c : array
        Calculated centiles of pairwise distances, of the same size as the 
        input parameter "centiles".
    """
    N = X.shape[0]
    n_pairs = min(max_pairs, N)
    n_centiles = len(centiles)
    randorder1 = np.random.permutation(N)
    randorder2 = np.random.permutation(N)

    dists = np.zeros(n_pairs)

    for i in range(n_pairs):
        pairdiff = X[randorder1[i], :] - X[randorder2[i], :]
        dists[i] = np.dot(pairdiff.T, pairdiff)
    dists.sort()

    out = np.zeros(n_centiles)
    for i in range(n_centiles):
        out[i] = dists[int(n_pairs * centiles[i] / 100.)]

    return np.sqrt(out)


class LSPC(base.BaseEstimator):
    """Least Squares Probabilistic Classifier.

    Parameters
    ----------
    n_kernels_max : int, optional (default 5000)
        Maximum number of kernel basis centres to use.
    kernel_pos : optional
        The positions of the kernel centres can be specified. Default is to
        select them as a random subset of the training data.
    basis_set: 'full' or 'classwise' (default: 'classwise')
        If set to 'full', use training points from all classes as the kernel
        basis set. If 'classwise', use a separate set of kernel bases for
        each class, using training points from only that class (much faster).

    Example
    -------

    >>> from quantification.utils.models import LSPC, pair_distance_centiles
    >>> from sklearn import datasets, model_selection, metrics
    >>> iris = datasets.load_iris()
    >>> X = iris.data
    >>> y = iris.target
    >>> sigma_candidates = pair_distance_centiles(X,
                            centiles=[10, 50, 90]) 
    >>> rho_candidates = [.01, .1, .5, 1.]    
    >>> param_grid = {'sigma':sigma_candidates, 'rho':rho_candidates}
    >>> cv_folds = 5
    >>> clf = model_selection.GridSearchCV(LSPC(),
    ...                                param_grid, cv=cv_folds,
    ...                                score_func=metrics.accuracy_score,
    ...                                n_jobs=-1)  
    >>> kf = model_selection.KFold(n=X.shape[0], n_folds=3, shuffle=True)
    >>> predictions= np.zeros(X.shape[0])
    >>> for train_index, test_index in kf:
    ...     clf.fit(X[train_index,:], y[train_index])
    ...     predictions[test_index] = clf.predict(X[test_index])
    """

    def __init__(self, n_kernels_max=7000, kernel_pos=None,
                 basis_set='classwise', sigma=None, gamma=None, rho=None):
        self.n_kernels_max = n_kernels_max
        self.kernel_pos = kernel_pos
        self.rho = None
        self.theta = None
        self.basis_classes = None
        self.basis_set = basis_set
        self.sigma = None
        self.gamma = None

    def fit(self, X, y, rho=.01, sigma=None, gamma=None):
        """Fit the LSPC model according to the given training data.

        Parameters
        ----------
        X : array, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target class labels
        sigma : float, optional
            Kernel scale parameter.
        gamma : float, optional
            An alternative way of specifying the kernel scale parameter,
            for compatibility with SVM notation. sigma = 1/sqrt(gamma)
        rho : float, optional
            Regularization parameter.
        """

        self.classes_ = np.unique(y)
        self.classes_.sort()
        self.n_classes = len(self.classes_)

        N = X.shape[0]

        # If no kernel parameters have been specified, try to choose some
        # reasonable defaults.      
        if sigma:
            self.sigma = sigma
            self.gamma = sigma ** -2
        if gamma:
            self.gamma = gamma
            self.sigma = gamma ** -.5
        if not self.sigma:
            neighbour_dist = median_kneighbour_distance(X)
            # median_dist = pair_distance_centiles(X,[50])[0]
            self.sigma = neighbour_dist
        if not self.gamma:
            self.gamma = self.sigma ** -2
        if not self.rho:
            self.rho = 0.1

        # choose kernel basis centres
        if self.kernel_pos == None:
            B = min(self.n_kernels_max, N)
            kernel_idx = np.random.permutation(N)
            self.kernel_pos = X[kernel_idx[:B], :]
            self.basis_classes = y[kernel_idx[:B]]
        else:
            B = self.kernel_pos.shape[0]

        # fit coefficients
        Phi = metrics.pairwise.rbf_kernel(X, self.kernel_pos, self.gamma)
        theta = {}
        '''
        for c in self.classes:
            m = (y==c).astype(int)
            if self.basis_set=='full':
                kidx = np.ones(Phi.shape[1]).astype('bool')
            else:
                kidx = self.basis_classes==c
            theta[c] = np.linalg.inv(np.dot(Phi[:,kidx].T,Phi[:,kidx]) 
                        + self.rho*np.eye(sum(kidx)))
            theta[c] = np.dot(theta[c],np.dot(Phi[:,kidx].T,m))   

        self.theta = theta
        '''

        if self.basis_set == 'full':
            inv_part = np.linalg.inv(np.dot(Phi.T, Phi)
                                     + self.rho * np.eye(B))

        for c in self.classes_:
            m = (y == c).astype(int)
            if self.basis_set == 'full':
                kidx = np.ones(Phi.shape[1]).astype('bool')
            else:
                kidx = self.basis_classes == c
                inv_part = np.linalg.inv(np.dot(Phi[:, kidx].T, Phi[:, kidx])
                                         + self.rho * np.eye(sum(kidx)))

            theta[c] = np.dot(inv_part, np.dot(Phi[:, kidx].T, m))

        self.theta = theta

    def predict(self, X):
        """Perform classification on samples in X, and return most likely
        classes for each test instance.

        Parameters
        ----------
        X : array, shape = [n_samples, n_features]

        Returns
        -------
        y_pred : array, shape = [n_samples]
            Class labels for samples in X.
        """
        predictions_proba = self.predict_proba(X)
        predictions = []
        for i in range(X.shape[0]):
            predictions.append(self.classes_[predictions_proba[i, :].argmax()])
        return predictions

    def predict_proba(self, X):
        """Compute probabilities of possible outcomes for samples in X.

        Parameters
        ----------
        X : array, shape = [n_samples, n_features]

        Returns
        -------
        Y_prob : array, shape = [n_samples, n_classes]
            Returns the probability of the sample for each class in
            the model.
        """
        Phi = metrics.pairwise.rbf_kernel(X, self.kernel_pos, self.gamma)
        N = X.shape[0]
        predictions = np.zeros((N, self.n_classes))
        for i in range(N):
            post = np.zeros(self.n_classes)
            for c in range(self.n_classes):
                if self.basis_set == 'full':
                    kidx = np.ones(Phi.shape[1]).astype('bool')
                else:
                    kidx = self.basis_classes == self.classes_[c]
                post[c] = max(0, np.dot(self.theta[self.classes_[c]].T,
                                        Phi[i, kidx]))
            post = post / sum(post)
            predictions[i, :] = post
        return predictions

