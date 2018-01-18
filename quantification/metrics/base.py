import numpy as np
from sklearn.utils import check_consistent_length, check_array


def check_array_and_consistent_length(p_true, p_pred):
    p_true = check_array(p_true)
    p_pred = check_array(p_pred)
    check_consistent_length(p_true, p_pred)
    return p_true, p_pred


def binary_kl_divergence(p_true, p_pred, eps=1e-12):
    """Also known as discrimination information, relative entropy or normalized cross-entropy
    (see [Esuli and Sebastiani 2010; Forman 2008]). KL Divergence is a special case of the family of f-divergences and
    it can be defined for binary quantification.

    Parameters
    ----------
    p_true : float
        True prevalence.

    p_pred : float
        Predicted prevalence.
    """
    p_true += eps
    p_pred += eps

    kl = p_true * np.log(p_true / p_pred) + (1 - p_true) * np.log((1 - p_true) / (1 - p_pred))
    return kl


def multiclass_kl_divergence(p_true, p_pred):
    """Also known as discrimination information, relative entropy or normalized cross-entropy
        (see [Esuli and Sebastiani 2010; Forman 2008]). KL Divergence is a special case of the family of f-divergences and
        it can be defined for binary quantification.

    Parameters
    ----------
    p_true : array_like, shape=(n_classes)
        True prevalences. In case of binary quantification, this parameter could be a single float value.

    p_pred : array_like, shape=(n_classes)
        Predicted prevalences. In case of binary quantification, this parameter could be a single float value.
    """
    p_true, p_pred = check_array_and_consistent_length(p_true, p_pred)
    if np.any(p_pred == 0):
        raise NotImplementedError
    return np.sum(p_true * np.log(p_true / p_pred))


def bias(p_true, p_pred):
    """Measures when a binary quantifier tends to overestimate or underestimate the proportion of positives

    Parameters
    ----------
    p_true : array_like, shape=(n_classes)
        True prevalences. In case of binary quantification, this parameter could be a single float value.

    p_pred : array_like, shape=(n_classes)
        Predicted prevalences. In case of binary quantification, this parameter could be a single float value.
    """
    p_true, p_pred = check_array_and_consistent_length(p_true, p_pred)
    return p_pred - p_true


def absolute_error(p_true, p_pred):
    """Just the absolute difference between both prevalences.

        Parameters
        ----------
        p_true : array_like, shape=(n_classes)
            True prevalences. In case of binary quantification, this parameter could be a single float value.

        p_pred : array_like, shape=(n_classes)
            Predicted prevalences. In case of binary quantification, this parameter could be a single float value.
        """
    return np.abs(p_pred - p_true)


def square_error(p_true, p_pred):
    """Penalizes large mistakes

    Parameters
    ----------
    p_true : array_like, shape=(n_classes)
        True prevalences. In case of binary quantification, this parameter could be a single float value.

    p_pred : array_like, shape=(n_classes)
        Predicted prevalences. In case of binary quantification, this parameter could be a single float value.
    """
    return np.power(p_pred - p_true, 2)


def relative_absolute_error(p_true, p_pred):
    """The relation between the absolute error and the true prevalence.

        Parameters
        ----------
        p_true : array_like, shape=(n_classes)
            True prevalences. In case of binary quantification, this parameter could be a single float value.

        p_pred : array_like, shape=(n_classes)
            Predicted prevalences. In case of binary quantification, this parameter could be a single float value.
        """
    p_true, p_pred = check_array_and_consistent_length(p_true, p_pred)
    if np.any(p_true == 0):
        raise NotImplementedError
    return np.mean(np.abs(p_pred - p_true) / p_true)


def symmetric_absolute_percentage_error(p_true, p_pred):
    """SAPE. A symmetric version of RAE.

        Parameters
        ----------
        p_true : array_like, shape=(n_classes)
            True prevalences. In case of binary quantification, this parameter could be a single float value.

        p_pred : array_like, shape=(n_classes)
            Predicted prevalences. In case of binary quantification, this parameter could be a single float value.
        """
    p_true, p_pred = check_array_and_consistent_length(p_true, p_pred)
    if np.any((p_pred + p_true) == 0):
        raise NotImplementedError
    return np.abs(p_pred - p_true) / (p_pred + p_true)


def normalized_absolute_score(p_true, p_pred):
    """NAS.

        Parameters
        ----------
        p_true : array_like, shape=(n_classes)
            True prevalences. In case of binary quantification, this parameter could be a single float value.

        p_pred : array_like, shape=(n_classes)
            Predicted prevalences. In case of binary quantification, this parameter could be a single float value.
        """
    p_true, p_pred = check_array_and_consistent_length(p_true, p_pred)
    return 1 - np.abs(p_pred - p_true) / np.max([p_true, 1 - p_true])


def normalized_absolute_error(p_true, p_pred):
    """A loss function, ranging from 0 (best) and 1 (worst)

        Parameters
        ----------
        p_true : array_like, shape=(n_classes)
            True prevalences. In case of binary quantification, this parameter could be a single float value.

        p_pred : array_like, shape=(n_classes)
            Predicted prevalences. In case of binary quantification, this parameter could be a single float value.
        """
    p_true, p_pred = check_array_and_consistent_length(p_true, p_pred)
    return np.sum(np.abs(p_pred - p_true)) / (2 * (1 - np.min(p_true)))


def normalized_square_score(p_true, p_pred):
    """NSS.

        Parameters
        ----------
        p_true : array_like, shape=(n_classes)
            True prevalences. In case of binary quantification, this parameter could be a single float value.

        p_pred : array_like, shape=(n_classes)
            Predicted prevalences. In case of binary quantification, this parameter could be a single float value.
        """
    p_true, p_pred = check_array_and_consistent_length(p_true, p_pred)
    return 1 - np.power(np.abs(p_pred - p_true) / np.max([p_true, 1 - p_true]), 2)


def normalized_relative_absolute_error(p_true, p_pred):
    """NRAE.

        Parameters
        ----------
        p_true : array_like, shape=(n_classes)
            True prevalences. In case of binary quantification, this parameter could be a single float value.

        p_pred : array_like, shape=(n_classes)
            Predicted prevalences. In case of binary quantification, this parameter could be a single float value.
        """
    p_true, p_pred = check_array_and_consistent_length(p_true, p_pred)
    l = p_true.shape[0]
    return relative_absolute_error(p_true, p_pred) / (l - 1 + (1 - np.min(p_true)) / np.min(p_true))


def bray_curtis(p_true, p_pred):
    """Bray-Curtis dissimilarity.

        Parameters
        ----------
        p_true : array_like, shape=(n_classes)
            True prevalences. In case of binary quantification, this parameter could be a single float value.

        p_pred : array_like, shape=(n_classes)
            Predicted prevalences. In case of binary quantification, this parameter could be a single float value.
        """
    p_true, p_pred = check_array_and_consistent_length(p_true, p_pred)
    return np.sum(np.abs(p_true - p_pred)) / np.sum(p_true + p_pred)
