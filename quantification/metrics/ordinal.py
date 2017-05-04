from sklearn.utils import check_consistent_length


def emd(p_true, p_pred):
    check_consistent_length(p_true, p_pred)
    return sum([abs(sum(p_pred[:j+1]) - sum(p_true[:j+1])) for j in range(len(p_true)-1)])

