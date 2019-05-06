from collections import Counter
import numpy as np

def compute_entropy(label_array):
    """Calulate the entropy of given label list
    
    :param label_array: a numpy array of labels shape = (n, 1)
    :return: entropy value
    """
    n_elems = label_array.size
    _, elem_counts = np.unique(label_array, return_counts=True)
    probs = elem_counts / n_elems

    return -probs * np.log2(probs + 1e-7)


def compute_gini(label_array):
    """Calulate the gini index of label list
    
    :param label_array: a numpy array of labels shape = (n,)
    :return: gini index value
    """
    n_elems = label_array.size
    _, elem_counts = np.unique(label_array, return_counts=True)
    probs = elem_counts / n_elems

    return 1 - np.sum(probs)


def most_common_label(y):
    """Find most common label

    :param y: numpy array of labels shape=(n,)
    :return: most frequent label
    """
    label_cnt = Counter(y)
    most_common_label, _ = label_cnt.most_common(1)[0]

    return most_common_label


def mean_absolute_deviation_around_median(y):
    """Calulate the mean absolute deviation around the median of a given target list
    
    :param y: a numpy array of targets shape = (n, 1)
    :return: mean absolute deviation from the median
    """
    m = np.median(y)
    y_centered = y - m
    mae = np.mean(np.abs(y_centered))

    return mae

def pseudo_residual_L2(train_target, train_predict):
    """Compute the pseudo-residual for half the L2 norm based on 
    current predicted value

    :param train_target:
    :param train_predict:
    :return: pseudo-residual of the l2 norm
    """
    return train_target - train_predict