import matplotlib.pyplot as plt
from itertools import product
import numpy as np
from collections import Counter
from sklearn.base import BaseEstimator, ClassifierMixin

from DecisionTree import DecisionTree
from utils import compute_entropy, compute_gini, most_common_label


class ClassificationTree(BaseEstimator):

    loss_function_dict = {"entropy": compute_entropy, "gini": compute_gini}

    def __init__(self, loss_function="entropy", min_sample=5, max_depth=10):
        """
        :param loss_function(str): loss function for splitting internal node
        :param min_sample: minimum number of data points in a leaf
        :param max_depth: maximum depth the tree can grow
        """

        self.tree = DecisionTree(
            self.loss_function_dict[loss_function],
            most_common_label,
            0,
            min_sample,
            max_depth,
        )

    def fit(self, X, y):
        self.tree.fit(X, y)
        return self

    def predict_instance(self, instance):
        value = self.tree.predict_instance(instance)
        return value
