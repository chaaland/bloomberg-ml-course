import numpy as np
from DecisionTree import DecisionTree
from utils import mean_absolute_deviation_around_median


class RegressionTree:
    """
    :attribute loss_function_dict: dictionary containing the loss functions used for splitting
    :attribute estimator_dict: dictionary containing the estimation functions used in leaf nodes
    """

    loss_function_dict = {"mse": np.var, "mae": mean_absolute_deviation_around_median}

    estimator_dict = {"mean": np.mean, "median": np.median}

    def __init__(
        self, loss_function="mse", estimator="mean", min_sample=5, max_depth=10
    ):
        """
        Initialize Regression_Tree
        :param loss_function(str): loss function used for splitting internal nodes
        :param estimator(str): value estimator of internal node
        """

        self.tree = DecisionTree(
            self.loss_function_dict[loss_function],
            self.estimator_dict[estimator],
            0,
            min_sample,
            max_depth,
        )

    def fit(self, X, y=None):
        self.tree.fit(X, y)
        return self

    def predict_instance(self, instance):
        value = self.tree.predict_instance(instance)
        return value
