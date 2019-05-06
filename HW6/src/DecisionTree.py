import matplotlib.pyplot as plt
from itertools import product
import numpy as np
from collections import Counter
from sklearn.base import BaseEstimator


class DecisionTree(BaseEstimator):
    def __init__(
        self,
        split_loss_function,
        leaf_value_estimator,
        depth=0,
        min_sample=5,
        max_depth=10,
    ):
        """Initialize the decision tree classifier

        :param split_loss_function: method for splitting node
        :param leaf_value_estimator: method for estimating leaf value
        :param depth: depth indicator, default value is 0, representing root node
        :param min_sample: an internal node can be splitted only if it contains points more than min_smaple
        :param max_depth: restriction of tree depth.
        """
        self.split_loss_function = split_loss_function
        self.leaf_value_estimator = leaf_value_estimator
        self.depth = depth
        self.min_sample = min_sample
        self.max_depth = max_depth

    def fit(self, X, y):
        """Fit the decision tree in place

        This should fit the tree classifier by setting the values:
        self.is_leaf
        self.split_id (the index of the feature we want to split on, if we're splitting)
        self.split_value (the corresponding value of that feature where the split is)
        self.value (the prediction value if the tree is a leaf node).  
        
        If we are splitting the node, we should also init self.left and self.right to 
        be DecisionTree objects corresponding to the left and right subtrees. These 
        subtrees should be fit on the data that fall to the left and right, respectively, 
        of self.split_value. This is a recursive tree building procedure. 
        
        :param X: a numpy array of training data, shape = (n, m)
        :param y: a numpy array of labels, shape = (n,)
        :return: self
        """
        n, m = X.shape
        if self.depth == self.max_depth or n <= self.min_sample:
            self.is_leaf = True
            self.value = self.leaf_value_estimator(y)
        else:
            best_feature, best_score = 0, np.inf
            for feature in range(m):
                values = X[:, feature]
                sort_indices = np.argsort(values)
                sorted_values = values[sort_indices]
                sorted_labels = y[sort_indices]
                _, unique_indexes = np.unique(sorted_values, return_index=True)
                for i in range(unique_indexes):
                    y_l, y_r = sorted_labels[:i], sort_indices[i:]
                    l_split_val = self.split_loss_function(y_l)
                    r_split_val = self.split_loss_function(y_r)
                    node_split_val = i / n * l_split_val + (1 - i / n) * r_split_val

                    if node_split_val < best_score:
                        best_score = node_split_val
                        best_feature = feature
                        best_split_val = sorted_values[i]

            self.split_id = best_feature
            self.split_value = best_split_val
            self.left = DecisionTree(
                self.split_loss_function,
                self.leaf_value_estimator,
                depth=self.depth + 1,
                min_sample=self.min_sample,
                max_depth=self.max_depth,
            )
            self.right = DecisionTree(
                self.split_loss_function,
                self.leaf_value_estimator,
                depth=self.depth + 1,
                min_sample=self.min_sample,
                max_depth=self.max_depth,
            )
            mask = X[:, self.split_value] < self.split_value
            n_l = np.sum(mask)
            n_r = n - n_l
            if n_l < self.min_sample or n_r < self.min_sample:
                self.is_leaf = True
                self.value = self.leaf_value_estimator(y)
            else: 
                self.is_leaf = False
                X_l, y_l = X[mask], y[mask]
                X_r, y_r = X[~mask], y[~mask]
                self.left.fit(X_l, y_r)
                self.right.fit(X_r, y_l)

        return self

    def predict_instance(self, instance):
        """ Predict label by decision tree

        :param instance: numpy array with new data, shape (1, m)
        :return: whatever is returned by leaf_value_estimator for leaf containing instance
        """
        if self.is_leaf:
            return self.value
        if instance[self.split_id] <= self.split_value:
            return self.left.predict_instance(instance)
        else:
            return self.right.predict_instance(instance)
