import matplotlib.pyplot as plt
from itertools import product
import numpy as np
from collections import Counter
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_graphviz
import graphviz


class DecisionTree(BaseEstimator):
     
    def __init__(
        self, 
        split_loss_function, 
        leaf_value_estimator,
        depth=0, 
        min_sample=5, 
        max_depth=10
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


    def fit(self, X, y=None):
        """Fit the decision tree in place

        This should fit the tree classifier by setting the values self.is_leaf, 
        self.split_id (the index of the feature we want to split on, if we"re splitting),
        self.split_value (the corresponding value of that feature where the split is),
        and self.value, which is the prediction value if the tree is a leaf node.  If we are 
        splitting the node, we should also init self.left and self.right to be DecisionTree
        objects corresponding to the left and right subtrees. These subtrees should be fit on
        the data that fall to the left and right,respectively, of self.split_value.
        This is a recursive tree building procedure. 
        
        :param X: a numpy array of training data, shape = (n, m)
        :param y: a numpy array of labels, shape = (n,)
        :return: self
        """
        n, m = X.shape
        if self.depth == self.max_depth or self.min_sample >= n:
            self.is_leaf = True
            # don't split, just return the majority class or the average value of the leaf
        else:
            # here we need to find the best split point for each feature, then the best feature out of those
            for feature in range(m):
                
        # Your code goes here
        
        return self

    def predict_instance(self, instance):
        """
        Predict label by decision tree

        :param instance: a numpy array with new data, shape (1, m)

        :return whatever is returned by leaf_value_estimator for leaf containing instance
        """
        if self.is_leaf:
            return self.value
        if instance[self.split_id] <= self.split_value:
            return self.left.predict_instance(instance)
        else:
            return self.right.predict_instance(instance)