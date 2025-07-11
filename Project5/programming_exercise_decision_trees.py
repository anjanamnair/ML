#!/usr/bin/env python3
import numpy as np
import pandas as pd
from typing import Callable, Optional, Tuple
########################################################################
# Starter code for exercise 6: Argument quality prediction with CART decision trees
########################################################################

GROUP = '5' # TODO: Your group number here


def load_feature_vectors(filename: str) -> np.array:
    """
    Load the feature vectors from the dataset in the given file and return
    them as a numpy array with shape (number-of-examples, number-of-features, 1).
    """
    # features = pd.read_csv(filename, sep='\t', usecols=["#id", "chars_count"])
    features = pd.read_csv(filename, sep='\t')
    features = features.select_dtypes(include=np.int64) # keep only numerical values
    feature_names = features.columns
    xs = features.loc[:, feature_names].values
    xs = xs.reshape(xs.shape[0], xs.shape[-1], 1)
    xs = np.concatenate([np.ones([xs.shape[0], 1, 1]), xs], axis=1)
    return xs

# From last exercise sheet
def load_class_values(filename: str) -> np.array:
    """
    Load the class values for overall quality (class 0 for quality 1 and class 1
    for overall quality 2 or 3) from the dataset in the given file and return
    them as a one-dimensional numpy array.
    """
    return np.ravel((pd.read_csv(filename, sep='\t', usecols=["overall quality"]).to_numpy() > 1) * 1)

def most_common_class(cs: np.array):
    """Return the most common class value in the given array

    Arguments:
    - cs: a 1-dimensional array of length n, containing of the class values c(x) for
          every element x of a dataset D
    """
    # TODO: Your code here
    most_common, counts = np.unique(cs, return_counts=True)
    result = most_common[np.argmax(counts)]
    print("Most occurring class value :", result)
    return result

def gini_impurity(cs: np.array) -> float:
    """Compute the Gini index for a set of examples represented by the list of
    class values

    Arguments:
    - cs: a 1-dimensional array of length n, containing of the class values c(x) for
          every element x of a dataset D
    """
    # TODO: Your code here
    total_num_classes = len(cs)
    most_common, counts = np.unique(cs, return_counts=True)
    probablities = counts/total_num_classes
    result = 1- np.sum(probablities**2)
    return result



def gini_impurity_reduction(impurity_D: float, cs_l: np.array, cs_r: np.array) -> float:
    """Compute the Gini impurity reduction of a binary split.

    Arguments:
    - impurity_D: the Gini impurity of the entire document D set to be split
    - cs_l: an array with the class values of the examples in the left split
    - cs_r: an array with the class values of the examples in the right split
    """
    size_D = len(cs_l) + len(cs_r)
    # TODO: Your code here
    impurity_l = gini_impurity(cs_l)
    impurity_r =  gini_impurity(cs_r)
    result = impurity_D - ((len(cs_l)/size_D)*impurity_l + (len(cs_r)/size_D)*impurity_r)
    return result

def possible_thresholds(xs: np.array, feature: int) -> np.array:
    """Compute all possible thresholds for splitting the example set xs along
    the given feature. Pick thresholds as the mid-point between all pairs of
    distinct, consecutive values in ascending order.

    Arguments:
    - xs: an array of shape (n, p, 1)
    - feature: an integer with 0 <= a < p, giving the feature to be used for splitting xs
    """
    # TODO: Your code here
    threshold_values = np.sort(np.unique(xs[:,feature]))
    return (threshold_values[:-1]+threshold_values[1:])/2

def find_split_indexes(xs: np.array, feature: int, threshold: float) -> Tuple[np.array, np.array]:
    """Split the given dataset using the provided feature and threshold.

    Arguments:
    - xs: an array of shape (n, p, 1)
    - feature: an integer with 0 <= a < p, giving the feature to be used for splitting xs
    - threshold: the threshold to be used for splitting (xs, cs) along the given feature

    Returns:
    - left: a 1-dimensional integer array, length <= n
    - right: a 1-dimensional integer array, length <= n
    """
    # This function is provided for you.
    smaller = (xs[:, feature, :] < threshold).flatten()
    bigger = ~smaller # element-wise negation

    idx = np.arange(xs.shape[0])

    return idx[smaller], idx[bigger]

def find_best_split(xs: np.array, cs: np.array) -> Tuple[int, float]:
    """
    Find the best split point for the dataset (xs, cs) from among the given
    possible feature indexes, as determined by the Gini index.

    Arguments:
    - xs: an array of shape (n, p, 1)
    - cs: a 1-dimensional array of length n

    Returns:
    - the feature index of the best split
    - the threshold value of the best split
    """
    # hints to start
    a_best = None
    threshold_best = None
    gini_reduction_best = 0
    gini_all = gini_impurity(cs) # impurity of the example set D
    features = np.arange(xs.shape[1]) # features available for splitting
    for a_i in features:
        for threshold in possible_thresholds(xs, a_i):
            l, r = find_split_indexes(xs,a_i,threshold)
            gini_red = gini_impurity_reduction(gini_all,cs[l],cs[r])
            if gini_red>gini_reduction_best:
                gini_reduction_best=gini_red
                a_best=a_i
                threshold_best=threshold
    return a_best,threshold_best       
            

def misclassification_rate(cs: np.array, ys: np.array) -> float:
    """
    This function takes two vectors with gold and predicted labels and
    returns the percentage of positions where truth and prediction disagree
    """
    if len(cs) == 0:
        return float('nan')
    else:
        return 1 - (np.sum(np.equal(cs, ys)) / len(cs))

class CARTNode:
    """A node in a CART decision tree
    """
    def __init__(self):
        self.left = None
        self.right = None
        self.feature = None
        self.threshold = None
        self.label = None

    def set_label(self, label):
        self.label = label

    def set_split(self, feature: int, threshold: float, left: 'CARTNode', right: 'CARTNode'):
        """Turn this node into an internal node splitting at the given feature
        and threshold, with the given left and right subtrees.
        """
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right

    def classify(self, x: np.array):
        """Return the class value for the given example as predicted by this subtree

        Arguments:
        - x: an array of shape (p, 1)
        """
        # This method is provided for you
        if self.feature is None:
            # this is a leaf node
            return self.label

        v = x[self.feature]

        if v < self.threshold:
            return self.left.classify(x)
        else:
            return self.right.classify(x)

    def __repr__(self):
        return f"[label={self.label};{self.feature}|{self.threshold};L={self.left};R={self.right}]"


def id3_cart(xs: np.array, cs: np.array, max_depth: int=10) -> CARTNode:
    """Construct a CART decision tree with the modified ID3 algorithm.

    Arguments:
    - xs: an array of shape (n, p, 1)
    - cs: a 1-dimensional array of length n
    - max_depth: limit to the size of the tree that is constructed; unlimited if None

    Returns:
    - the root node of the constructed decision tree
    """
    # TODO: Your code here
    root = CARTNode()

    if len(np.unique(cs)) == 1 or (max_depth is not None and max_depth == 0):
        root.set_label(cs[0])
    else:
        criterion = find_best_split(xs, cs)
        feature, threshold = criterion
        i_l, i_r = find_split_indexes(xs, feature, threshold)
        D_splits = [(xs[i_l], cs[i_l]), (xs[i_r], cs[i_r])]

        root.set_split(feature, threshold, None, None)

        for i in range(2):
            successor = id3_cart(D_splits[i][0], D_splits[i][1], None if max_depth is None else max_depth - 1)
            if i == 0:
                root.left = successor
            else:
                root.right = successor

    return root

class CARTModel:
    """Trivial model interface class for the CART decision tree.
    """
    def __init__(self, max_depth=None):
        self._t = None # root of the decision tree
        self._max_depth = max_depth

    def fit(self, xs: np.array, cs: np.array):
        self._t = id3_cart(xs, cs, self._max_depth)

    def predict(self, x):
        return self._t.classify(x)


def train_and_predict(training_features_file_name: str,
                      training_classes_file_name: str,
                      test_features_file_name: str) -> np.array:
    """Train a model on the given training dataset, and predict the class values
    for the given testing dataset.

    Return an array with the predicted class values, in the same order as the
    examples in the testing dataset.
    """
    
    train_Feature = load_feature_vectors(training_features_file_name)
    train_Class = load_class_values(training_classes_file_name)
    test_Feature = load_feature_vectors(test_features_file_name)
    tree = CARTModel()
    tree.fit(train_Feature, train_Class)
    predictions = [tree.predict(x) for x in test_Feature]
    train_predictions = [tree.predict(x) for x in train_Feature]
    misclassification_rate_train = misclassification_rate(train_Class, train_predictions)
    print(f"Misclassification Rate on Training Set: {misclassification_rate_train}")
    return np.array(predictions)


########################################################################
# Tests
import os
from pytest import approx

def test_most_common_class():
    cs = np.array(['red', 'green', 'green', 'blue', 'green'])
    assert most_common_class(cs) == 'green', \
        "Identify the correct most common class"

def test_gini_impurity():
    # should work with two classes
    cs = np.array(['a', 'a', 'b', 'a'])
    assert gini_impurity(cs) == approx(2*0.75*0.25), \
        "Compute the correct Gini index for a two-class dataset"

    # should also work with more classes
    cs = np.array(['a', 'b', 'c', 'b', 'a'])
    assert gini_impurity(cs) == approx(1 - (0.4**2 + 0.4**2 + 0.2**2)), \
        "Compute the correct Gini index for a three-class dataset"

def test_gini_impurity_reduction():
    # cs = np.array(['a', 'a', 'b', 'a'])
    i_D = 0.375

    assert gini_impurity_reduction(i_D, np.array(['a', 'a']), np.array(['b', 'a'])) == approx(0.125), \
        "Compute the correct gini reduction for the first test split"

    assert gini_impurity_reduction(i_D, np.array(['a', 'a', 'a']), np.array(['b'])) == approx(0.375), \
        "Compute the correct gini reduction for the second test split"


def test_possible_thresholds():
    xs = np.array([
        [[1],   [0]],
        [[0.5], [1]],
        [[0],   [0]],
        [[1],   [1]],
    ])
    # first feature allows two possible split points
    assert possible_thresholds(xs, 0) == approx(np.array([0.25, 0.75])), \
        "Find all possible thresholds for the first feature."

    # second feature only one
    assert possible_thresholds(xs, 1) == approx(np.array([0.5])), \
        "Find all possible thresholds for the second feature"

def test_find_split_indexes():
    xs = np.array([
        [[1],   [0]],
        [[0.5], [1]],
        [[0],   [0]],
        [[1],   [1]],
    ])
    l, r = find_split_indexes(xs, 0, 0.75)
    assert all(l == np.array([1, 2])) and all(r == np.array([0, 3]))

    l, r = find_split_indexes(xs, 0, 0.25)
    assert all(l == np.array([2])) and all(r == np.array([0, 1, 3]))

def test_find_best_split():
    xs = np.array([
        [[1],   [0]],
        [[0.5], [1]],
        [[0],   [0]],
        [[1],   [1]],
    ])
    cs = np.array(['a', 'a', 'c', 'a'])
    a, t = find_best_split(xs, cs)
    assert a == 0, "Choose the best feature."
    assert t == 0.25, "Choose the best threshold."

def test_cart_model():
    xs = np.array([
        [[1],   [0]],
        [[0.5], [1]],
        [[0],   [0]],
        [[1],   [1]],
    ])
    cs = np.array(['a', 'a', 'b', 'b'])
    tree = CARTModel()
    tree.fit(xs, cs)
    preds = [tree.predict(x) for x in xs]

    assert all(cs == preds), \
        "On a dataset without label noise, reach zero training error."


if __name__ == "__main__":
    import pandas as pd
    import pytest
    import sys

    train_features_file_name = sys.argv[1]
    train_classes_file_name = sys.argv[2]
    test_features_file_name = sys.argv[3]

    test_result = pytest.main(['--tb=short', __file__])
    if test_result != 0:
        sys.exit(test_result)
    print("Great! All tests passed!")
    print("Running train_and_predict function...")
    predictions = train_and_predict(train_features_file_name, train_classes_file_name, test_features_file_name)
    print("Writing predictions to file...")
    pd.DataFrame(predictions).to_csv(f"argument-clarity-predictions-mlp-group-{GROUP}.tsv", header=False, index=False)
