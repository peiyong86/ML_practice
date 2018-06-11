#!/usr/bin/env python
from .ops import cal_max_gain, most_common, get_subset
from .data_types import TreeNode
import numpy as np

__author__ = "Yong Pei"


class TreeC3:
    def __init__(self):
        self.__name__ = 'C3'
        self.tree_node = None

    def fit(self, X, Y, weights=None):
        def build_tree(x, y, weights):
            # if all labels are same, return
            labels = set(y)
            if len(labels) == 1:
                return TreeNode(list(labels)[0])
            size = len(y)
            # if set size < 50, return
            if size <= 50:
                return TreeNode(most_common(y))
            # pick feature
            fea_num = x.shape[1]
            info_gains = []
            for i in range(fea_num):
                info_gain = cal_max_gain(x, y, i, weights)
                info_gains.append(info_gain)
            picked = max(info_gains, key=lambda x: x[0])
            picked_a = info_gains.index(picked)
            t = picked[1]
            # build tree
            node = TreeNode({'feature': picked_a, 't': t})
            left_x, left_y, right_x, right_y, left_weights, right_weights = \
                get_subset(x, y, picked_a, t, weights)
            node.left = build_tree(left_x, left_y, left_weights)
            node.right = build_tree(right_x, right_y, right_weights)
            return node
        if weights is None:
            weights = np.array([1.0]*len(X))
        assert len(weights) == len(X)
        self.tree_node = build_tree(X, Y, weights)

    def display(self):
        if not self.tree_node:
            print("Tree not trained.")
            return
        stack = [self.tree_node]
        tree_values = []
        while stack:
            values = [node.val for node in stack]
            tree_values.append(values)
            stack = [child for node in stack
                     for child in [node.left, node.right]
                     if child is not None]
        print(tree_values)

    def inference(self, node, x):
        val = node.val
        if isinstance(val, float):
            return val
        else:
            a = val['feature']
            t = val['t']
            if x[a] > t:
                return self.inference(node.right, x)
            else:
                return self.inference(node.left, x)

    def predict(self, X):
        pred = []
        for x in X:
            pred.append(self.inference(self.tree_node, x))
        return pred
