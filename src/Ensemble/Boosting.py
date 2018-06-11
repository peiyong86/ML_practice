#!/usr/bin/env python
from ..DecisionTrees.trees import TreeC3
import numpy as np

__author__ = "Yong Pei"


class AdaBoost:
    def __init__(self):
        self.__name__ = 'AdaBoost'
        self.alpha = []
        self.classifiers = []

    def fit(self, x, y, n_estimators=50):
        assert len(x) == len(y)
        sample_num = len(y)
        weights = np.ones(sample_num)/sample_num
        for i in range(n_estimators):
            tree = TreeC3()
            tree.fit(x, y, weights)
            pred = tree.predict(x)
            error = np.dot(pred != y, weights)
            if error > 0.5:
                break
            alpha = np.log((1-error)/error)/2
            self.alpha.append(alpha)
            self.classifiers.append(tree)
            # update weights
            scale = np.ones(len(weights)) * error
            scale[pred==y] *= -1
            scale = np.exp(scale)
            weights = np.multiply(weights, scale)
            weights /= sum(weights)

    def predict(self, x):
        scores = []
        for cls, w in zip(self.classifiers, self.alpha):
            scores.append(w * np.array(cls.predict(x)))
        scores_final = np.sum(np.array(scores), axis=0)
        return [1 if s > 0 else -1 for s in scores_final]

