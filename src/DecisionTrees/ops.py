#!/usr/bin/env python
import numpy as np
from collections import defaultdict

__author__ = "Yong Pei"


def cal_entropy(y, w):
    labels = set(y)
    #size = len(y)
    total_w = sum(w)
    entropy = 0
    for la in labels:
        p = sum(w[y==la])/total_w
        entropy -= p * np.log(p)
    return entropy


def cal_gain(feature, y, t, weights):
    indexes = [feature > t, feature <= t]
    subsets = [y[ind] for ind in indexes]
    subsets_weights = [weights[ind] for ind in indexes]
    # size = len(y)
    entropy_split = 0
    total_w = sum(weights)
    for sub_y, sub_w in zip(subsets, subsets_weights):
        # w = len(sub_y)/size
        w = sum(sub_w)/total_w
        entropy_split += w * cal_entropy(sub_y, sub_w)
    entropy_original = cal_entropy(y, weights)
    return entropy_original - entropy_split


def cal_max_gain(x, y, ai, weights):
    feature = x[:, ai]
    max_v = max(feature)
    min_v = min(feature)
    step = (max_v - min_v) / 10

    T = [min_v + step * i for i in range(1, 9)]

    gains = [cal_gain(feature, y, t, weights) for t in T]
    return max(gains), T[gains.index(max(gains))]


def most_common(y):
    count = defaultdict(int)
    for i in y:
        count[i] += 1
    return max(count.items(), key=lambda x: x[1])[0]


def get_subset(x, y, a, t, weights):
    feature = x[:, a]
    right_part = feature > t
    left_part = feature <= t
    right_x = x[right_part, :]
    right_y = y[right_part]
    left_x = x[left_part, :]
    left_y = y[left_part]
    left_w = weights[left_part]
    right_w = weights[right_part]
    return left_x, left_y, right_x, right_y, left_w, right_w
