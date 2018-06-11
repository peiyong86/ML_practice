#!/usr/bin/env python
import numpy as np
from ..DecisionTrees.trees import TreeC3

__author__ = "Yong Pei"


class RandomForest:
    def __init__(self, n_estimators=10, max_features=None):
        self.__name__ = 'Random Forest'
        self.n_estimators = n_estimators
        self.max_features = max_features

        self.trees = []
        # 建立森林(bulid forest)
        for _ in range(self.n_estimators):
            tree = TreeC3()
            self.trees.append(tree)

    def get_bootstrap_data(self, X, Y):
        # 通过bootstrap的方式获得n_estimators组数据
        # get int(n_estimators) datas by bootstrap

        m = X.shape[0]
        Y = Y.reshape(m, 1)

        # 合并X和Y，方便bootstrap (conbine X and Y)
        X_Y = np.hstack((X, Y))
        np.random.shuffle(X_Y)

        data_sets = []
        for _ in range(self.n_estimators):
            idm = np.random.choice(m, m, replace=True)
            bootstrap_X_Y = X_Y[idm, :]
            bootstrap_X = bootstrap_X_Y[:, :-1]
            bootstrap_Y = bootstrap_X_Y[:, -1:]
            data_sets.append([bootstrap_X, bootstrap_Y.flatten()])
        return data_sets

    def fit(self, X, Y):
        # 训练，每棵树使用随机的数据集(bootstrap)和随机的特征
        # every tree use random data set(bootstrap) and random feature
        sub_sets = self.get_bootstrap_data(X, Y)
        n_features = X.shape[1]
        if self.max_features == None:
            self.max_features = int(np.sqrt(n_features))
        for i in range(self.n_estimators):
            # 生成随机的特征
            # get random feature
            sub_X, sub_Y = sub_sets[i]
            idx = np.random.choice(n_features, self.max_features, replace=True)
            sub_X = sub_X[:, idx]
            self.trees[i].fit(sub_X, sub_Y)
            self.trees[i].feature_indices = idx
            print("tree", i, "fit complete")

    def predict(self, X):
        y_preds = []
        for i in range(self.n_estimators):
            idx = self.trees[i].feature_indices
            sub_X = X[:, idx]
            y_pre = self.trees[i].predict(sub_X)
            y_preds.append(y_pre)
        y_preds = np.average(np.array(y_preds), axis=0)
        return [1 if s > 0 else -1 for s in y_preds]
