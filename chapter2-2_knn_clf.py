# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 01:13:05 2020

@author: coop
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from IPython.display import display


#K=1データはforge
mglearn.plots.plot_knn_classification(n_neighbors=1)
plt.show()

#K=3
mglearn.plots.plot_knn_classification(n_neighbors=3)
plt.show()

#境界分けをしてプロット
X,y = mglearn.datasets.make_forge()
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors = 3)
fig, axes = plt.subplots(1, 3, figsize=(10, 3))
for n_neighbors, ax in zip([1, 3, 9], axes):
    #fitメソッドは自分自身を返すので1行で
    #インスタンスを生成してfitすることができる
    clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X,y)
    mglearn.plots.plot_2d_separator(clf, X, fill = True, eps = 0.5, ax = ax, alpha = .4)
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax = ax)
    ax.set_title("{} neighbor(s)".format(n_neighbors))
    ax.set_xlabel("feature 0")
    ax.set_ylabel("feature 1")
axes[0].legend(loc = 3)
plt.show()


#cancerデータを使用し、Kの値による精度の推移をプロット
#cancerデータで学習させる
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
from sklearn.model_selection import train_test_split
X,y = mglearn.datasets.make_forge()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state = 66)
training_accuracy = []
test_accuracy = []
#n_neighborを1から10まで試す
neighbors_settings = range(1, 11)

for n_neighbors in neighbors_settings:
    #モデル構築
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train, y_train)
    #訓練セットを記録
    training_accuracy.append(clf.score(X_train, y_train))
    #汎化精度を記録
    test_accuracy.append(clf.score(X_test, y_test))

plt.plot(neighbors_settings, training_accuracy, label = "training accuracy")
plt.plot(neighbors_settings, test_accuracy, label = "test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()