# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 00:36:30 2020

@author: Shoma Mori
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from IPython.display import display

#forgeデータに対してロジスティック回帰とLinearSVMを行う
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
X, y = mglearn.datasets.make_forge()
fig, axes = plt.subplots(1, 2, figsize=(10,3))
for model, ax in zip([LinearSVC(), LogisticRegression()], axes):
    clf = model.fit(X,y)
    mglearn.plots.plot_2d_separator(clf, X, fill=False, eps=0.5,
                                    ax=ax, alpha=.7)
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
    ax.set_title(clf)
    ax.set_xlabel("Feature 0")
    ax.set_ylabel("Feature 1")
axes[0].legend()
plt.show()

#LinearSVCのCを変更した場合の挙動
mglearn.plots.plot_linear_svc_regularization()
plt.show()

#cancerデータにおけるロジスティック回帰の精度比較
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
from sklearn.model_selection import train_test_split
X,y = mglearn.datasets.make_forge()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state = 42)
logreg = LogisticRegression(C=1,solver='liblinear').fit(X_train, y_train)
#C=1
print("cancer")
print("C=1")
print("Training set score: {:.3f}".format(logreg.score(X_train, y_train)))
print("Test set score: {:.3f}".format(logreg.score(X_test, y_test))+"\n")
#C=100
logreg100 = LogisticRegression(C=100, solver='liblinear').fit(X_train, y_train)
print("C=100")
print("Training set score: {:.3f}".format(logreg100.score(X_train, y_train)))
print("Test set score: {:.3f}".format(logreg100.score(X_test, y_test))+"\n")
#C=0.01
logreg001 = LogisticRegression(C=0.01, solver='liblinear').fit(X_train, y_train)
print("C=0.01")
print("Training set score: {:.3f}".format(logreg001.score(X_train, y_train)))
print("Test set score: {:.3f}".format(logreg001.score(X_test, y_test))+"\n")

#Cの変化による係数の挙動をプロット
plt.plot(logreg.coef_.T, 'o', label = "C =1")
plt.plot(logreg100.coef_.T, '^', label = "C =100")
plt.plot(logreg001.coef_.T, 'v', label = "C =0.01")
plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation=90)
plt.hlines(0, 0, cancer.data.shape[1])
plt.ylim(-5, 5)
plt.xlabel("CFeature")
plt.ylabel("Coefficient magnitude")
plt.legend()
plt.show()

#L1正則化項を適用
for C,marker in zip([0.01, 1, 100], ['o','^', 'v']):
    lr_l1 = LogisticRegression(C=C, penalty='l1', solver='liblinear').fit(X_train, y_train)
    print("C=" + str(C))
    print("Training accuracy of l1 logreg with C={:.3f}: {:.2f}".format(
        C, lr_l1.score(X_train, y_train)))
    print("Test accuracy of l1 logreg with C={:.3f}: {:.2f}".format(
        C, lr_l1.score(X_test, y_test)))
    plt.plot(lr_l1.coef_.T, marker, label="C={:.3f}".format(C))
    print("\n")
    
plt.xticks(range(cancer.data.shape[1]),cancer.feature_names,rotation=90)
plt.hlines(0, 0, cancer.data.shape[1]) 
plt.xlabel("Feature")
plt.ylabel("Cpefficient magnitude")

plt.ylim(-5, 5)
plt.legend(loc=3)    
