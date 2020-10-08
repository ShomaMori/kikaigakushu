# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 21:43:40 2020

@author: Shoma Mori
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from IPython.display import display



#In[7]
from sklearn.datasets import load_boston
boston = load_boston()

#In[8]
X,y = mglearn.datasets.load_extended_boston()

#In[9]
mglearn.plots.plot_knn_classification(n_neighbors=1)
plt.show()

#In[10]
mglearn.plots.plot_knn_classification(n_neighbors=3)
plt.show()

#In[11]データを訓練セットとテストセットに分割
from sklearn.model_selection import train_test_split
X,y = mglearn.datasets.make_forge()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)
 
#In[12]
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors = 3)

#In[13]
clf.fit(X_train, y_train)

#In[14]
print("Test set predictions: {}".format(clf.predict(X_test)))

#In[15]
print("Test set acuracy: {:.2f}".format(clf.score(X_test,y_test)))

#In[16]
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


