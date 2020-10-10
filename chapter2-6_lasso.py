# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 01:28:29 2020

@author: coop
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from IPython.display import display

#拡張ボストンデータを使用しリッジ回帰を行う
X,y = mglearn.datasets.load_extended_boston()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
from sklearn.linear_model import Lasso
lasso = Lasso().fit(X_train, y_train)
#α=1での精度を表示
print("exboston")
print("alpha=1")
print("Training set score: {:.2f}".format(lasso.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lasso.score(X_test, y_test)))
print("Number of features used: {}".format(np.sum(lasso.coef_ != 0))+"\n")
#α=0.01での精度を表示
#max_iterの値を増やしている
#こうしておかないとモデルが"max_iterを増やすように警告を発する
lasso001 = Lasso(alpha = 0.01, max_iter=100000).fit(X_train, y_train)
print("alpha=0.01")
print("Training set score: {:.2f}".format(lasso001.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lasso001.score(X_test, y_test)))
print("Number of features used: {}".format(np.sum(lasso001.coef_ != 0))+"\n")
#α=1での精度を表示
lasso00001 = Lasso(alpha = 0.0001, max_iter=100000).fit(X_train, y_train)
print("alpha=0.0001")
print("Training set score: {:.2f}".format(lasso00001.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lasso00001.score(X_test, y_test)))
print("Number of features used: {}".format(np.sum(lasso00001.coef_ != 0)))

#αの値による係数の挙動をプロット
#比較用にリッジ回帰を行う
from sklearn.linear_model import Ridge
ridge01 = Ridge(alpha=0.1).fit(X_train, y_train)
plt.plot(lasso.coef_, 's', label = "Lasso alpha =1")
plt.plot(lasso001.coef_, '^', label = "Lasso alpha =0.01")
plt.plot(lasso00001.coef_, 'v', label = "Lasso alpha =0.0001")
plt.plot(ridge01.coef_, 'o', label = "Ridge alpha =0.1")
plt.legend(ncol=1, loc=(0, 1.05))
plt.ylim(-25, 25)
plt.xlabel("Coefficient index")
plt.ylabel("Coefficient magnitude")