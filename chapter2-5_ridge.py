# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 01:27:49 2020

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
from sklearn.linear_model import Ridge
ridge = Ridge().fit(X_train,y_train)
#α=1での精度を表示
print("exboston")
print("alpha=1")
print("Training set score: {:.2f}".format(ridge.score(X_train, y_train)))
print("Test set score: {:.2f}".format(ridge.score(X_test, y_test))+"\n")
#α=10での精度を表示
ridge10 = Ridge(alpha=10).fit(X_train, y_train)
print("alpha=10")
print("Training set score: {:.2f}".format(ridge10.score(X_train, y_train)))
print("Test set score: {:.2f}".format(ridge10.score(X_test, y_test))+"\n")
#α=0.1での精度を表示
ridge01 = Ridge(alpha=0.1).fit(X_train, y_train)
print("alpha=0.1")
print("Training set score: {:.2f}".format(ridge01.score(X_train, y_train)))
print("Test set score: {:.2f}".format(ridge01.score(X_test, y_test)))

#αの値による係数の挙動をプロット
#比較用に線形回帰を行う
from sklearn.linear_model import LinearRegression
lr = LinearRegression().fit(X_train, y_train)
#プロット
plt.plot(ridge.coef_, 's', label = "Ridge alpha =1")
plt.plot(ridge10.coef_, '^', label = "Ridge alpha =10")
plt.plot(ridge01.coef_, 'v', label = "Ridge alpha =0.1")
plt.plot(lr.coef_, 'o', label = "LinearRegression")
plt.xlabel("Coefficient index")
plt.ylabel("Coefficient magnitude")
plt.hlines(0, 0, len(lr.coef_))
plt.ylim(-25, 25)
plt.legend()
plt.show()

#リッジ回帰と線形回帰のテスト制度の比較
mglearn.plots.plot_ridge_n_samples()

