# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 01:18:43 2020

@author: coop
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from IPython.display import display
from sklearn.neighbors import KNeighborsClassifier

#K=1データはwave
mglearn.plots.plot_knn_regression(n_neighbors=1)

#K=3
mglearn.plots.plot_knn_regression(n_neighbors=3)

#K=3での精度
from sklearn.neighbors import KNeighborsRegressor
X,y = mglearn.datasets.make_wave(n_samples=40)
#waveデータセットを訓練セットとテストセットに分割
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
#3つの最近傍点を考慮するように設定してモデルのインスタンスを生成
reg = KNeighborsRegressor(n_neighbors=3)
#訓練データと訓練ターゲットを用いてモデルを学習させる
reg.fit(X_train, y_train)
print("\nTest set R^2: {:.2f}".format(reg.score(X_test, y_test)))

#-3から3のすべての値に対する予測値の推移
fig, axes = plt.subplots(1, 3, figsize=(15,4))
#-3から3までの間に1000点のデータポイントを作る
line =np.linspace(-3, 3, 1000).reshape(-1,1)
for n_neighbors, ax in zip([1, 3, 9,], axes):
    #1, 3, 9近傍点で予測
    reg = KNeighborsRegressor(n_neighbors=n_neighbors)
    reg.fit(X_train, y_train)
    ax.plot(line, reg.predict(line))
    ax.plot(X_train, y_train, '^', c=mglearn.cm2(0),markersize = 8)
    ax.plot(X_test, y_test, 'v', c=mglearn.cm2(1),markersize = 8)
    ax.set_title(
        "{} neighbor(s)\n train score: {:.2f} test score {:2f}".format(
            n_neighbors, reg.score(X_train, y_train),
            reg.score(X_test, y_test)))
    ax.set_xlabel("Feature")
    ax.set_ylabel("Target")
axes[0].legend(["Model predictions", "Training data/target",
                "Test data/target"], loc="best")


