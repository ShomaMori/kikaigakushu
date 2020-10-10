# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 00:31:17 2020

@author: coop
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from IPython.display import display

#forgeデータセットの生成
X, y = mglearn.datasets.make_forge()
#データセットをプロット
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.legend(["Class 0", "Class 1"], loc = 4)
plt.xlabel("First feature")
plt.ylabel("Second feature")
plt.title("forge")
plt.show()

#waveデータセットの生成
X,y = mglearn.datasets.make_wave(n_samples=40)
#データセットをプロット
plt.plot(X,y,'o')
plt.ylim(-3,3)
plt.xlabel("Feature")
plt.ylabel("Target")
plt.title("wave")
plt.show()

#cancerデータセット
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()


#ボストンデータセット
from sklearn.datasets import load_boston
boston = load_boston()


#拡張ボストンデータセット
#特徴量の積も特徴量としている
exboston = mglearn.datasets.load_extended_boston()
