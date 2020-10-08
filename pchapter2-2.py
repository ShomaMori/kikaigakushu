# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 21:00:23 2020

@author: Shoma Mori
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from IPython.display import display

#In[2]
X,y = mglearn.datasets.make_wave(n_samples=40)
plt.plot(X,y,'o')
plt.ylim(-3,3)
plt.xlabel("Feature")
plt.ylabel("Target")
plt.show()

#In[3]
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

#In[5]
print("Sample counts per class: \n{}".format({n: v for n, v in zip(
    cancer.target_names, np.bincount(cancer.target))}))

#In[7]
from sklearn.datasets import load_boston
boston = load_boston()

#In[8]
X,y = mglearn.datasets.load_extended_boston()

#In[9]
mglearn.plots.plot_knn_classification(n_neighbors=1)
plt.show()