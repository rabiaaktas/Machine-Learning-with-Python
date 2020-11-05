# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 16:59:12 2020

@author: rabia
"""
#%%

from sklearn.datasets import load_iris
import pandas as pd
import numpy as np

#%%

iris = load_iris()

x = iris.data
y = iris.target

#%%

x = (x - np.mean(x)) / (np.max(x) - np.min(x))

#%%

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)

#%%

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(x_train, y_train)
#%%

from sklearn.model_selection import cross_val_score
accurisies = cross_val_score(estimator=knn, X = x_train, y = y_train, cv = 10)
print("Avarage accuracy on cross validation : ", np.mean(accurisies))
print("Avarage std: ", np.std(accurisies))

#%%
predicted = knn.predict(x_test)
print("Test accuracy: ", knn.score(x_test, y_test))

#%% grid search cross validarion

from sklearn.model_selection import GridSearchCV
grid = {"n_neighbors" : np.arange(1,50)}
knn = KNeighborsClassifier()

knn_cv = GridSearchCV(knn, grid, cv = 10)
knn_cv.fit(x, y)

#%%
print("Hyperparameter: ", knn_cv.best_params_)
print("Tuned best accuracy : ", knn_cv.best_score_)