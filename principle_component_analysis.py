# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 22:12:50 2020

@author: rabia
"""
#%%
from sklearn.datasets import load_iris
import pandas as pd

#%%
iris =  load_iris()
data = iris.data
features = iris.feature_names
y = iris.target

df = pd.DataFrame(data, columns = features)
df["class"] = iris.target
type(data)

x = data

#%%
from sklearn.decomposition import PCA

pca = PCA(n_components = 2, whiten = True) #whiten -- normalize etmek , feature sayısı kadar component var
pca.fit(x) #boyut düşürülmüyor, model elde ediliyor.

x_pca = pca.transform(x)
print("variance ratio: ", pca.explained_variance_ratio_)
print("sum: ", sum(pca.explained_variance_ratio_))

#%%

df["p1"] = x_pca[:,0]
df["p2"] = x_pca[:,1]

color = ["red", "green", "blue"]

import matplotlib.pyplot as plt

for each in range(3):
    plt.scatter(df.p1[df["class"] == each], df.p2[df["class"] == each], color = color[each], label = iris.target_names[each])
    
plt.legend()
plt.xlabel("p1")
plt.ylabel("p2")
plt.show()