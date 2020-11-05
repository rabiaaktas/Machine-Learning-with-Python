# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 19:25:48 2020

@author: rabia
"""

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%% create data set
 #ortalaması 25 olan, 5 sigmalı 1000 değer üret.
# 20 - 30 değerleri arasında olacak 666
#class1
x1 = np.random.normal(25, 5, 100)
y1 = np.random.normal(25, 5, 100)

#class 2 
x2 = np.random.normal(55, 5, 100)
y2 = np.random.normal(60, 5, 100)

#class3
x3 = np.random.normal(55, 5, 100)
y3 = np.random.normal(15, 5, 100)
    
x = np.concatenate((x1, x2, x3), axis = 0) #axis 0 column olarak birleştir 
y = np.concatenate((y1, y2, y3), axis = 0)

dictionary = {"x" : x, "y" : y}
data = pd.DataFrame(dictionary)
#k- means bunu görür.
plt.scatter(x1, y1, color = 'black')
plt.scatter(x2, y2, color = 'black')
plt.scatter(x3, y3, color = 'black')
plt.show()

#%% dendogram

from scipy.cluster.hierarchy import dendrogram, linkage
merg = linkage(data, method = 'ward')
dendrogram(merg, leaf_rotation = 90)
plt.xlabel("Data points")
plt.ylabel("Euclidian Distance")
plt.show() 

#%%
from sklearn.cluster import AgglomerativeClustering

hierarchical_clustering = AgglomerativeClustering(n_clusters = 3, affinity = "euclidean", linkage = "ward")
cluster = hierarchical_clustering.fit_predict(data)

data["label"]  = cluster

plt.scatter(data.x[data.label == 0], data.y[data.label == 0], color = 'red')
plt.scatter(data.x[data.label == 1], data.y[data.label == 1], color = 'blue')
plt.scatter(data.x[data.label == 2], data.y[data.label == 2], color = 'green')
plt.show()
