# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 17:43:06 2020

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
x1 = np.random.normal(25, 5, 1000)
y1 = np.random.normal(25, 5, 1000)

#class 2 
x2 = np.random.normal(55, 5, 1000)
y2 = np.random.normal(60, 5, 1000)

#class3
x3 = np.random.normal(55, 5, 1000)
y3 = np.random.normal(15, 5, 1000)
    
x = np.concatenate((x1, x2, x3), axis = 0) #axis 0 column olarak birleştir 
y = np.concatenate((y1, y2, y3), axis = 0)

dictionary = {"x" : x, "y" : y}
data = pd.DataFrame(dictionary)
#k- means bunu görür.
plt.scatter(x1, y1, color = 'black')
plt.scatter(x2, y2, color = 'black')
plt.scatter(x3, y3, color = 'black')
plt.show()

#%%
from sklearn.cluster import KMeans
wcss = []

for each in range(1, 15):
    kmeans = KMeans(n_clusters = each)
    kmeans.fit(data)
    wcss.append(kmeans.inertia_) #wcss değeri -- inertia
    
plt.plot(range(1, 15), wcss)
plt.xlabel("k value")
plt.ylabel("WCSS")
plt.plot()

#%% k = 3 için model

kmean = KMeans(n_clusters = 3)
clusters = kmean.fit_predict(data)
data["label"] = clusters

plt.scatter(data.x[data.label == 0], data.y[data.label == 0], color = 'red')
plt.scatter(data.x[data.label == 1], data.y[data.label == 1], color = 'blue')
plt.scatter(data.x[data.label == 2], data.y[data.label == 2], color = 'green')
plt.scatter(kmean.cluster_centers_[:, 0], kmean.cluster_centers_[:, 1], color = "purple")
plt.show()

