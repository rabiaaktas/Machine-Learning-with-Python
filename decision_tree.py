# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 17:19:41 2020

@author: rabia
"""
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('decision+tree+regression+dataset.csv', sep = ';', header = None)

x = df.iloc[:,0].values.reshape(-1,1)
y = df.iloc[:,1].values.reshape(-1,1)

#%%

from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()

tree_reg.fit(x,y)

tree_reg.predict(np.array([[5.5]]))
x_ = np.arange(min(x), max(x), 0.1).reshape(-1,1)
h_head = tree_reg.predict(x_)
#%%

plt.scatter(x, y, color = 'red')
plt.plot(x_, h_head, color = 'green')
plt.show()