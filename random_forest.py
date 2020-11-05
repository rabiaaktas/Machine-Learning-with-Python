# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 17:46:50 2020

@author: rabia
"""

#%%

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv('random+forest+regression+dataset.csv', sep = ';', header = None)

x = df.iloc[:,0].values.reshape(-1,1)
y = df.iloc[:,1].values.reshape(-1,1)

#%%
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators = 100, random_state = 42)

rf.fit(x,y)

#%%

rf.predict(np.array([[7.7]]))

x_ = np.arange(min(x), max(x), 0.01).reshape(-1,1)
y_head = rf.predict(x_)

plt.scatter(x,y,color = 'red')
plt.plot(x_,y_head,color = 'green')
plt.xlabel('tribün level')
plt.ylabel('Fiyat')
plt.show()

#%%

from sklearn.metrics import r2_score
yhead = rf.predict(x)
print("r score: ", r2_score(y,yhead))