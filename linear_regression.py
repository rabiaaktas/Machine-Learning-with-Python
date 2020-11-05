# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 16:36:42 2020

@author: rabia
"""
#%%
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("linear_regression_dataset.csv", sep = ";")



from sklearn.linear_model import LinearRegression
import numpy as np

linear = LinearRegression()

x = df.deneyim.values.reshape(-1,1)  #.values for change it to numpy, normally pandas series
y = df.maas.values.reshape(-1,1)

linear.fit(x,y)

array = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]).reshape(-1,1)

plt.scatter(df.deneyim, df.maas)

pred = linear.predict(array)
y_head = linear.predict(x)
plt.plot(x, y_head, color = 'red')
plt.show()

#%%
from sklearn.metrics import r2_score

print("score", r2_score(y,y_head))

#%%
#b0 = linear.predict(0).reshape(-1,1)

b0_ = linear.intercept_
print("b0_: ", b0_)

b1 = linear.coef_
print("slope", b1)

den = linear.predict(np.array([[20]]))
print("20 deneyim : ", den)

#%%

df_mult = pd.read_csv('multiple_linear_regression_dataset.csv', sep = ';')

x = df_mult.iloc[:,[0,2]].values
y = df_mult.maas.values.reshape(-1,1)

mult = LinearRegression()
mult.fit(x,y)

print('b0 : ', mult.intercept_)
print('b1, b2 : ', mult.coef_)

mult.predict(np.array([[10,35],[5,35]]))
