# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 16:12:14 2020

@author: rabia
"""

#%%
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np


df = pd.read_csv('polynomial+regression.csv', sep = ';')

y = df.araba_max_hiz.values.reshape(-1,1)
x = df.araba_fiyat.values.reshape(-1,1)



#polynomial regression y = b0+b1x+b2x(square)

#%%

reg = LinearRegression()

reg.fit(x,y)

#%%
pred = reg.predict(x)
plt.scatter(x, y)
plt.plot(x, pred, color = 'red', label = 'linear')
plt.ylabel('araba max hız')
plt.xlabel('araba fiyat')

#%%

pp = reg.predict(np.array([[1000000]]))

#%%

#polynomial -- y^hat = b0+b1x+b2x^2+b3x^3......

from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree = 4)

x_pol = poly.fit_transform(x)

#%%
lin_reg = LinearRegression()
lin_reg.fit(x_pol,y)

#%%

pred2 = lin_reg.predict(x_pol)
plt.scatter(x, y)
plt.plot(x, pred, color = 'red', label = 'linear')
plt.ylabel('araba max hız')
plt.xlabel('araba fiyat')
plt.plot(x, pred2, color = 'green',label = 'polynomial')
plt.show()