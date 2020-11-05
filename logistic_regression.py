# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 00:10:04 2020

@author: rabia
"""

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

# %% read csv
data = pd.read_csv("data.csv")
data.drop(["Unnamed: 32","id"],axis=1,inplace = True)
data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]
print(data.info())

y = data.diagnosis.values
x_data = data.drop(["diagnosis"],axis=1)

# %% normalization
x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data)).values

# (x - min(x))/(max(x)-min(x))

# %% train test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2,random_state=42)

x_train = x_train.T
x_test = x_test.T
y_train = y_train.T
y_test = y_test.T

#%%
def initialize_w_b(dimension): 
    w = np.full((dimension,1),0.01)
    b = 0.0
    return w,b

# w,b = initialize_weights_and_bias(30)

def sigmoid(z): 
    sig = 1/(1+ np.exp(-z))
    return sig
# print(sigmoid(0))

#%% propagations   
def forward_backward_propagation(w, b, x_train, y_train):
    # forward propagation
    z = np.dot(w.T, x_train) + b
    y_head = sigmoid(z)
    loss = -y_train * np.log(y_head) - (1-y_train) * np.log(1-y_head)
    cost = (np.sum(loss))/x_train.shape[1]      # x_train.shape[1]  is for scaling
    
    # backward propagation
    derivative_weight = (np.dot(x_train,((y_head-y_train).T))) / x_train.shape[1] # x_train.shape[1]  is for scaling
    derivative_bias = np.sum(y_head-y_train) / x_train.shape[1]                 # x_train.shape[1]  is for scaling
    gradients = {"derivative_weight": derivative_weight, "derivative_bias": derivative_bias}
    return cost,gradients

#%% Updating(learning) parameters
def update(w, b, x_train, y_train, learning_rate, epochs):
    cost_list = []
    cost_list2 = []
    index = []
    
    # updating(learning) parameters is number_of_iterarion times
    for iteration in range(epochs):
        cost,gradients = forward_backward_propagation(w,b,x_train,y_train)
        cost_list.append(cost)

        w = w - learning_rate * gradients["derivative_weight"]
        b = b - learning_rate * gradients["derivative_bias"]

        if iteration % 10 == 0:
            cost_list2.append(cost)
            index.append(iteration)
            print ("Cost after iteration %iteration: %f" %(iteration, cost))
            
    # we update(learn) parameters weights and bias
    parameters = {"w": w,"b": b}
    plt.plot(index, cost_list2)
    plt.xticks(index, rotation='vertical')
    plt.xlabel("Number of Iterarion")
    plt.ylabel("Cost")
    plt.show()
    return parameters, gradients, cost_list

#%%
    
def predict(w, b, x_test):
    z = sigmoid(np.dot(w.T, x_test) + b)
    y_pred = np.zeros((1, x_test.shape[1]))
    
    for i in range(z.shape[1]):
        if z[0, i] <= 0.5:
            y_pred[0,i] = 0
        else:
            y_pred[0,i] = 0
    return y_pred

#%% logistic regression
    
def logistic_regression(x_train, y_train, x_test, y_test, learning_rate, epochs):
    
    dimension = x_train.shape[0]
    w, b = initialize_w_b(dimension)
    
    parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate, epochs)
    
    y_predict_test = predict(parameters["w"], parameters["b"], x_test)
    y_predict = predict(parameters["w"], parameters["b"], x_train)

    
    #print("Train accuracy : {}".format(100 - np.mean(np.abs(y_predict, y_train)) * 100 ))
    print("Test accuracy : {} %".format(100 - np.mean(np.abs(y_predict_test - y_test)) * 100))
    
#%%
    
logistic_regression(x_train, y_train, x_test, y_test, learning_rate = 3, epochs = 30)

#%%

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(x_train.T, y_train.T)
print("Test accuracy: ".format(lr.score(x_train.T, y_train.T)))



