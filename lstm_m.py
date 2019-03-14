#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 10 22:16:11 2018

@author: rohi
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values
                                 
#feature scaling
from sklearn.preprocessing import MinMaxScaler #normalization best for RNN - used here MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)


#60 timesteps, 1 output (next financial stock price for that day) - 3 months - using past 3 months data to predict output of next day (t)
X_train = [] #3 months input
Y_train = [] #output

for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0]) #row, coloumn
    Y_train.append(training_set_scaled[i, 0])

X_train, Y_train = np.array(X_train), np.array(Y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
 
from keras.layers import LSTM
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

regressor = Sequential()
 #layer 1
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

#layer 2
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

#layer 3
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

#layer 4
regressor.add(LSTM(units = 50, return_sequences = False))
regressor.add(Dropout(0.2))

#output layer
regressor.add(Dense(units = 1))

#compile
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

#training
regressor.fit(X_train, Y_train, epochs = 100, batch_size = 32)

#making predictions and visualizing output
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_prices_open = dataset_test.iloc[:, 1:2].values
                                          
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_train) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)

X_test = [] #3 months input
for i in range(60, 80): #20+60
    X_test.append(inputs[i-60:i, 0]) #row, coloumn

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

plt.plot(real_stock_prices_open, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google stock price prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

