#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 18:56:25 2018

@author: lakshay
"""

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training set
dataset_train = pd.read_csv('bitcoin_price.csv')
training_set = dataset_train.iloc[31:1127, 1:2].values

## Taking care of missing data
#from sklearn.preprocessing import Imputer
#imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
#imputer = imputer.fit(training_set)
#training_set = imputer.transform(training_set)

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 1 timesteps and 1 output
X_train = training_set_scaled[0:1095]
y_train = training_set_scaled[1:1096]

# Reshaping
X_train = np.reshape(X_train, (1095,1,1))
#y_train = np.reshape(y_train, (1095,1,1))

# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 4, activation = 'sigmoid' , input_shape = (None, 1)))
regressor.add(Dropout(0.2))

## Adding a second LSTM layer and some Dropout regularisation
#regressor.add(LSTM(units = 4, activation = 'sigmoid'))
#regressor.add(Dropout(0.2))
#
## Adding a third LSTM layer and some Dropout regularisation
#regressor.add(LSTM(units = 4, activation = 'sigmoid'))
#regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)


# Part 3 - Making the predictions and visualising the results

# Getting the real stock price from 16th January 2018 to 15th February 2018
real_price = dataset_train.iloc[15:31, 1:2].values
real_price = sc.transform(real_price)
real_price = sc.inverse_transform(real_price)
real_price = np.array(real_price)

# Getting the predicted stock price
inputs = real_price
inputs = sc.transform(inputs)
inputs = np.reshape(inputs, (16, 1, 1))
predicted_price = regressor.predict(inputs)
predicted_price = sc.inverse_transform(predicted_price)
predicted_price = np.array(predicted_price)

# Visualising the results
plt.plot(real_price, color = 'red', label = 'Real Price')
plt.plot(predicted_price, color = 'blue', label = 'Predicted Price')
plt.title('Bitcoin Price Prediction')
plt.xlabel('Time')
plt.ylabel('Bitcoin Price')
plt.legend()
plt.show()



