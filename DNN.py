#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 12:01:42 2020

@author: sammy
"""

#Import Models
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LeakyReLU,PReLU
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor


import statistics

import warnings

# Supress NaN warnings
warnings.filterwarnings("ignore",category =RuntimeWarning)

data = pd.read_csv("data/telecom.csv")
X = data.iloc[:, :data.shape[1]-1]
y = data.iloc[:, data.shape[1]-1]

def generate_model(dropout, lr, neuronPct, neuronShrink):
    def build_model():
        # We start with some percent of 5000 starting neurons on 
        # the first hidden layer.
        neuronCount = int(neuronPct * 5000)
        
        # Construct neural network
        model = Sequential()
    
        # So long as there would have been at least 25 neurons and 
        # fewer than 10
        # layers, create a new layer.
        layer = 0
        while neuronCount>15 and layer<5:
            # The first (0th) layer needs an input input_dim(neuronCount)
            if layer==0:
                model.add(Dense(neuronCount, 
                    input_dim=X.shape[1], 
                    activation=PReLU()))
            else:
                model.add(Dense(neuronCount, activation=PReLU())) 
            layer += 1
    
            # Add dropout after each hidden layer
            model.add(Dropout(dropout))
    
            # Shrink neuron count for each layer
            neuronCount = neuronCount * neuronShrink
    
        model.add(Dense(1, activation='linear')) # Output
        model.compile(loss='mae', optimizer=tf.keras.optimizers.RMSprop(lr))
        return model
    return build_model


loocv = LeaveOneOut()
loocv.get_n_splits(X)


def DNN(dropout,lr,neuronPct,neuronShrink):
    
    # Track progress
    mean_benchmark = []
    epochs_needed = []
    num = 0
    
    predictions = pd.DataFrame(columns = ['Actual Effort', 'Predicted Effort', 'Absolute Error'])
    
    for train_index, test_index in loocv.split(X):
        num+=1
        
        X_train, X_val = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_val = y[train_index], y[test_index]
        
        monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, 
        patience=100, verbose=0, mode='auto', restore_best_weights=True)
        
        estimator = KerasRegressor(build_fn=generate_model(dropout, lr, neuronPct, neuronShrink), validation_data=(X_val,y_val),
                  callbacks=[monitor], epochs=1000, verbose=0)
        estimator.fit(X_train, y_train)
        
        epochs = monitor.stopped_epoch
        epochs_needed.append(epochs)
        
        y_pred = pd.Series(estimator.predict(X_val))
        
        MAE = mean_absolute_error(y_val, y_pred)
        mean_benchmark.append(MAE)
        
        pred = pd.DataFrame(list(zip(y_val,y_pred,[MAE])),columns = ['Actual Effort', 'Predicted Effort', 'Absolute Error'])
        predictions=predictions.append(pred,ignore_index=True)
        
    m1 = statistics.mean(mean_benchmark)
    
    from Cliffs_Delta import cliffs_delta
    
    cliffs = cliffs_delta(predictions)
    
    return (float("{0:.4f}". format(m1)),cliffs)

m, c = DNN(
            dropout=0.2081,
            lr=0.072,
            neuronPct=0.0101,
            neuronShrink=0.3093)