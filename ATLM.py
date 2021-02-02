#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 22:57:15 2020

@author: sammy
"""

#Import Models
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression

import statistics

import warnings

# Supress NaN warnings
warnings.filterwarnings("ignore",category =RuntimeWarning)

data = pd.read_csv("data/kemerer.csv")
X = data.iloc[:, :data.shape[1]-1]
y = data.iloc[:, data.shape[1]-1]


loocv = LeaveOneOut()
loocv.get_n_splits(X)

def ATLM():
    
    # Track progress
    mean_benchmark = []
    num = 0
    
    predictions = pd.DataFrame(columns = ['Actual Effort', 'Predicted Effort', 'Absolute Error'])
    
    for train_index, test_index in loocv.split(X):
        num+=1
        
        X_train, X_val = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_val = y[train_index], y[test_index]
        
        model = LinearRegression()
        model.fit(X_train,y_train)
        
        y_pred = model.predict(X_val)
        MAE = mean_absolute_error(y_val, y_pred)
        mean_benchmark.append(MAE)
        
        pred = pd.DataFrame(list(zip(y_val,y_pred,[MAE])),columns = ['Actual Effort', 'Predicted Effort', 'Absolute Error'])
        predictions=predictions.append(pred,ignore_index=True)
    
    m1 = statistics.mean(mean_benchmark)
    
    from Cliffs_Delta import cliffs_delta
    
    cliffs = cliffs_delta(predictions)
    
    return (float("{0:.4f}". format(m1)),cliffs)

m, c = ATLM()