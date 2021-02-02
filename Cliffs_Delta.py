#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 10:27:13 2020

@author: sammy
"""
import pandas as pd

def cliffs_delta(values):
    act = values['Actual Effort']
    pred = values['Predicted Effort']
    
    n = len(act)
    
    ap = 0
    pa = 0
    
    for i in range(0,n):
        if act[i]>pred[i]:
            ap+=1
        elif act[i]<pred[i]:
            pa+=1
    cliffs = (ap-pa)/(n*n)
    return float("{0:.4f}". format(cliffs))
