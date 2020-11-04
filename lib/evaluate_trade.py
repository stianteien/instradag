# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 11:12:51 2020

@author: Stian

Evaluate trades
"""


import pandas as pd
import numpy as np

class evaluate_trades:
    def __init__(self):
        pass
    
    def prepare_evaluate_trade(self, prediction):
        # Setter indeks til hver for å få riktig.. Litt tungvindt
        where_are_NaNs = np.isnan(prediction) #Byttet NaN til 1
        prediction[where_are_NaNs] = 1
        
        dict_pred = {}
        
        for i, pred in enumerate(prediction):
            for j, v in enumerate(pred):
                if i+j not in dict_pred:
                    dict_pred[i+j] = [v]
                else:
                    dict_pred[i+j].append(v)
        
        
        df_pred = pd.DataFrame(columns=['tid','verdier','avg'])
        df_pred.tid = list(dict_pred.keys())
        df_pred.verdier = list(dict_pred.values())
        df_pred.avg = [sum(l)/len(l) for l in list(dict_pred.values())]
        
        return df_pred
    
    def evaluate_trade(self):
        return 0

