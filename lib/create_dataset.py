# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 22:06:04 2020

@author: Stian
Creates a dataset that looks back and forwards
Made for machine learning (RNN)
Return x_data and y_data
"""


import pandas as pd
import numpy as np
from sklearn import preprocessing


class create_dataset:
    def __init__(self):
        pass
    
    def create(self, dataset, true_price, look_back = 1, look_forward = 1):
        
        # Standarised -- DET ER JO STANDARISERT ALLEREDE!
        #sc = preprocessing.StandardScaler()
        #dataset = sc.fit_transform(dataset)
        
        if type(dataset) is not np.ndarray:
            dataset = dataset.to_numpy()
        lookback = look_back
        trueprice = []
        datax = []
        datay = []
    
        for i,v in enumerate(dataset):
            if i >= lookback -1:
                datax.append([dataset[j] for j in range(i-lookback+1, i+1, 1)])
                datay.append(true_price[i])   
        
        trueprice = [i for i in true_price]
        datax = np.array(datax)
        datay = np.array(datay)
        trueprice = np.array(trueprice)
        
        
        # Standardize y data for å skille dem fra hverandre
        #mean = [1 for _ in range(datay.shape[1])]
        #std = [0.0006 for _ in range(datay.shape[1])]
        #datay -= mean
        #trueprice -= mean[0]
        
        #datay /= std
        #trueprice /= std[0]
        
        return datax, datay, trueprice