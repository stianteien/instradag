# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 21:57:00 2020

@author: Stian

This is a test file. Run this file to test new editfukcs
"""

import pandas as pd
import numpy as np
import pytest


from lib.rens import rens
from lib.make_ready import make_ready
from lib.create_dataset import create_dataset

data = pd.read_excel('../data/Aker BP 11.12.2020.xlsx')
filnavn = ['../data/Aker BP 11.12.2020.xlsx']

# Rensefilen
def test_rens_make(): 
    r = rens()
    test_data = r.clean_data(data)
    
def test_rens_clean_data():
    r = rens()
    test_data = r.clean_data(data)
    assert len(test_data.columns) > 0, "Ingen columns"
    assert test_data.tid.shape[0] > 0, "Ingen data inni"
    

#  Make ready
def test_make_ready_make():
    m = make_ready()
    
def test_fillna():
    dummy = pd.DataFrame({'A':[1,2,3,np.nan], 'B':[3,4,5,6], 'C':[1,2,3,np.nan]})
    m = make_ready()
    mod_dummy = m.fillna(dummy)
    
    assert np.isnan(mod_dummy.C[3]) == False, "Byttet ikke nan i siste col. "
    assert np.isnan(mod_dummy.A[3]) == True, "Byttet ikke til nan på den første :)"
    
    
def test_use_stockstats():
    m = make_ready()
    data = m.use_stockstats(filnavn)
    
    assert data[0].isnull().values.any() == False, "Noen verdier er nan"

    
# Create dataset
def test_create_dataset_make():
    c = create_dataset()
    
def test_create_dataset_data():
    test_data = make_ready().use_stockstats(filnavn)
    c = create_dataset()
    look_back = 10; look_forward = 10
    x, y, z = c.create(test_data[0][['macd', 'rsi_20', 'sma8-16', 'trix', 'volume']],
                       test_data[0].sma30_derivert,
                       look_back = look_back,
                       look_forward = look_forward)
    
    assert x.shape[1] == look_back, "lookback correction"
    assert y.shape[1] == look_forward, "lookfowrdad"
    assert x.shape[2] > 0, "x amount of feacuters"
    assert x.shape[0] == y.shape[0], "same amount of answers"
    
    


