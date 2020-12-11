# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 19:46:56 2020

@author: Stian
"""


import os
import sys
import pandas as pd
import pathlib

path = '../lib/'
files = (os.listdir(path=path))

for fil in files:
    if ('quote' in fil and '.csv' in fil):
        data = pd.read_csv(fil)
        
        # Gjør den lik som det andre.
        data['Kjøper'] = [0 for i in range(len(data['time']))]
        data['Selger'] = [0 for i in range(len(data['time']))]
        data['Type'] = [0 for i in range(len(data['time']))]
        navn = data.columns[1]
        
        # Rename columns
        data = data.rename(columns={data.columns[1]: 'Pris', data.columns[2]: 'Volum'})
        
        # Change time from yyyy-mm-dd hh:mm to dd.mm.yyyy hh:mm:ss
        data['time'] = pd.to_datetime(data['time'])
        data['time'] = data['time'].dt.strftime('%dd.%mm.%YYYY %h:%MM:%ss')
        
        # Flip the shit
        data = data.iloc[::-1]
        data = data.reset_index(drop=True)
        
        # Lagre as .xlsx
        data.to_excel('test.xlsx', index=False)
        
        # Bare flipp hele rundt.
        #siste_dato = data['time'].iloc[-1]
        
        #data.columns = 
        #print(navn)
        

#print(siste_dato)    