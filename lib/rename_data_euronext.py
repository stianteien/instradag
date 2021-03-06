# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 19:46:56 2020

@author: Stian
"""


import os
import sys
import pandas as pd
import pathlib
import datetime


class rename_data_euronext:
    def __init__(self):
        self.path = '../data/'
        self.files = (os.listdir(path=self.path))

    def rename(self):
        for fil in self.files:
            if ('quote' in fil and '.csv' in fil):
                data = pd.read_csv(self.path + fil)
                
                # Gjør den lik som det andre.
                data['Kjøper'] = [0 for i in range(len(data['time']))]
                data['Selger'] = [0 for i in range(len(data['time']))]
                data['Type'] = [0 for i in range(len(data['time']))]
                navn = data.columns[1]
                
                # Rename columns
                data = data.rename(columns={data.columns[1]: 'Pris', data.columns[2]: 'Volum'})
                
                # Change time from yyyy-mm-dd hh:mm to dd.mm.yyyy hh:mm:ss
                data['time'] = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M')
                data['time'] = data['time'].dt.strftime('%d.%m.%Y %H:%M:%S')
                
                # Flip the shit
                data = data.iloc[::-1]
                data = data.reset_index(drop=True)
                
                # Lagre as .xlsx
                #navn + data
                tid = pd.to_datetime(data['time'].iloc[0], format='%d.%m.%Y %H:%M:%S').strftime('%d.%m.%Y')
                data.to_excel(self.path + navn+' '+tid+'.xlsx', index=False)
                
                # delete old file?
                try:
                    os.remove(self.path+fil)
                    print(f"Gjort om {fil}")
                except Exception as e:
                    print(e)
 
        print("---DONE---")    
        

rename_data_euronext().rename()
