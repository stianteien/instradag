# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 15:16:14 2020

@author: Stian
"""

import os
import sys
import pandas as pd
import pathlib

path = '../data/'
files = (os.listdir(path=path))

for i in files:
    if ('data' in i and '.xlsx' in i):
        fil = pd.read_excel(path + i)
        navn = fil.columns[0]+' '
        dato = fil.iloc[0,0].strftime("%d.%m.%Y")

        old_path = (os.path.join(path, i))
        new_path = (os.path.join(path, navn+dato+'.xlsx'))
        
        try:
            os.rename(old_path, new_path)
            print('Byttet:',navn+dato)
        except:
            print('could not make rename file: ',old_path)

print('---DONE---')
        