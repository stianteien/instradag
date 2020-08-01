# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 15:16:14 2020

@author: Stian
"""

import os
import pandas as pd
import pathlib

files = (os.listdir())
path = pathlib.Path().absolute()

for i in files:
    if ('data' in i and '.xlsx' in i):
        fil = pd.read_excel(i)
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
        