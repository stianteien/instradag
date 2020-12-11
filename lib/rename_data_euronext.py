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
    if ('.csv' in fil):
        data = pd.read_csv(fil)
        siste_dato = data['time'].iloc[-1]

print(siste_dato)    