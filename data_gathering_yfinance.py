# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 09:23:44 2021

@author: Stian
"""

import yfinance as yf
import os
import pandas as pd
import re
from lib.make_ready import make_ready
import json
import datetime
import time


aksjer = ["EQNR.OL", "Aker.OL", "OSEBX.OL", "DNB.OL", "NEL.OL", "TEL.OL", "STB.OL"]
dager = ['2021-04-08', '2021-04-09']

ekstra = ""
path = 'data_clean/'
filer = [fil.lower() for fil in os.listdir(path)]

# Find the aksjer i don't hvae
combos = []
for aksje in aksjer:
    for dag in dager:
        c = aksje.lower() + " " + ".".join(dag.split("-")[::-1]) + ".xlsx"
        if c not in filer:
            combos.append(c)
            
print(f"Downloading {len(combos)}")   
        


for aksjedag in combos:
    dato = re.findall('\d+.\d+.\d+', aksjedag)[0]
    navn = re.findall('(.*?) '+dato, aksjedag)[0]
    
    dato = dato.split(".")
    dato = datetime.datetime(int(dato[2]), int(dato[1]), int(dato[0]))
    dato_end = dato + datetime.timedelta(days=1)
    
    try:
        print("start download")
        data = yf.download(navn.upper(),
                           start = dato.strftime("%Y-%m-%d"),
                           end = dato_end.strftime("%Y-%m-%d"),
                           interval="1m")
        time.sleep(1)
        
        data = data.rename(columns= {'Open':'open', 'High':'high', 'Low':'low', 'Close':'close', 'Volume':'volume'})
        data = data.drop(columns=['Adj Close'])
        
        stock = make_ready().use_stockstats_simple(data)
        stock.to_excel(path + aksjedag, index=False)
        print("200 ok")
    except:
        print("not working, something wrong in download")
        
    
        
    






