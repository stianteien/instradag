# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 09:17:01 2020

@author: Stian

Makes files ready with stockstats.
In this file all indicators is made
"""

import pandas as pd
import numpy as np
import stockstats
from lib.rens import rens


class make_ready:
    def __init__(self):
        pass
    
    def use_stockstats(self, filer):
        # Return stocks as a list of pd dataframes
        stocks = []
        for i, fil in enumerate(filer):
            print(f"{i+1} av {len(filer)} renset" ,end='\r')
            data = rens().clean_data(pd.read_excel(fil))
            stock = stockstats.StockDataFrame.retype(data)
            indikatorer = ['rsi_20', 'trix', 'open_8_sma', 'open_16_sma', 'macds', 'open_30_sma', 'open_15_sma']
            for ind in indikatorer:
                stock.get(ind)
            stock['sma8-16'] = [stock.open_8_sma[i] - stock.open_16_sma[i] for i, value in enumerate(stock.open_8_sma)]
            stock['sma30_derivert'] = [stock.open_30_sma[i] / stock.open_30_sma[i-1] if i>1 else 1 
                                       for i, value in enumerate(stock.open_30_sma)]
            stock['sma15_derivert'] = [stock.open_15_sma[i] / stock.open_15_sma[i-1] if i>1 else 1 
                                       for i, value in enumerate(stock.open_15_sma)]
            stock['derivert'] = [stock.open[i] / stock.open[i-1] if i>1 else 1
                                 for i, value in enumerate(stock.open)]
    
            stocks.append(stock)
        
        return stocks