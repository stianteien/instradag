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
from rens import rens

pd.options.mode.use_inf_as_na = True


class make_ready:
    def __init__(self):
        pass
    
    def fillna(self, stock):                

        mean_values = {col:np.mean(stock[col]) for col in stock.drop(stock.columns[0], axis=1)}
        stock = stock.fillna(value=mean_values)
        return stock
    
    def use_stockstats(self, filer):
        # Return stocks as a list of pd dataframes
        stocks = []
        for i, fil in enumerate(filer):
            print(f"{i+1} av {len(filer)} renset" ,end='\r')
            data = rens().clean_data(pd.read_excel(fil))
            stock = stockstats.StockDataFrame.retype(data)
            indikatorer = ['rsi_20', 'trix', 'open_8_sma', 'open_16_sma',
                           'macds', 'open_30_sma', 'open_15_sma', 'open_3_sma',
                           'adx']
            for ind in indikatorer:
                stock.get(ind)
            stock['sma8-16'] = [stock.open_8_sma[i] - stock.open_16_sma[i] for i, value in enumerate(stock.open_8_sma)]
            stock['sma30_derivert'] = [stock.open_30_sma[i] / stock.open_30_sma[i-1] if i>1 else 1 
                                       for i, value in enumerate(stock.open_30_sma)]
            stock['sma15_derivert'] = [stock.open_15_sma[i] / stock.open_15_sma[i-1] if i>1 else 1
                                       for i, value in enumerate(stock.open_15_sma)]
            stock['derivert'] = [stock.open[i] / stock.open[i-1] if i>1 else 1
                                 for i, value in enumerate(stock.open)]
            
            #Gjennomsnitt fra forann og bak
            span = 8 # 0 1 2 _3_ 4 5 6
            stock['open_mean'] = stock.open.rolling(span, center=True).mean()
            stock['open_mean'] = stock['open_mean'].fillna(stock.open)
            
            # Ta å normaliser rundt 0 for det som er mulig
            stock['rsi_20'] /= 100
            stock['rsi_20'] -= 0.5
            stock['sma30_derivert'] -= 1
            stock['sma30_derivert'] *= 800 # Få dem opp med resten
            stock['sma15_derivert'] -= 1
            stock['sma15_derivert'] *= 800
            
            stock['trix'] *= 10
            stock['adx'] /= 100
            stock['adx'] -= .25
            
            stock['derivert'] -= 1
            
    
            # Fill nan verdier med tilpassede initialverdier
            stock = self.fillna(stock)
            stocks.append(stock)
        
        return stocks