# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 15:30:42 2021

@author: Stian
"""

import pandas as pd
import numpy as np
import os

#from lib.make_ready import make_ready
from lib.create_dataset import create_dataset


class Stock_Environment:
    def __init__(self):
        self.state = None
        self.done = False
        self.reward = 0
        self.info = None
        self.data = None
        self.timestep = 0
        self.filer = os.listdir('data_clean')
        self.n_actions = 2
        self.input_dim = (30, 5)
        self.status = 0 #0 ute,1 inne?
        self.buy_index = 0
        self.pris = None
        self.look_back = 0
        self.portofilo = 100
        
        self.buy_indexes = []
        self.sell_indexes = []
        self.gevinster = []
        self.best_score = -1000
        
        #self.reset()
        

    def reset(self):
        # plukk random fil
        self.fil = 'Aker BP 11.12.2020.xlsx' #np.random.choice(self.filer)
        self.data = self.prepare()
        #self.state = self.data[['macd', 'rsi_20', 'sma8-16', 'trix', 'volume']].iloc[0]
        self.state = self.data[0]
        self.done = False
        self.timestep = 0
        self.status = 0
        self.buy_index = 0
        self.buy_indexes = []
        self.sell_indexes = []
        self.gevinster = []
        self.portofilo = 100
        return self.state
        
        
    def step(self, action):
        '''
        Actions:
            0: Investert i aksje
            1: Ute av aksje
        '''
        self.timestep += 1
        self.reward = self.portofilo
        if self.timestep >= self.data.shape[0]:
            self.done = True
            if self.status == 1: # Sleg ut hvis man er inne på slutten
                self.timestep -= 1
                self.sell()
                self.timestep += 1
            return self.state, self.reward, self.done, self.info 
        
        
        # Do action here
        if self.status == 1 and action == 0: # inne men selger
            self.sell()
        
        elif self.status == 0 and action == 1: # ute men kjøper
            self.buy()
            
        #elif self.status == 0 and action == 0:
        #    print("forblir ute")      

        elif self.status == 1 and (action == 1 or action == 2):
        #    print("forblir inne")
            #self.inside()
            pass
            
        # Result of action/next state
        self.state = self.data[self.timestep]
    
        return self.state, self.reward, self.done, self.info
     

    def buy(self):
        # Noe galt her kanksje?
        self.buy_index = self.timestep-1
        #print(f"kjøper på {self.buy_index}")
        self.status = 1
        self.buy_indexes.append(self.look_back + self.buy_index)
    
    def sell(self):
        # Fin en måte å straffe på for hver handel som gjøres også!
        
        #self.buy_price = self.data['open'].iloc[self.buy_index]
        #self.sell_price = self.data['open'].iloc[self.timestep-1]
        self.buy_price = self.pris[self.look_back + self.buy_index]
        self.sell_price = self.pris[self.look_back + self.timestep-1]
        self.status = 0
        forandring = ((self.sell_price/self.buy_price)-1)
        gevinst = self.portofilo + self.portofilo * forandring 
        self.reward = gevinst #- self.portofilo * (1-0.04) #genvinst - kost
        self.gevinster.append(forandring)
        self.sell_indexes.append(self.look_back + self.timestep-1)
        
    def inside(self):
        #self.buy_price = self.data['open'].iloc[self.buy_index]
        #self.sell_price = self.data['open'].iloc[self.timestep-1]
        self.buy_price = self.pris[self.buy_index]
        self.sell_price = self.pris[self.timestep-1]
        self.reward = ((self.sell_price/self.buy_price)-1)
    
    
    def save_trades(self):
        #self.buy
        # Lagre de grafisk her så kan vi se hva som skjer :)
        return np.array(self.pris), self.buy_indexes, \
            self.sell_indexes, self.gevinster
            
    def save_best(self, score):
        self.best_score = score
        self.best_buy_indexes = self.buy_indexes
        self.best_sell_indexes = self.sell_indexes
        self.best_data = self.pris
        self.best_gevinster = self.gevinster
        print(f"saved best with score: {score:.4f}")
            
           
    def prepare(self):
        #return make_ready().use_stockstats(['data/'+self.fil])[0]
        #return pd.read_excel('data_clean/'+self.fil)
        self.pris = pd.read_excel('data_clean/'+self.fil)['open'].tolist()
        return self.lstm_prepare(pd.read_excel('data_clean/'+self.fil))[0]
    
    def lstm_prepare(self, data):
        dataxes = []
        datayes = []
        sanne = []
        self.look_back=30
        look_forward=10

        datax, datay, sann = create_dataset().create(data[['macd', 'rsi_20', 'sma8-16', 'trix', 'volume']],
                                    data.sma30_derivert, look_back=self.look_back, look_forward=look_forward)
        dataxes.extend(datax)
        datayes.extend(datay)
        sanne.append(sann)
    
        return np.array(dataxes), np.array(datayes)
    

env = Stock_Environment()
