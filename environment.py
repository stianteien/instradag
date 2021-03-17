# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 15:30:42 2021

@author: Stian
"""

import pandas as pd
import numpy as np
import os
import gym

#from lib.make_ready import make_ready
from lib.create_dataset import create_dataset


class Stock_Environment:
    def __init__(self, random_files = False, sma_as_price = False):
        self.random_files = random_files
        self.sma_as_price = sma_as_price
        self.state = None
        self.done = False
        self.reward = 0
        self.info = {}
        self.data = None
        self.timestep = 0
        #self.path = '/content/gdrive/My Drive/intradag/data_clean/'
        self.path = 'data_clean/'
        self.filer = os.listdir(self.path)
        self.n_actions = 2
        self.input_dim = (30, 6)
        self.status = 0 #0 ute,1 inne?
        self.buy_index = 0
        self.pris = None
        self.best_sma = None
        self.look_back = 0
        self.portofilo = 1
        self.handler = 0
        self.inside_counter = 0
        
        self.action_space = gym.spaces.discrete.Discrete(2)
        self.observation_space = gym.spaces.box.Box(low=-1000, high=1000, 
                                                    shape=(30,6,), 
                                                    dtype=np.float)
        self.reward_range = (-10000, 10000)
        self.metadata = {'Hakkepeil': 'måteåprinte?'}
        
        self.buy_indexes = []
        self.sell_indexes = []
        self.gevinster = []
        self.best_score = -1000
        
        #self.reset()
        

    def reset(self):
        # plukk random fil
        if self.random_files:
            self.fil = np.random.choice(self.filer)
        else:
            self.fil = 'Aker BP 11.12.2020.xlsx'    
        self.data = self.prepare()
        self.state = self.data[0]
        self.done = False
        self.timestep = 0
        self.status = 0
        self.buy_index = 0
        self.buy_indexes = []
        self.sell_indexes = []
        self.gevinster = []
        self.portofilo = 1
        self.gevinst_serie = 0
        self.handler = 0
        self.inside_counter = 0
        return self.state
        
        
    def step(self, action):
        '''
        Actions:
            1: Investert i aksje
            0: Ute av aksje
        '''
        self.timestep += 1
        self.reward = 0
        if self.timestep >= self.data.shape[0]:
            self.done = True
            if self.status == 1: # Selg ut hvis man er inne på slutten
                self.timestep -= 1
                self.sell()
                self.timestep += 1
            return self.state, self.reward, self.done, self.info 
        
        
        #if action == 1:
            # Reard all action with 1
        #    self.reward_all_action()
            
        
        # Do action here
        if self.status == 1 and action == 0: # inne men selger 
            self.sell()
        
        elif self.status == 0 and action == 1: # ute men kjøper
            self.buy()
            
        elif self.status == 0 and action == 0:
            #self.outside()
            #self.reward_all_action()
            pass
        #    print("forblir ute")      

        elif self.status == 1 and action == 1:
            self.reward_all_action()
            #print("forblir inne")
            #self.inside()

            
        # Result of action/next state
        self.state = self.data[self.timestep]
    
        return self.state, self.reward, self.done, self.info
     

    def buy(self):
        self.handler += 1
        self.buy_index = self.timestep-1
        self.status = 1
        self.buy_indexes.append(self.look_back + self.buy_index)
        self.buy_price = self.pris[self.look_back + self.buy_index]
        self.reward_all_action()
        #self.reward = 0 
    
    def sell(self):
        self.inside_counter = 0
        # Fin en måte å straffe på for hver handel som gjøres også!
        self.sell_index = self.look_back + self.timestep-1
        #self.buy_price = self.data['open'].iloc[self.buy_index]
        #self.sell_price = self.data['open'].iloc[self.timestep-1]
        self.sell_price = self.pris[self.look_back + self.timestep-1]
        self.status = 0
        
        forandring = ((self.sell_price / self.buy_price)-1) * 10000
        self.reward_all_action()
        #self.reward = forandring
        
        #gevinst = self.portofilo + self.portofilo * forandring 
        #self.reward = forandring*100 #- self.portofilo * (1-0.04) #genvinst - kost
        self.gevinster.append(forandring)
        self.sell_indexes.append(self.look_back + self.timestep-1)

        
    def inside(self):
        #self.buy_price = self.data['open'].iloc[self.buy_index]
        #self.sell_price = self.data['open'].iloc[self.timestep-1]
        #self.sell_price = self.pris[self.timestep-1]
        #self.reward = ((self.pris[self.look_back + self.timestep-1]/
        #                self.pris[self.look_back + self.timestep-2])-1)
        #self.gevinster.append(self.reward)
        forandring = (self.pris[self.look_back + self.timestep -0]
                       /self.pris[self.look_back + self.timestep -1])-1
        #gevinst = self.portofilo + self.portofilo * forandring 
        self.reward = forandring*10000
        
        
    def outside(self):
        forandring = (self.pris[self.look_back + self.timestep -1]
                       /self.pris[self.look_back + self.timestep -2])-1
        
        #self.reward = -forandring*1000
        
    def reward_all_action(self):
        forandring = (self.pris[self.look_back + self.timestep -1]
                       /self.pris[self.look_back + self.timestep -2])-1
        r = forandring*10000
        self.inside_counter += 1
        r_mark = r + self.reward_long_trades(self.inside_counter) \
            + self.punish_short_trades(self.inside_counter)
        self.reward = r #100 x prosenten
        
    def reward_long_trades(self, x):
        return 1/(1+150*np.exp(-0.5*x))
    
    def punish_short_trades(self, x):
        return 1/(1+150*np.exp(-x)) - 1
    
    def save_trades(self):
        # Lagre de grafisk her så kan vi se hva som skjer :)
        return np.array(self.pris), self.buy_indexes, \
            self.sell_indexes, self.gevinster
            
    def save_best(self, score):
        self.best_score = score
        self.best_buy_indexes = self.buy_indexes
        self.best_sell_indexes = self.sell_indexes
        self.best_data = self.pris
        self.best_sma = self.sma
        self.best_gevinster = self.gevinster
        print(f"saved best with score: {score:.4f}")
            
           
    def prepare(self):
        #return make_ready().use_stockstats(['data/'+self.fil])[0]
        #return pd.read_excel('data_clean/'+self.fil)
        data = pd.read_excel(self.path + self.fil)
        self.pris = data['open'].tolist()
        self.sma = data['open_mean'].tolist()
        #self.sma8 = data['open_8_sma'].tolist()
        if self.sma_as_price:
            self.pris = self.sma
        return self.lstm_prepare(pd.read_excel(self.path + self.fil))[0]
        #return self.ann_prepare(pd.read_excel(self.path + self.fil))
        
    def ann_prepare(self, data):
        X = data[['macd',
              'sma15_derivert',
              'rsi_20',
              'sma8-16',
              'trix',
              'adx' ]]
        return X.to_numpy()
    
    def lstm_prepare(self, data):
        dataxes = []
        datayes = []
        sanne = []
        self.look_back=30
        look_forward=10

        datax, datay, sann = create_dataset().create(data[['macd',
                                                           'sma15_derivert',
                                                           'rsi_20',
                                                           'sma8-16',
                                                           'trix',
                                                           'adx'
                                                          # 'volume'
                                                          ]],
                                    data.sma30_derivert, look_back=self.look_back, look_forward=look_forward)
        dataxes.extend(datax)
        datayes.extend(datay)
        sanne.append(sann)
    
        return np.array(dataxes), np.array(datayes)
    
    def render(self):
        pass
        # Render here!
    

env = Stock_Environment()
