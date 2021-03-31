# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 22:07:46 2021

@author: Stian
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from lib.create_dataset import create_dataset
from lib.punish_shopping import Punish_shopping
import seaborn as sns
from tensorflow.keras.layers import Dense, Activation, LSTM, Dropout, Input, Bidirectional
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

def lstm_prepare(data):
    dataxes = []
    datayes = []
    sanne = []
    look_back=15
    look_forward=10

    datax, datay, sann = create_dataset().create(data[['macd',
                                                           'sma15_derivert',
                                                           'rsi_20',
                                                           'sma8-16',
                                                           'trix',
                                                           'adx'
                                                          # 'volume'
                                                          ]],
                                    data.sma30_derivert, look_back=look_back, look_forward=look_forward)
    dataxes.extend(datax)
    datayes.extend(datay)
    sanne.append(sann)
    
    return np.array(dataxes)

def change_action(actions, rewards):
    for i in range(len(rewards)-1, 0, -1):
        if rewards[i] > 0:
            actions[i] = [0, 1]
        else:
            actions[i] = [1, 0]
        
    actions[0] = [1,0]
    actions[-1] = [1,0]
    return actions

def Xy(filnavn):
    data = pd.read_excel(filnavn)
    X = lstm_prepare(data)
    pris = data.open_mean[-X.shape[0]:].reset_index(drop=True)
    actions = np.zeros((pris.shape[0],2))
    
    # .fit(state, actionspace)
    rewards = []
    for i in range(len(pris)-1):
        rewards.append((pris[i+1]/pris[i] -1) * 10000)
    rewards[-1] = 0
    
    actions = change_action(actions, rewards)
    rewards = shopping.punish_shopping_progressiv(np.argmax(actions,axis=1), rewards)
    actions = change_action(actions, rewards)
    
    return X, actions

# make net and train
model = Sequential([
    Input(shape=(15,6,)),
    Bidirectional(LSTM(units=128, return_sequences=True)),
    LSTM(units=128),
    Dense(64, activation='relu'),
    Dense(2, activation='softmax')          
])
model.compile(optimizer=Adam(lr=0.0005), loss='mse')

# get data
shopping = Punish_shopping()
shopping.cost = 30

path = 'data_clean/'
filer = np.random.choice(os.listdir(path), 30)
filer = ['Aker BP 11.12.2020.xlsx']

X = []
actions = []
for fil in filer:
    x, action = Xy(path+fil)
    X.extend(x)
    actions.extend(action)

X = np.array(X)
actions = np.array(actions)

# Split up
X_train, X_test, y_train, y_test = train_test_split(
            X, actions, test_size=0.20, shuffle=True, random_state=42)

# Fit net
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100)
model.save("ddqp_pretrained.h5")

plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='test_loss')
plt.legend()
plt.grid()
plt.show()




#loss: 0.0842 - val_loss: 0.1078