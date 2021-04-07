# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 13:02:36 2021

@author: Stian
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 22:07:46 2021

@author: Stian
"""

# == Basic stuff == 
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import time
import seaborn as sns

# == Own files (with or without lib) ==
from lib.create_dataset import create_dataset
from lib.punish_shopping import Punish_shopping

# == Keras ==
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input, Bidirectional, Conv1D
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.optimizers import Adam

# == Sklearn ==
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import learning_curve
from sklearn.model_selection import RandomizedSearchCV

#import sklearn
#if sklearn.__version__ == '1.0.dev0' or int(sklearn.__version__.split('.')[1]) >= 24:
#    from sklearn.experimental import enable_halving_search_cv
#    from sklearn.model_selection import HalvingGridSearchCV
#    print('Halving imported')


def lstm_prepare(data):
    dataxes = []
    datayes = []
    sanne = []
    look_back=15
    look_forward=10
    data_ = data[['macd','sma15_derivert','rsi_20',
                 'sma8-16','trix','adx']]
    #data_ = PCA().fit_transform(data_)

    datax, datay, sann = create_dataset().create(data_,
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


def make_model(LSTMneuron=128, DenseNeuron=64, CNNneurons=16,
               n_dense=1, n_lstm=2, dropout_rate=0):
    # make net and train
    model = Sequential()
    model.add(Input(shape=(15,6,)))
    
    # Add LSTM
    if n_lstm >= 3:
        model.add(Bidirectional(LSTM(units=LSTMneuron, return_sequences=True)))
        for n in range(n_lstm - 2):
            model.add(LSTM(units=LSTMneuron, return_sequences=True))
        model.add(LSTM(units=LSTMneuron, return_sequences=False))  
    elif n_lstm == 2:
        model.add(Bidirectional(LSTM(units=LSTMneuron, return_sequences=True)))
        model.add(LSTM(units=LSTMneuron, return_sequences=False))
    elif n_lstm == 1:
        model.add(Bidirectional(LSTM(units=LSTMneuron, return_sequences=False)))
            

    
    # Add Dense
    for n in range(n_dense):
        model.add(Dense(DenseNeuron, activation='relu'))
       
    # Output
    model.add(Dense(2, activation='softmax'))     

    model.compile(optimizer=Adam(lr=0.0005), 
                  loss='binary_crossentropy',
                  metrics=["accuracy"])
    
    return model

# start
tid1 = time.time()

# make model and pipeline


# get data
shopping = Punish_shopping()
shopping.cost = 40

ekstra = '/content/gdrive/My Drive/intradag/'
path = ekstra+'data_clean/'
np.random.seed(1)
filer = np.random.choice(os.listdir(path), 60)
#filer = ['Aker BP 11.12.2020.xlsx']

#prepare_for_lstm()
X = []
actions = []
for fil in filer:
    x, action = Xy(path+fil)
    X.extend(x)
    actions.extend(action)

X = np.array(X)
actions = np.array(actions)

acc = []
val_acc = []
z = 2
for i in range(z):
    print(f"Cross validation {i+1} of {z}")
    X_train, X_test, y_train, y_test = train_test_split(
             X, actions, test_size=0.2, shuffle=True, stratify=actions, random_state=i)
    
    model = make_model(LSTMneuron=64, DenseNeuron=64, n_dense=3, n_lstm=2)   
    # Make a training curve and print results
    h = model.fit(X_train,y_train,
                  epochs=42,validation_data=(X_test, y_test),verbose=0)
    acc.append(h.history['accuracy'])
    val_acc.append(h.history['val_accuracy'])

acc = np.array(acc)
val_acc = np.array(val_acc)

train_scores_mean = np.mean(acc, axis=0); train_scores_std = np.std(acc, axis=0)
test_scores_mean = np.mean(val_acc, axis=0); test_scores_std = np.std(val_acc, axis=0)

train_sizes = [i for i in range(acc.shape[1])]
plt.grid()

plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc="best")
plt.show()

print('Traning last model')
model = make_model(LSTMneuron=64, DenseNeuron=64, n_dense=3, n_lstm=2) 
model.fit(X,actions, epochs=40, verbose=0)
model.save(ekstra+'ddqp_pretrained.h5')



#loss: 0.0842 - val_loss: 0.1078
# med 1 min vanlige den er god: loss: 0.0276 - val_loss: 0.0303