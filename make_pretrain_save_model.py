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

import keras.backend as K

#import sklearn
#if sklearn.__version__ == '1.0.dev0' or int(sklearn.__version__.split('.')[1]) >= 24:
#    from sklearn.experimental import enable_halving_search_cv
#    from sklearn.model_selection import HalvingGridSearchCV
#    print('Halving imported')


def get_f1(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

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
    
    return X, actions, pris


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
                  metrics=["accuracy", get_f1])
    
    return model

# start
tid1 = time.time()

# make model and pipeline
all_result_handler = []
all_result_gevinst = []
for r in range(20):
    # get data
    print(r)
    result_handler = []
    result_gevinst = []
    p =  [5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41]
    for punish in p:
        shopping = Punish_shopping()
        shopping.cost = punish#33
        
        ekstra = '/content/gdrive/My Drive/intradag/'
        path = 'data_clean/'
        np.random.seed(r)
        filer = np.random.choice(os.listdir(path), 1)
        #filer = ['Aker BP 11.12.2020.xlsx']
        
        #prepare_for_lstm()
        X = []
        actions = []
        priser = []
        for fil in filer:
            x, action, pris = Xy(path+fil)
            #X.extend(x)
            #actions.extend(action)
            X.append(x)
            actions.append(action)
            priser.append(pris)
        
        X = np.array(X, dtype=object)
        actions = np.array(actions, dtype=object)
        #print(f"shape of X: {X.shape}")
        
        for ax, prices in zip(actions, priser):
            buy_index = []
            sell_index = []
            gevinst = []
            forrige = 0
            for i, a in enumerate(np.argmax(ax, axis=1)):
                if a == 1 and forrige == 0: # kjøp
                    buy_index.append(i)
                elif a == 0 and forrige == 1:
                    sell_index.append(i)
                forrige = a
                
            #plt.plot(prices)
            for j in range(len(sell_index)):
            #    plt.axvspan(buy_index[j], sell_index[j], alpha=0.3, color='green')
                gevinst.append((prices[sell_index[j]]/prices[buy_index[j]]-1)*100 - 0.08)
            #plt.show()
            
            result_handler.append(len(sell_index))
            result_gevinst.append(sum(gevinst))
    '''  
    fig, ax1 = plt.subplots()
    l1 = ax1.plot(p, result_gevinst, label="gevinst i %")
    ax1.set_ylabel("gevinst i %")
    ax1.set_xlabel("punish factor")
    ax2 = ax1.twinx()
    l2 = ax2.plot(p, result_handler, color="red", label="handler")
    ax2.set_ylabel("handler")
    labs = [l.get_label() for l in l1+l2]
    ax1.legend(l1+l2, labs, loc=0)
    plt.show()
    '''
    all_result_handler.append(result_handler)
    all_result_gevinst.append(result_gevinst)
    
all_result_handler = np.array(all_result_handler)
all_result_gevinst = np.array(all_result_gevinst)

m_h = np.mean(all_result_handler, axis=0); std_h = np.std(all_result_gevinst, axis=0)
m_g = np.mean(all_result_gevinst, axis=0); std_g = np.std(all_result_gevinst, axis=0)

fig, ax1 = plt.subplots()
l1 = ax1.plot(p, m_g, label="gevinst i %")
ax1.fill_between(p, m_g + std_g, m_g - std_g, alpha=0.2, color='blue')
ax1.set_ylabel("gevinst i %")
ax1.set_xlabel("punish factor")
ax2 = ax1.twinx()
l2 = ax2.plot(p, m_h, color="red", label="handler")
ax2.fill_between(p, m_h + std_h, m_h - std_h, alpha=0.2, color='red')
ax2.set_ylabel("handler")
labs = [l.get_label() for l in l1+l2]
ax1.legend(l1+l2, labs, loc=0)
plt.show()
 
'''
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
                  epochs=80,validation_data=(X_test, y_test),verbose=0)
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
plt.plot(train_sizes, train_scores_mean, '-', color="r", label="Training score")
plt.plot(train_sizes, test_scores_mean, '-', color="g", label="Cross-validation score")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc="best")
plt.ylim(top=1.001)
plt.show()

epochs = 77#np.argmax(test_scores_mean) + 1
print(f'Traning last model with {epochs} epochs')
model = make_model(LSTMneuron=64, DenseNeuron=64, n_dense=3, n_lstm=2) 
model.fit(X,actions, epochs=epochs, shuffle=True, verbose=1)
model.save(ekstra+'ddqp_pretrained.h5')



#loss: 0.0842 - val_loss: 0.1078
# med 1 min vanlige den er god: loss: 0.0276 - val_loss: 0.0303 '''