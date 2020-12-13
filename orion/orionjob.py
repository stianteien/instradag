# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 19:06:53 2020

@author: Stian
"""

import pandas as pd
import numpy as np
import pickle
import argparse
import time

from tensorflow.keras.layers import Dense, LSTM, Dropout, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras import backend as K
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


tic = time.time()

# =============================================================================
# Add save location
# =============================================================================

ap = argparse.ArgumentParser()
ap.add_argument(
    "--save_folder", 
     help="path to where results is saved",
     required=True)
args = vars(ap.parse_args())
save_folder = args["save_folder"]

# =============================================================================
# Load data 
# =============================================================================

# X data
with open('X_88dager.pkl', 'rb') as f:
    dataxes = pickle.load(f)
    

# Y data
with open('y_88dager.pkl', 'rb') as f:
    datayes = pickle.load(f)
    

X_train, X_test, y_train, y_test = train_test_split(dataxes,
                                                    datayes,
                                                    test_size=0.33,
                                                    shuffle=True,
                                                    random_state=42)

# =============================================================================
# Build model 
# =============================================================================

def r2_score_nn(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )


model = Sequential()
model.add(LSTM(units=256, input_dim=5, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=256, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(10))

model.compile(optimizer='adam', loss='mse', metrics=['mae', r2_score_nn])


# =============================================================================
# Train model 
# =============================================================================

history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                    epochs=400, batch_size=64, verbose=0)


# =============================================================================
# Save results
# =============================================================================

pd.DataFrame({"loss": history.history["loss"],
              "val_loss": history.history["val_loss"],
              "r2": history.history["r2_score_nn"],
              "r2_val": model.history.history["val_r2_score_nn"]}).to_csv(save_folder + "/losses_and_r2.csv", index=False)


toc = time.time()
f = open(save_folder + "/tidsbrukt.txt","w+")
f.write("tid brukt: "+ str(toc-tic ))
f.close()

print(f" ============ Ferdig paa {toc-tic}  =============")