# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 10:48:51 2021

@author: Stian
"""

from a2c import a2cAgent
import tensorflow as tf
import numpy as np
from tensorflow import keras

from environment import Stock_Environment
from lib.punish_shopping import Punish_shopping
import time
import warnings
import matplotlib.pyplot as plt
from matplotlib import cm
cmap = plt.cm.get_cmap('RdYlGn')
import matplotlib.colors as mcolors
offset = mcolors.TwoSlopeNorm(vmin=-0.001, vcenter=0., vmax=0.001)

env = Stock_Environment()
shopping = Punish_shopping()
n_games = 3

agent = a2cAgent(n_actions=2, input_dims=(30, 6), batch_size=64)
#agent.load_model('a2c.h5')

r_hist = []
best_score = -1000
all_story = []

for i in range(n_games):
    state = env.reset()
    tic = time.time()
    done = False
    
    score = 0
    with tf.GradientTape() as tape:
        
        # Do action on env
        while not done:
            
            action = agent.choose_action(state)
            state, reward, done, _ = env.step(action)
            agent.reward_history.append(reward)
            score += reward
        
        
        # == Manipulate rewards ==
        agent.reward_history = shopping.punish_shopping_progressiv(agent.action_history,
                                                                   agent.reward_history).tolist()
        score = sum(agent.reward_history)

        # == Learn from action ==
        agent.learn(tape)

        r_hist.append(score)
    
    all_story.append(env.save_trades())

    # == Save best model ==
    if score < agent.best_score:
        agent.save_model('/content/gdrive/My Drive/intradag/a2c_temp.h5')

    # == Load best model if convergate to one thing ==
    #if sum(r_hist[-10:]) == 0 or r_hist[-1] == mean(r_hist[:-10]):
    #    agent.load_model('/content/gdrive/My Drive/intradag/a2c_temp.h5')

    # == PLOT ==
    #if env.best_score < score:
    #  plot_best()
         

    # == PRINT ==
    toc = time.time()
    if i % 1 == 0:
        print(f'Episode {i}, reward: {score:.2f}.  tid: {(toc-tic):.2f}. handler: {len(all_story[-1][2])}')
           