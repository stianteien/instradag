# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 13:06:25 2021

@author: Stian
"""


from a2c import a2cAgent
from ddqp import DDQNAgent
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
from lib.punish_shopping import Punish_shopping

from environment import Stock_Environment
import time
import warnings
import matplotlib.pyplot as plt
from matplotlib import cm
cmap = plt.cm.get_cmap('RdYlGn')
import matplotlib.colors as mcolors
offset = mcolors.TwoSlopeNorm(vmin=-0.001, vcenter=0., vmax=0.001)



# =============================================================================
# Helping function 
# =============================================================================
def plot_best():
    cost = shopping.cost
    best_score = score
    env.save_best(score)
    env.best_gevinster = [b-cost for b in env.best_gevinster]
    plt.plot(env.best_data, color='blue', label="pris")
    #plt.plot(env.best_sma, color='green')
    colors = [cmap(offset(j)) for j in env.best_gevinster]
            
    for j in range(len(env.best_sell_indexes)):
        plt.axvspan(env.best_buy_indexes[j], env.best_sell_indexes[j], alpha=0.3, color=colors[j])
    plt.title('best with score: '+str(round(score, 4)))
    plt.show()
  
shopping = Punish_shopping() # 33 sweetpoints
shopping.cost = 5

# =============================================================================
# Setting up enviroment with agent
# =============================================================================
env = Stock_Environment()
agent = DDQNAgent(alpha=0.0005, gamma=0.99, n_actions=2, epsilon=1.0,
                  batch_size=64, input_dims=(30,6))


# =============================================================================
# Running and training agent
# =============================================================================
n_games = 1
best_score = -1000
scores = []
eps_history = []
all_story = []

for i in range(n_games):

    actions = []
    observations = []
    observations_ = []
    rewards = []
    dones = []

    done = False
    score = 0
    observation = env.reset()
    tic = time.time()
    
    while not done:
        action = agent.choose_action(observation)
        observation_, reward, done, info = env.step(action)
        score += reward
        
        # Save all actions and hendelser
        actions.append(action)
        observations.append(observation)
        observations_.append(observation_)
        rewards.append(reward)
        dones.append(done)

        observation = observation_

    # == Manipulate rewards ==
    rewards = shopping.punish_shopping_progressiv(actions, rewards)
    
    # == Learn ==
    for observation, action, reward, observation_, done in zip(observations, actions, rewards, observations_, dones):  
        agent.remeber(observation, action, reward, observation_, done)
        agent.learn()
    
    eps_history.append(agent.epsilon)
    score = sum(rewards)
    scores.append(score)

    all_story.append(env.save_trades())
    if score > env.best_score:
      plot_best()
    
    toc = time.time()
    print(f'epsiode {i} score {score:.2f}, time: {toc-tic:.2f}, Handler: {len(all_story[-1][2])}')
  