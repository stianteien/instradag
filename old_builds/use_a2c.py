# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 11:40:32 2021

@author: Stian
"""

import gym
from a2c import a2cAgent
import tensorflow as tf
import numpy as np
from tensorflow import keras

env = gym.make('LunarLander-v2')
n_games = 1500

agent = a2cAgent(fc1=128, fc2=128, n_actions=4, input_dims=8)
agent.load_model('a2c.h5')
best_score = 0

r_hist = []
r_hist_mean = []

for i in range(n_games):
    state = env.reset()
    done = False
    
    episode_reward = 0
    with tf.GradientTape() as tape:
        
        # Do action on evn
        while not done:
            
            action = agent.choose_action(state)
            state, reward, done, _ = env.step(action)
            agent.reward_history.append(reward)
            episode_reward += reward
        
        # Learn from action
        #agent.discount_reward()
        agent.learn(tape)
        r_hist.append(episode_reward)
      
    if  episode_reward > best_score:
        pass
        #agent.save_model('a2c_best.h5')
    r_hist_mean.append(np.mean(r_hist[-50:]))
    if i % 10 == 0:
        print(f'Episode {i}, reward: {episode_reward:.2f}. Mean: {r_hist_mean[-1]:.2f}, std: {np.std(r_hist[-50:]):.2f}')
            
        
            
            