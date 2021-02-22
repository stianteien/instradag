# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 19:05:35 2020

@author: Stian

frozen lake
"""

import gym
import numpy as np
import random
import time
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v0') #4x4 bane stor

action_space_size = env.action_space.n
state_space_size = env.observation_space.n

q_table = np.zeros((state_space_size, action_space_size))

num_episodes = 10000
max_steps_per_epsode = 100 

learning_rate = .1
discount_rate = .99

exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.001

rewards_all_episodes = []

# Q learning
for episode in range(num_episodes):
    state = env.reset()
    done = False
    reward_current_epsiode = 0
    
    for step in range(max_steps_per_epsode):
        
        # Explore or exploit!
        if random.uniform(0,1) > exploration_rate: # Then explote
            action = np.argmax(q_table[state,:]) # Pick the one with most score
        else: # Then explore
            action= env.action_space.sample()
            
        new_state, reward, done, info = env.step(action)
        
        # Update q-table
        q_table[state, action] = q_table[state, action]* (1-learning_rate) + \
            learning_rate * (reward + discount_rate* np.max(q_table[new_state,:]))
            
        state = new_state
        reward_current_epsiode += reward
        
        if done:
            break
    
    # Lower the exploration_rate
    exploration_rate = min_exploration_rate + \
        (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate * episode)
        
    rewards_all_episodes.append(reward_current_epsiode)
    


acum = [0]
for i in rewards_all_episodes:
    acum.append(acum[-1] + i)
        
        
        
np.mean(rewards_all_episodes[-1000:])
        
        
