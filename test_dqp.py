# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 14:57:12 2021

@author: Stian
"""

import numpy as np
import gym

env = gym.make('Acrobot-v1')

history_score = []
running_reward = 0


done = False
observation = env.reset()
score = 0
count = 0
    
while not done:
    env.render()
    action = agent.choose_action(observation)
    observation_, reward, done, info = env.step(action)
    score += reward
    print(reward)    
    
    observation = observation_
        
    count += 1
    if count >= 250:
        done = True
            
            
env.close()