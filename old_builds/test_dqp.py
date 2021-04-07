# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 14:57:12 2021

@author: Stian
"""

import numpy as np
import gym

agent = a2cAgent(fc1=512, fc2=512, n_actions=4, input_dims=8)
agent.load_model('a2c_best.h5')

env = gym.make('LunarLander-v2')

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
    #print(score)
    
    observation = observation_
        
    count += 1
    #if count >= 250:
    #    done = True
            
            
env.close()