# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 13:43:13 2021

@author: Stian
"""

import numpy as np
import gym
from dqp import Agent
import time

env = gym.make('Acrobot-v1')
n_games = 100

#env.action_space
#env.observation_space

agent = Agent(gamma=0.99, epsilon=1.0, alpha=0.05, input_dims=6,
              n_actions=3, mem_size=100000, batch_size=64)


history_score = []
running_reward = 0

for i in range(n_games):
    done = False
    observation = env.reset()
    score = 0
    count = 0
    episode_score = []
    tic = time.time()
    
    while not done:
        action = agent.choose_action(observation)
        observation_, reward, done, info = env.step(action)
        score += reward
        agent.remember(observation, action, reward, observation_, done)
        observation = observation_
        agent.learn()
        
        episode_score.append(reward)
        
        count += 1
        if count >= 250:
            done = True
               
    running_reward = 0.05*score + (1-0.05)*running_reward
    history_score.append(score)
    
    toc = time.time()
    if i%1 == 0:
        print(f"Epsiode: {i}, episode_score: {score:.2f}, tid: {(toc-tic):.2f}")
    
env.close()