# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 14:41:28 2021

@author: Stian
"""

#import gym
from environment import Stock_Environment
from stable_baselines3 import A2C

import matplotlib.pyplot as plt
cmap = plt.cm.get_cmap('RdYlGn')
import matplotlib.colors as mcolors
offset = mcolors.TwoSlopeNorm(vmin=-50, vcenter=0., vmax=50)

import gym
env = gym.make('BipedalWalker-v3')

#env = Stock_Environment()
#env.reset()

model = A2C('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=50000)

score = 0
obs = env.reset()
for i in range(1000):
    env.render()
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    score += reward
    
    
    if done:
        env.close()
        # plot her!
        break
        
   
    
def plot_best():
    best_score = score
    env.save_best(score)
    plt.plot(env.best_data, color='red', label="pris")
    colors = [cmap(offset(j)) for j in env.best_gevinster]
            
    for j in range(len(env.best_sell_indexes)):
        plt.axvspan(env.best_buy_indexes[j], env.best_sell_indexes[j], alpha=0.3, color=colors[j])
    plt.title('best with score: '+str(round(score, 4)))
    plt.show()
  
