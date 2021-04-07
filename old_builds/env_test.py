# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 21:30:30 2021

@author: Stian
"""

from environment import Stock_Environment
from dqp import Agent
import time
import warnings
import matplotlib.pyplot as plt
from matplotlib import cm
cmap = plt.cm.get_cmap('RdYlGn')
import matplotlib.colors as mcolors
offset = mcolors.DivergingNorm(vmin=-0.001, vcenter=0., vmax=0.001)


warnings.filterwarnings("ignore")

agent = Agent(gamma=0.99, epsilon=1.0, alpha=0.05, input_dims=(30, 1),
              n_actions=2, mem_size=100000, batch_size=64)


env = Stock_Environment()
n_games = 40
running_reward = 0
history_score = []  
all_story = []
tid1 = time.time()

for i in range(n_games):
    done = False
    observation = env.reset()
    score = 0
    count = 0
    episode_score = []
    tic = time.time()
    
    # Hovedproblemet er at den ikke veet når den er inne, men vurderer bare fra X at den skal stå inne
    # OGså får den 0 i reward! Kanskje gi litt reward underveis?<- Bare gi litt for å holde seg inn..
    # Reward - portofilo size??:) 
    
    # 27.01:
    # Problemet er at modellen ser om t+1 gir bedre enn t og det stemmer ikke alltid?
    # Løsning hva om man hadde kjørt samme sekvens annerledes?

    while not done:
        action = agent.choose_action(observation)
        observation_, reward, done, info = env.step(action)
        score += reward

        agent.remember(observation, action, reward, observation_, done)
        agent.learn()
            
        observation = observation_
        episode_score.append(reward)
        

        
    # Alt her er bare lagre scoren
    if score > env.best_score:
        env.save_best(score)
        plt.plot(env.best_data)
        colors = [cmap(offset(j)) for j in env.best_gevinster]
        
        for j in range(len(env.best_sell_indexes)):
            c = 'green' if env.best_gevinster[j] > 0 else 'red'
            plt.axvspan(env.best_buy_indexes[j], env.best_sell_indexes[j], alpha=0.3, color=colors[j])
        plt.title('best with score: '+str(round(score, 4)))
        plt.show()
        agent.save_model()
        
    all_story.append(env.save_trades())
               
    running_reward = 0.05*score + (1-0.05)*running_reward
    history_score.append(score)
    
    toc = time.time()
    if i%1 == 0:
        print(f"Epsiode: {i}, episode_score: {score:.4f}, tid: {(toc-tic):.2f}, handler: {len(all_story[-1][2])}")
        
        
        
print(f"Total time: {(time.time()-tid1)/60:.2f} Minutter")
'''    

plt.plot(env.best_data)
for i in range(len(env.best_sell_indexes)):
    plt.axvspan(env.best_buy_indexes[i], env.best_sell_indexes[i], alpha=0.3, color='green')
    
z = 1
plt.plot(all_story[-z][0])
for i in range(len(all_story[-z][2])):
    c = 'green' if all_story[-z][3][i] > 0 else 'red'
    plt.axvspan(all_story[-z][1][i], all_story[-z][2][i], alpha=0.3, color=c)    


for x in all_story[-3][1]:
    plt.axvline(x=x, color="green", linewidth=.5)
for y in all_story[-3][2]:
    plt.axvline(x=y, color="red",  linewidth=.5)
'''