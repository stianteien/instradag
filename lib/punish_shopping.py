# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 12:05:48 2021

@author: Stian

This script is made for combining the punish shopping function and make
cost of shopping easier to change.
"""

import numpy as np

class Punish_shopping:
    def __init__(self):
        self.cost = 8
    
    def punish_shopping_linear(self, actions, rewards):
        
        buy_index = 0
        status = 0
        actions_ = np.array(actions)
        rewards_ = np.array(rewards).copy()
        
        for i, j in enumerate(zip(actions_,rewards_)):
            action, reward = j
            if action and not status:
                status = 1
                buy_index = i
        
            elif status and not action:
                # Sells and do everything
                status = 0
                rewards_[buy_index:i+1] -= (self.cost/(i+1-buy_index)) # Thats the series i want to change
        
        return rewards_
        
    
    def punish_shopping_progressiv(self, actions, rewards):
        buy_index = 0
        status = 0
        actions_ = np.array(actions)
        rewards_ = np.array(rewards).copy()
        for i, j in enumerate(zip(actions_,rewards_)):
            action, reward = j
            if action and not status:
                status = 1
                buy_index = i
        
            elif status and not action:
              # Sells and do everything here
                status = 0
                magnitude = abs(rewards_[buy_index:i+1])
                percent = magnitude/(sum(magnitude) + np.finfo(np.float32).eps.item())
                rewards_[buy_index:i+1] -= percent*self.cost
        
        return rewards_