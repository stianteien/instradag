# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 11:07:03 2021

@author: Stian
"""

import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class a2cAgent(object):
    def __init__(self, fc1, fc2, n_actions, input_dims):
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.model = self.make_net()
        self.critic_value_history = []
        self.action_probs_history = []
        self.reward_history = []
        self.huber_loss = keras.losses.Huber()
        self.optimizer = keras.optimizers.Adam(learning_rate=0.01)
        self.best_score = 0
        
    def make_net(self):
        #inputs = layers.Input(shape=(self.input_dims,))
        #common = layers.Dense(256, activation="relu")(inputs)
        #common = layers.Dense(256, activation="relu")(common)
        
        # For stocks
        inputs = layers.Input(shape=(30,4,))
        common = layers.LSTM(units=128, return_sequences=True)(inputs)
        drop1 = layers.Dropout(0.2)(common)
        common = layers.LSTM(units=128)(drop1)
        drop2 = layers.Dropout(0.2)(common)
        common = layers.Dense(64, activation='relu')(drop2)
        
        
        action = layers.Dense(self.n_actions, activation="softmax")(common)
        critic = layers.Dense(1)(common)
        
        model = keras.Model(inputs=inputs, outputs=[action, critic])
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def choose_action(self, state):
        state = tf.convert_to_tensor(state)
        state = tf.expand_dims(state, 0)
        action_probs, critic_value = self.model(state)
        self.critic_value_history.append(critic_value[0, 0])
        action = np.random.choice(self.n_actions, p=np.squeeze(action_probs))
        self.action_probs_history.append(tf.math.log(action_probs[0, action]))
        
        return action
    
    def spread_out_reward(self, info):
        # Spread reward out on "active" area where it was inside trade
        # Need buy index, sold index and gevinst for trade
        # info: (pris, buy_ix, sell_ix, gevinst)
        # reward_history = [0,0,0,1(buy),2,3,4,5(sell)]
        # KAN være indekseringen er 1 feil i fronten men tror ikke
        pris, buy_indexes, sell_indexes, gevinster = info
        look_back = 30
        #returns = []
        
        #self.reward_history[buy_index[0] - look_back]
        for buy_ix, sell_ix in zip(buy_indexes, sell_indexes):
            gevinst = self.reward_history[sell_ix- look_back]
            ny_gevinster = [gevinst/j for j in range(sell_ix-buy_ix+1, 0, -1)]
            for i, ix in enumerate(range(buy_ix, sell_ix)):
                self.reward_history[ix - look_back] = ny_gevinster[i]
            
        #self.reward_history[sell_index[2]- look_back]
        
    def punish_shopping(self, info):
        # Takes total earning that round and takes minus the cost of sales
        # 0,08% minst
        
        epsilon = 0.08*100
        
        pris, buy_indexes, sell_indexes, gevinster = info
        look_back = 30
        for buy_ix, sell_ix in zip(buy_indexes, sell_indexes):
            #eps = sum(self.reward_history[buy_ix - look_back:sell_ix - look_back])*epsilon
            eps = epsilon
            # Sum all gevinst from buy_ix to sell_ix
            if sell_ix != buy_ix:
                gevinst_straff = eps/(sell_ix-buy_ix)
            else:
                gevinst_straff = eps
            
            for i, ix in enumerate(range(buy_ix, sell_ix)):
                self.reward_history[ix - look_back] -= gevinst_straff

        
        
    
    def discount_reward(self):
        gamma = 0.99
        eps = np.finfo(np.float32).eps.item() 
        returns = []
        discounted_sum = 0
        for r in self.reward_history[::-1]:
            discounted_sum = r + gamma * discounted_sum
            returns.insert(0, discounted_sum)

        # Normalize
        returns = np.array(returns)
        returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
        returns = returns.tolist()
        self.reward_history = returns
        
        
    def learn(self, tape):
        history = zip(self.action_probs_history, self.critic_value_history, self.reward_history)
        actor_losses = []
        critic_losses = []
        for log_prob, value, ret in history:
            # At this point in history, the critic estimated that we would get a
            # total reward = `value` in the future. We took an action with log probability
            # of `log_prob` and ended up recieving a total reward = `ret`.
            # The actor must be updated so that it predicts an action that leads to
            # high rewards (compared to critic's estimate) with high probability.
            diff = ret - value
            actor_losses.append(-log_prob * diff)  # actor loss

            # The critic must be updated so that it predicts a better estimate of
            # the future rewards.
            critic_losses.append(
                self.huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))
            )
            
        # Backpropagation
        loss_value = sum(actor_losses) + sum(critic_losses)
        grads = tape.gradient(loss_value, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        # Clear the loss and reward history
        self.action_probs_history.clear()
        self.critic_value_history.clear()
        self.reward_history.clear()
        
    def save_model(self, fname):
        self.model.save(fname)
    
    def load_model(self, fname):
        self.model = keras.models.load_model(fname)
        self.model.compile(optimizer='adam', loss='mse')


        