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
    def __init__(self, n_actions, input_dims, batch_size=64):
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.model = self.make_net()
        self.batch_size = batch_size
        self.critic_value_history = []
        self.action_probs_history = []
        self.action_history = []
        self.reward_history = []
        self.huber_loss = keras.losses.Huber()
        self.optimizer = keras.optimizers.Adam(learning_rate=0.01)
        self.best_score = 0
        
    def make_net(self):
        #inputs = layers.Input(shape=(self.input_dims,))
        #common = layers.Dense(256, activation="relu")(inputs)
        #common = layers.Dense(256, activation="relu")(common)
        
        # For stocks
        inputs = layers.Input(shape=(30,6,))
        x = layers.LSTM(units=128, return_sequences=True)(inputs)
        x = layers.LSTM(units=128)(x)
        common = layers.Dense(64, activation='relu')(x)
        
        
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
        self.action_history.append(action)
        
        return action
    
    def spread_out_reward(self, info):
        # Spread reward out on "active" area where it was inside trade
        # Need buy index, sold index and gevinst for trade
        # info: (pris, buy_ix, sell_ix, gevinst)
        # reward_history = [0,0,0,1(buy),2,3,4,5(sell)]
        # KAN vÃƒÂ¦re indekseringen er 1 feil i fronten men tror ikke
        pris, buy_indexes, sell_indexes, gevinster = info
        look_back = 30
        #returns = []
        
        #self.reward_history[buy_index[0] - look_back]
        for buy_ix, sell_ix in zip(buy_indexes, sell_indexes):
            gevinst = self.reward_history[sell_ix- look_back]
            ny_gevinster = [gevinst for j in range(sell_ix-buy_ix+1, 0, -1)]
            for i, ix in enumerate(range(buy_ix, sell_ix)):
                self.reward_history[ix - look_back] = ny_gevinster[i]
            
        #self.reward_history[sell_index[2]- look_back]
        
    def punish_shopping(self):
        # Prøv å unngå å bruke info men bruk det som er lagret i agent allerede
        cost = 15
        buy_index = 0
        status = 0
        for i, j in enumerate(zip(self.action_history, self.reward_history)):
            action, reward = j
            if action and not status:
                status = 1
                buy_index = i
        
            elif status and not action:
                # Sells and do everything here
                status = 0
                self.reward_history[buy_index:i+1] -= (cost/(i+1-buy_index))
                
        self.reward_history = self.reward_history.tolist()
                
        
    def punish_shopping_progresiv(self):
        # Takes total earning that round and takes minus the cost of sales
        # 0,08% minst, men hva er det da??
        # Progresivt!
        
        cost = 8
        buy_index = 0
        status = 0
        self.reward_history = np.array(self.reward_history)
        for i, j in enumerate(zip(self.action_history, self.reward_history)):
            action, reward = j
            if action and not status:
                status = 1
                buy_index = i
        
            elif status and not action:
                # Sells and do everything here
                status = 0
                magnitude = abs(self.reward_history[buy_index:i+1])
                percent = magnitude/(sum(magnitude) + np.finfo(np.float32).eps.item())
                self.reward_history[buy_index:i+1] -= percent*cost
        
        self.reward_history = self.reward_history.tolist()

    
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
        #  == Pick out random from batch ==
        max_size = min(len(self.reward_history), self.batch_size)
        self.indexes = np.random.choice(len(self.reward_history), self.batch_size, replace=False)
        
        # ? Tensors in list need to do thiis..
        history = zip([self.action_probs_history[i] for i in range(len(self.indexes))],
                      [self.critic_value_history[i] for i in range(len(self.indexes))], 
                      [self.reward_history[i] for i in range(len(self.indexes))])
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
        self.action_history.clear()
        
    def save_model(self, fname):
        self.model.save(fname)
    
    def load_model(self, fname):
        self.model = keras.models.load_model(fname)
        self.model.compile(optimizer='adam', loss='mse')


        