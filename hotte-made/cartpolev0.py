#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 08:10:12 2020

@author: horst
"""

import gym
env = gym.make('CartPole-v1')
print('action space: ', env.action_space)
print('observation space', env.observation_space)
print('upper limit of observation space', env.observation_space.high)
print('lower limit of observation space', env.observation_space.low)
for i_episode in range(20): # 20 episodes
    observation = env.reset() # start the process by calling reset() which returns an initial observation
    for t in range(100): # number of steps within an episode
        env.render()
        print(observation)
        action = env.action_space.sample() # take a random action
        observation, reward, done, info = env.step(action) # next step with its return values
        if done:
            print('Episode finished after {} timesteps.'.format(t+1))
            break
env.close()
