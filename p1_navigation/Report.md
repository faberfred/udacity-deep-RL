[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"
[image2]: /home/horst/ml/udacity/udacity-deep-RL/p1_navigation/training_results_ddqn.png "Training results"

# Project 1: Navigation - Report

### Settings

**Implementation:** The project is implemented in `python 3.6` and within a `jupyter notebook`. 
The following packages where imported:
```python
from unityagents import UnityEnvironment
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import time
import random

from collections import namedtuple, deque

import matplotlib.pyplot as plt
%matplotlib inline
```
**`Pytorch 1.4.0`** has been used with **`CUDAToolkit 10.1`**.
The operating system is `ubuntu 18.04`.
Unity 2019.3 has been installed on the computer.
The Unity environment is `Banana_Linux/Banana.x86_64` and has been provided by Udacity. 

### Model

The project has been solved with the help of a deep neural network consisting of **5 linear layers**:
- one input layer of size 37 (=size of the state space).
- three hidden layers of size 128 each.
- one output layer of size 4 (= size of the action space).
The activateion function is **Relu**.
The **dropout probability is 0,1** (=10%).

### Agent

The project has been solved with a **Double-DQN** agent. 
- The replay buffer size is 100000
- The batch size is 32
- Gamma = 0.99
- Tau = 0.01 (for soft update of target parameters)
- Learning rate = 0.0005
- The weights will be updated every 4 steps
The Implementation of the Double-DQN agent is done according to [`Mnih et al., 2015`](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) and  [`van Hasselt et al., 2015`](https://arxiv.org/pdf/1509.06461.pdf)

### Training

The maximum number of training episodes is 2000.
The maximum number of timesteps per episode is 1000.
The starting value of epsilon, for epsilon-greedy action selection is 1.0.
The minimum value of epsilon is 0.1
The multiplicative factor (per episode) for decreasing epsilon is 0.995.

The result of the training is shown below: 

![Training results][image2]

For training purposses the score has been raised to +14.