[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"
[image2]: /home/horst/ml/udacity/udacity-deep-RL/p1_navigation/training_results_ddqn.png "Training results"

# Project 1: Navigation - Readme

### Introduction & goal of the project

The objective of this project is to train an agent to navigate (and collect bananas!) in a large, square world. It's a Unity environment provided by Udacity.  

![Trained Agent][image1]

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

### Settings

**Implementation:** The project is implemented in `python 3.6` and within a `jupyter notebook`. 

**`Pytorch 1.4.0`** has been used with **`CUDAToolkit 10.1`**.
The operating system is `ubuntu 18.04`.
Unity 2019.3 has been installed on the computer.
The Unity environment is `Banana_Linux/Banana.x86_64` and has been provided by Udacity. 

**Train the agent by executing the whole jupyter notebook from the beginning to the end**