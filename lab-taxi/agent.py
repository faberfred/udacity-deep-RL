import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.eps_start = 0.2            # starting value of epsilon
        self.eps_decay = 0.999          # decay rate of epsilon -> exploration is increased the closer this value is to 1. 
        self.eps_min = 0.0001           # min value of epsilon -> the smaler this value is the more exploitation us done in the limit
        self.eps = self.eps_start
        self.alpha = 0.05
        self.gamma = 0.95

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        self.eps = max(self.eps * self.eps_decay, self.eps_min)
#        self.eps = 0.005
        
        if np.random.random_sample() > self.eps:
            return np.argmax(self.Q[state])
        else:
            return np.random.choice(np.arange(self.nA))

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        self.old_qsa = self.Q[state][action]
        
        # Q-learning 
        self.Q[state][action] = self.old_qsa + self.alpha * (reward + self.gamma * np.max(self.Q[next_state]) - self.old_qsa)
        
        # SARSA learning
        #self.Q[state][action] = self.old_qsa + self.alpha * (reward + self.gamma * self.Q[next_state][self.select_action(next_state)] - self.old_qsa)
        
        # Expected-SARSA learning
#        self.prob = np.ones(self.nA) * self.eps / self.nA
#        self.prob[np.argmax(self.Q[next_state])] = 1 - self.eps + (self.eps / self.nA)
#        self.expected_value = np.dot(self.Q[next_state], self.prob)
#        
#        self.Q[state][action] = self.old_qsa + self.alpha * (reward + self.gamma * self.expected_value - self.old_qsa)
        