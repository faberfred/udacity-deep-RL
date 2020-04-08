import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork
# from dynamic_model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network -> this is variable C within the DQN paper

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        """Update the relay memory and update everey UPDATE_EVERY steps the weights
        
        Params
        ======
        
            state (): 
            action():
            reward ():
            next_state ():
            done ():
        
        
        """
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)
                # return statement for debug purposes
                # return(self.learn(experiences, GAMMA))

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        # get the state as a torch tensor
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        # set network into evaluation mode -> turn off dropout!
        self.qnetwork_local.eval()
        # turn off gradient computation outside training mode -> safes memory & computations
        with torch.no_grad():
            # get the action values due to a forward pass of the state through the network!
            action_values = self.qnetwork_local(state)
        # set network back into training mode -> turn on dropout!
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            # return the action with the biggest q-value
            return np.argmax(action_values.cpu().data.numpy())
        else:
            # return a random uniformly distributed value out of the action space
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.
        
           Deep-Q-Network implementation according to Mnih et al., 2015

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # compute and minimize the loss
        
        """ Compute Q targets for next states 
        
            Get the max predicted Q values for the next states target model / target network
        
            detach(): return a new Tensor (it does not change the current one) that does not share the history of the 
            original Tensor / detached from the current graph.(no autograd)
            .detach() returns a new tensor without history!
            
            torch.max(input, dim, keepdim=False, out=None) -> (Tensor, LongTensor): Returns a namedtuple (values, indices) where values is the maximum 
            value of each row of the input tensor in the given dimension dim. And indices is the index location of each maximum value found (argmax).
            .max(1)[0] returns the maximum value of the 1st dimension
            
            torch.unsqueeze(input, dim, out=None) → Tensor: Returns a new tensor with a dimension of size one inserted at the specified position.
            .unsqueeze(1) transform the data from a row-vector into a column vector
            
            Q_targets_next is a column vector of size BATCH_SIZE with the max action values from each forwarded next_state!
       
        """
        
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Q_targets_next_test = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        
        """ Compute Q targets for current states:
            
            Use Q_targets_next to compute the current Q targets
        
            if done == 1 there will be no next state -> Q_targets is just the reward
            otherwise it's the immediate reward + discount factor * the estimated Q_target of the next state
            Q_targets is a column vector of size BATCH_SIZE with the computed values of the immediate reward and the Q-values (action-values) of the next state
            
        """
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        """ Get expected Q values from local model
        
            Q_expected_full_values has the Q-values / action values of a pass trough / forward run throug the network qnetwork_local
            Q_expected_full_values has dimension 0 = BATCH_SIZE and dimension 1 = size of action space
            
            torch.gather(input, dim, index, out=None, sparse_grad=False) → Tensor: Gathers values along an axis specified by dim.
            gather(1, actions) extracts the values out of Q_expected_full_values at the position specified in action! 
            If action has value 3 -> get the value from index 3 out of Q_expected_full_values
            Q_expected has dimension 0 = BATCH_SIZE and dimension 1 = 1
        
        """
        # for debug purposes:
        # Q_expected_test = self.qnetwork_local(states).gather(1, actions)
        
        # Get all Q-values of qnetwork_local
        Q_expected_full_values = self.qnetwork_local(states)
        
        # Extract the Q-values accorting to actions (=indices of the values)
        Q_expected = Q_expected_full_values.gather(1, actions)
        
        # The two previous steps can be done within one step
        # Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss -> use the mean sqared error loss function (from torch.nn.functional)
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        # Clear the gradients, because backward accumulates the gradients and so they have to be cleared
        self.optimizer.zero_grad()
        # calculate the gradients
        loss.backward()
        # update the weights / parameter
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)          

        # return statement for debug purposes
        # return(Q_expected_test)                  

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)