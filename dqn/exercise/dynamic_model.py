import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, hidden_layers = [64, 64], drop_p = 0.1):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden_layers (list of int):  list of integers, amount and size of the hidden layers
            drop_p (float): dropout probability
            
        """
        super().__init__()
        self.seed = torch.manual_seed(seed)

        # add the first layer: input to first hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(state_size, hidden_layers[0])])
        
        # add a viariable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        
        # add the last layer: last hidden to output
        self.output = nn.Linear(hidden_layers[-1], action_size)
        
        # dropout probabilities
        self.drop_p = drop_p

    def forward(self, x):
        """Build a network that maps state -> action values."""
        
        # Forward through each layer in the hidden_layers, with ReLU activation and dropouts
        for linear in self.hidden_layers:
            x = linear(x)
            x = F.relu(x)
            x = F.dropout(x, p = self.drop_p)
            
        x = self.output(x)
        
        return(x)
