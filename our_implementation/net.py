import torch
from torch import nn
import numpy as np
import torch.nn.functional as F


class ActorCriticNetwork(nn.Module):
    """
        Actor-Critic network with variable layer sizes.
    """

    def __init__(self, shared_dims, value_dims, policy_dims):
        super(ActorCriticNetwork, self).__init__()
        self.shared_layers = nn.ModuleList()
        [self.shared_layers.append(nn.Linear(shared_dims[i], shared_dims[i+1])) for i in range(len(shared_dims)-1)]

        self.value_layers = nn.ModuleList()
        self.value_layers.append(nn.Linear(shared_dims[-1], value_dims[0]))
        [self.value_layers.append(nn.Linear(value_dims[i], value_dims[i+1])) for i in range(len(value_dims)-1)]

        self.policy_layers = nn.ModuleList()
        self.policy_layers.append(nn.Linear(shared_dims[-1], policy_dims[0]))
        [self.policy_layers.append(nn.Linear(policy_dims[i], policy_dims[i + 1])) for i in range(len(policy_dims) - 1)]


    def forward(self, obs, calculate_policy=True):
        """
            Runs a forward pass on the neural network.
            Parameters:
                obs - observation to pass as input
            Return:
                output - the output of our forward pass
        """
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)

        act = obs
        for layer in self.shared_layers:
            act = F.relu(layer(act))
        for layer in (self.policy_layers[:-1] if calculate_policy else self.value_layers[:-1]):
            act = F.relu(layer(act))
        out = (self.policy_layers[-1] if calculate_policy else self.value_layers[-1])(act)
        return out
