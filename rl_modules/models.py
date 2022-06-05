import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical


"""
the input x in both networks should be [o, g], where o is the observation and g is the goal.

"""

# Inverse tanh torch function
def atanh(z):
    return 0.5 * (torch.log(1 + z) - torch.log(1 - z))

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

# define the actor network
class actor(nn.Module):
    def __init__(self, env_params):
        super(actor, self).__init__()
        self.max_action = env_params['action_max']
        self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.action_out = nn.Linear(256, env_params['action'])
        
        self.apply(weights_init_)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        actions = self.max_action * torch.tanh(self.action_out(x))

        return actions

# define the actor network
class actor_inverse(nn.Module):
    def __init__(self, env_params):
        super(actor_inverse, self).__init__()
        self.max_action = env_params['action_max']
        self.fc1 = nn.Linear(3*env_params['goal'], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.action_out = nn.Linear(256, env_params['action'])
        
        self.apply(weights_init_)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        actions = self.max_action * torch.tanh(self.action_out(x))

        return actions

# define the planner network 
class planner(nn.Module):
    def __init__(self, env_params):
        super(planner, self).__init__()
        self.fc1 = nn.Linear(2 * env_params['goal'], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.action_out = nn.Linear(256, env_params['goal'])
        
        self.apply(weights_init_)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        actions = self.action_out(x)

        return actions

class critic(nn.Module):
    def __init__(self, env_params, activation=None):
        super(critic, self).__init__()
        self.activation = activation
        self.max_action = env_params['action_max']
        self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'] + env_params['action'], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.q_out = nn.Linear(256, 1)

        self.apply(weights_init_)

    def forward(self, x, actions):
        x = torch.cat([x, actions / self.max_action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_value = self.q_out(x)
        if self.activation == 'sigmoid':
            q_value = torch.sigmoid(q_value)
        return q_value

class doublecritic(nn.Module):
    def __init__(self, env_params):
        super(doublecritic, self).__init__()
        self.max_action = env_params['action_max']
        self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'] + env_params['action'], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.q1_out = nn.Linear(256, 1)

        self.fc4 = nn.Linear(env_params['obs'] + env_params['goal'] + env_params['action'], 256)
        self.fc5 = nn.Linear(256, 256)
        self.fc6 = nn.Linear(256, 256)
        self.q2_out = nn.Linear(256, 1)

        self.apply(weights_init_)

    def forward(self, x, actions):
        x = torch.cat([x, actions / self.max_action], dim=1)
        x1 = F.relu(self.fc1(x))
        x1 = F.relu(self.fc2(x1))
        x1 = F.relu(self.fc3(x1))
        q1_value = self.q1_out(x1)

        x2 = F.relu(self.fc4(x))
        x2 = F.relu(self.fc5(x2))
        x2 = F.relu(self.fc6(x2))
        q2_value = self.q2_out(x2)

        return q1_value, q2_value
    
    def Q1(self, x, action):
        x = torch.cat([x, action / self.max_action], dim=1)
        x1 = F.relu(self.fc1(x))
        x1 = F.relu(self.fc2(x1))
        x1 = F.relu(self.fc3(x1))
        q1_value = self.q1_out(x1)
        
        return q1_value

class value(nn.Module):
    def __init__(self, env_params):
        super(value, self).__init__()
        self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.q_out = nn.Linear(256, 1)

        self.apply(weights_init_)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_value = self.q_out(x)

        return q_value