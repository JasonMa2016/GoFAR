import gym 
import numpy as np
import pickle 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch import autograd
from tqdm import tqdm 

# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/algo/gail.py
class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, lr=5e-4):
        super(Discriminator, self).__init__()


        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1))

        self.trunk.train()

        self.optimizer = torch.optim.Adam(self.trunk.parameters(), lr=5e-4)

    def compute_grad_pen(self,
                         expert_state,
                         offline_state,
                         lambda_=20.):
        alpha = torch.rand(expert_state.size(0), 1)
        expert_data = expert_state 
        offline_data = offline_state

        alpha = alpha.expand_as(expert_data).to(expert_data.device)

        mixup_data = alpha * expert_data + (1 - alpha) * offline_data
        mixup_data.requires_grad = True

        disc = self.trunk(mixup_data)
        ones = torch.ones(disc.size()).to(disc.device)
        grad = autograd.grad(
            outputs=disc,
            inputs=mixup_data,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

        grad_pen = lambda_ * (grad.norm(2, dim=1) - 1).pow(2).mean()
        return grad_pen

    def predict_reward(self, state):
        with torch.no_grad():
            # self.eval()
            d = self.trunk(state)
            s = torch.sigmoid(d)
            reward = s.log() - (1 - s).log()
            return reward 