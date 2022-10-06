import torch
import torch.nn.functional as F
import os
from datetime import datetime
import numpy as np
from mpi4py import MPI
from mpi_utils.mpi_utils import sync_networks, sync_grads
from rl_modules.base_agent import BaseAgent
from rl_modules.models import actor, critic
from rl_modules.discriminator import Discriminator

"""
GCSL (MPI-version)

"""
class GCSL(BaseAgent):
    def __init__(self, args, env, env_params):
        super().__init__(args, env, env_params) 
        # create the network
        self.actor_network = actor(env_params)
        self.critic_network = critic(env_params)
        # sync the networks across the cpus
        sync_networks(self.actor_network)
        sync_networks(self.critic_network)
        # build up the target network
        self.actor_target_network = actor(env_params)
        self.critic_target_network = critic(env_params)
        # load the weights into the target networks
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())

        self.discriminator = Discriminator(2 * env_params['goal'], lr=args.lr_critic)

        # if use gpu
        if self.args.cuda:
            self.actor_network.cuda()
            self.critic_network.cuda()
            self.actor_target_network.cuda()
            self.critic_target_network.cuda()
            self.discriminator.cuda()

        # create the optimizer
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.lr_critic)

    # this function will choose action for the agent and do the exploration
    def _stochastic_actions(self, input_tensor):
        pi = self.actor_network(input_tensor)
        action = pi.cpu().numpy().squeeze()

        # add the gaussian
        action += self.args.noise_eps * self.env_params['action_max'] * np.random.randn(*action.shape)
        action = np.clip(action, -self.env_params['action_max'], self.env_params['action_max'])
        # random actions...
        random_actions = np.random.uniform(low=-self.env_params['action_max'], high=self.env_params['action_max'], \
                                            size=self.env_params['action'])
        # choose if use the random actions
        action += np.random.binomial(1, self.args.random_eps, 1)[0] * (random_actions - action)
        return action
    
    def _deterministic_action(self, input_tensor):
        action = self.actor_network(input_tensor)
        return action

    # update the network
    def _update_network(self, future_p=None):
        # sample the episodes
        sample_batch = self.sample_batch(future_p=future_p)
        transitions = sample_batch['transitions'] 

        # start to do the update
        obs_norm = self.o_norm.normalize(transitions['obs'])
        ag_norm = self.g_norm.normalize(transitions['ag'])
        g_norm = self.g_norm.normalize(transitions['g'])
        
        inputs_norm = np.concatenate([obs_norm, g_norm], axis=1)
        obs_next_norm = self.o_norm.normalize(transitions['obs_next'])
        ag_next_norm = self.g_norm.normalize(transitions['ag_next'])
        g_next_norm = self.g_norm.normalize(transitions['g_next'])
        inputs_next_norm = np.concatenate([obs_next_norm, g_next_norm], axis=1)
        
        # transfer them into the tensor
        inputs_norm_tensor = torch.tensor(inputs_norm, dtype=torch.float32)
        inputs_next_norm_tensor = torch.tensor(inputs_next_norm, dtype=torch.float32)
        actions_tensor = torch.tensor(transitions['actions'], dtype=torch.float32)
        r_tensor = torch.tensor(transitions['r'], dtype=torch.float32) 

        if self.args.reward_type == 'positive':
            r_tensor = r_tensor + 1.
        elif self.args.reward_type == 'square':
            # Question: does it make sense to do this here?
            r_tensor = - torch.tensor(np.linalg.norm(ag_next_norm-g_norm, axis=1) ** 2, dtype=torch.float32).unsqueeze(1)

        if self.args.cuda:
            inputs_norm_tensor = inputs_norm_tensor.cuda()
            inputs_next_norm_tensor = inputs_next_norm_tensor.cuda()
            actions_tensor = actions_tensor.cuda()
            r_tensor = r_tensor.cuda()

        # Compute the actions
        actions_real = self.actor_network(inputs_norm_tensor)

        # calculate the target Q value function
        if self.args.method == 'wgcsl' or self.args.method == 'wgcbc':
            offset = sample_batch['future_offset']
            weights = pow(self.args.gamma, offset)  
            weights = torch.tensor(weights[:, None]).to(actions_tensor.device)
            with torch.no_grad():
                actions_next = self.actor_target_network(inputs_next_norm_tensor)
                q_next_value = self.critic_target_network(inputs_next_norm_tensor, actions_next)
                q_next_value = q_next_value.detach()
                target_q_value = r_tensor + self.args.gamma * q_next_value
                target_q_value = target_q_value.detach()
                # clip the q value
                clip_return = 1 / (1 - self.args.gamma)
                target_q_value = torch.clamp(target_q_value, -clip_return, 0)

            # the q loss
            real_q_value = self.critic_network(inputs_norm_tensor, actions_tensor)
            critic_loss = (target_q_value - real_q_value).pow(2).mean()

            # Compute the advantage weighting
            with torch.no_grad():
                v = self.critic_network(inputs_norm_tensor, actions_real)
                v = torch.clamp(v, -clip_return, 0)
                adv = target_q_value - v
                adv = torch.clamp(torch.exp(adv.detach()), 0, 10)
            weights = weights * adv
        else:
            weights = torch.ones(actions_tensor.shape).to(actions_tensor.device)
        
        actor_loss = torch.mean(weights * torch.square(actions_real - actions_tensor))

        # update the actor network
        self.actor_optim.zero_grad()
        actor_loss.backward()
        sync_grads(self.actor_network)
        self.actor_optim.step()

        # update the critic_network
        if self.args.method == 'wgcsl':
            self.critic_optim.zero_grad()
            critic_loss.backward()
            sync_grads(self.critic_network)
            self.critic_optim.step()
