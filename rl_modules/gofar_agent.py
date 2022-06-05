from xml.dom import NoModificationAllowedErr
import torch
import torch.nn.functional as F
import os
from datetime import datetime
import numpy as np
from mpi4py import MPI
from mpi_utils.mpi_utils import sync_networks, sync_grads
from rl_modules.base_agent import BaseAgent
from rl_modules.models import actor, critic, value
from rl_modules.discriminator import Discriminator

torch.autograd.set_detect_anomaly(True)

"""
GoFAR (Goal-conditioned f-Advantage Regression)

"""
class GoFAR(BaseAgent):
    def __init__(self, args, env, env_params):
        super().__init__(args, env, env_params) 
        # create the network
        self.actor_network = actor(env_params)
        self.value_network = value(env_params)
        # sync the networks across the cpus
        sync_networks(self.actor_network)
        sync_networks(self.value_network)
        # build up the target network
        self.actor_target_network = actor(env_params)
        self.value_target_network = value(env_params)
        # load the weights into the target networks
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.value_target_network.load_state_dict(self.value_network.state_dict())

        self.discriminator = Discriminator(2 * env_params['goal'], lr=args.lr_critic)

        # if use gpu
        if self.args.cuda:
            self.actor_network.cuda()
            self.value_network.cuda()
            self.actor_target_network.cuda()
            self.value_target_network.cuda()
            self.discriminator.cuda()

        # create the optimizer
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor)
        self.value_optim = torch.optim.Adam(self.value_network.parameters(), lr=self.args.lr_critic)

        if self.args.use_explorer:
            self.explorer_network = actor(env_params)
            self.critic_network = critic(env_params)
            
            self.explorer_optim = torch.optim.Adam(self.explorer_network.parameters(), lr=self.args.lr_actor)
            self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.lr_critic)
            if self.args.cuda:
                self.explorer_network.cuda()
                self.critic_network.cuda() 

    # soft update
    def _soft_update(self):
        self._soft_update_target_network(self.actor_target_network, self.actor_network)
        self._soft_update_target_network(self.value_target_network, self.value_network)

    # this function will choose action for the agent and do the exploration
    def _stochastic_actions(self, input_tensor):
        if self.args.use_explorer:
            pi = self.explorer_network(input_tensor)
        else: 
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
        if self.args.use_explorer:
            action = self.explorer_network(input_tensor)
        else: 
            action = self.actor_network(input_tensor)
        return action

    def _update_discriminator(self, future_p=None):
        # sample the episodes
        sample_batch = self.sample_batch(future_p=future_p)
        transitions = sample_batch['transitions']

        # start to do the update
        ag_norm = self.g_norm.normalize(transitions['ag'])
        g_norm = self.g_norm.normalize(transitions['g'])

        pos_pairs = torch.tensor(np.concatenate([g_norm, g_norm], axis=1), dtype=torch.float32)
        neg_pairs = torch.tensor(np.concatenate([ag_norm, g_norm], axis=1), dtype=torch.float32)

        if self.args.cuda:
            pos_pairs = pos_pairs.cuda()
            neg_pairs = neg_pairs.cuda()

        expert_d = self.discriminator.trunk(pos_pairs)
        policy_d = self.discriminator.trunk(neg_pairs)

        expert_loss = F.binary_cross_entropy_with_logits(
            expert_d,
            torch.ones(expert_d.size()).to(pos_pairs.device))
        policy_loss = F.binary_cross_entropy_with_logits(
            policy_d,
            torch.zeros(policy_d.size()).to(neg_pairs.device))

        gail_loss = expert_loss + policy_loss
        grad_pen = self.discriminator.compute_grad_pen(pos_pairs, neg_pairs, lambda_=self.args.disc_lambda)

        self.discriminator.optimizer.zero_grad()
        (gail_loss + grad_pen).backward()
        self.discriminator.optimizer.step()

    def _check_discriminator(self):
        transitions = self.buffer.sample(self.args.batch_size)
        ag_norm = self.g_norm.normalize(transitions['ag'])
        g_norm = self.g_norm.normalize(transitions['g'])
        goal_pair = torch.tensor(np.concatenate([g_norm, g_norm], axis=1), dtype=torch.float32)
        ag_pair = torch.tensor(np.concatenate([ag_norm, ag_norm], axis=1), dtype=torch.float32)
        diff_pair = torch.tensor(np.concatenate([ag_norm, g_norm], axis=1), dtype=torch.float32)
        if self.args.cuda:
            goal_pair = goal_pair.cuda()
            ag_pair = ag_pair.cuda()
            diff_pair = diff_pair.cuda()
        with torch.no_grad():
            goal_pair_score = self.discriminator.predict_reward(goal_pair).mean().cpu().detach().numpy()
            ag_pair_score = self.discriminator.predict_reward(ag_pair).mean().cpu().detach().numpy() 
            ag_g_score = self.discriminator.predict_reward(diff_pair).mean().cpu().detach().numpy()
        print(f"goal pair: {goal_pair_score:.3f}, ag pair: {ag_pair_score:.3f}, ag-g: {ag_g_score:.3f}")

    # update the network
    def _update_network(self, future_p=None):
        # sample the episodes
        sample_batch = self.sample_batch(future_p=future_p)
        transitions = sample_batch['transitions']
        # start to do the update
        io_norm = self.o_norm.normalize(transitions['initial_obs'])
        obs_norm = self.o_norm.normalize(transitions['obs'])
        ag_norm = self.g_norm.normalize(transitions['ag'])
        g_norm = self.g_norm.normalize(transitions['g'])

        inputs_initial_norm = np.concatenate([io_norm, g_norm], axis=1)
        inputs_norm = np.concatenate([obs_norm, g_norm], axis=1)
        obs_next_norm = self.o_norm.normalize(transitions['obs_next'])
        ag_next_norm = self.g_norm.normalize(transitions['ag_next'])
        g_next_norm = self.g_norm.normalize(transitions['g_next'])
        inputs_next_norm = np.concatenate([obs_next_norm, g_next_norm], axis=1)
        
        # transfer them into the tensor
        inputs_initial_norm_tensor = torch.tensor(inputs_initial_norm, dtype=torch.float32)
        inputs_norm_tensor = torch.tensor(inputs_norm, dtype=torch.float32)
        inputs_next_norm_tensor = torch.tensor(inputs_next_norm, dtype=torch.float32)
        actions_tensor = torch.tensor(transitions['actions'], dtype=torch.float32)
        r_tensor = torch.tensor(transitions['r'], dtype=torch.float32) 
        # r_tensor = - torch.tensor(np.linalg.norm(transitions['ag_next']-transitions['g']), dtype=torch.float32) ** 2

        # obtain discriminator reward
        disc_inputs_norm_tensor = torch.tensor(np.concatenate([ag_norm, g_norm], axis=1), dtype=torch.float32)

        if self.args.reward_type == 'disc':
            if self.args.cuda:
                disc_inputs_norm_tensor = disc_inputs_norm_tensor.cuda()
            r_tensor = self.discriminator.predict_reward(disc_inputs_norm_tensor)
        elif self.args.reward_type == 'positive':
            r_tensor = r_tensor + 1.
        elif self.args.reward_type == 'square':
            r_tensor = - torch.tensor(np.linalg.norm(ag_next_norm-g_norm, axis=1) ** 2, dtype=torch.float32).unsqueeze(1)
        elif self.args.reward_type == 'laplace':
            r_tensor = - torch.tensor(np.linalg.norm(ag_next_norm-g_norm, ord=1, axis=1) ** 2, dtype=torch.float32).unsqueeze(1)

        if self.args.cuda:
            inputs_initial_norm_tensor = inputs_initial_norm_tensor.cuda()
            inputs_norm_tensor = inputs_norm_tensor.cuda()
            inputs_next_norm_tensor = inputs_next_norm_tensor.cuda()
            actions_tensor = actions_tensor.cuda()
            r_tensor = r_tensor.cuda()

        # Calculate value loss
        v_initial = self.value_network(inputs_initial_norm_tensor)
        v_current = self.value_network(inputs_norm_tensor)
        with torch.no_grad():
            v_next = self.value_target_network(inputs_next_norm_tensor).detach()
            v_onestep = (r_tensor + self.args.gamma * v_next).detach()

            # if self.args.reward_type == 'binary':
            # v_onestep = torch.clamp(v_onestep, -clip_return, 0)

        # e_v = r_tensor + self.args.gamma * v_next - v_current
        e_v =  v_onestep - v_current 

        v_loss0 = (1 - self.args.gamma) * v_initial 
        if self.args.f == 'chi':
            v_loss1 = torch.mean((e_v + 1).pow(2))
        elif self.args.f == 'kl':
            v_loss1 = torch.log(torch.mean(torch.exp(e_v)))
        value_loss = (v_loss0 + v_loss1).mean()

        if self.args.use_explorer:
            actions_explorer = self.explorer_network(inputs_norm_tensor)
            # explorer_loss = -torch.log(self.critic_network(inputs_norm_tensor, actions_explorer)).mean()
            explorer_loss = - self.critic_network(inputs_norm_tensor, actions_explorer).mean()

            self.explorer_optim.zero_grad()
            explorer_loss.backward()
            self.explorer_optim.step() 

            critic_loss = (e_v.detach() - self.critic_network(inputs_norm_tensor, actions_tensor)).pow(2).mean()
            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()
        else:
            # Compute policy loss (L2 because Gaussian with fixed sigma)
            if self.args.f == 'chi':
                w_e = torch.relu(e_v + 1).detach()
            elif self.args.f == 'kl':
                w_e = torch.clamp(torch.exp(e_v.detach()), 0, 10)
            actions_real = self.actor_network(inputs_norm_tensor)
            actor_loss = torch.mean(w_e * torch.square(actions_real - actions_tensor))

            # update the actor network
            self.actor_optim.zero_grad()
            actor_loss.backward()
            sync_grads(self.actor_network)
            self.actor_optim.step()

        # update the value_network
        self.value_optim.zero_grad()
        value_loss.backward()
        sync_grads(self.value_network)
        self.value_optim.step()
