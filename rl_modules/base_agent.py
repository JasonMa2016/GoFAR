import torch
import os
from datetime import datetime
import numpy as np
from numpngw import write_apng
import matplotlib.pyplot as plt
from mpi4py import MPI
import threading
import time 
from tqdm import tqdm
from PIL import Image 
import wandb 

from rl_modules.replay_buffer import replay_buffer
from her_modules.her import her_sampler
from mpi_utils.normalizer import normalizer
from mpi_utils.mpi_utils import sync_networks, sync_grads, discounted_return

def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class BaseAgent:
    def __init__(self, args, env, env_params):
        self.args = args
        self.env = env
        self.env_params = env_params
        
        self.num_episodes = 0

        # reward function
        if self.args.threshold == 0.05:
            # default compute_reward for Fetch environment takes threshold of 0.05
            compute_reward = self.env.compute_reward
        else:
            compute_reward = self.compute_reward 
        # her sampler
        self.her_module = her_sampler(self.args.replay_strategy, self.args.relabel_percent, compute_reward)
        # create the replay buffer
        self.buffer = replay_buffer(self.env_params, self.args.buffer_size, self.her_module.sample_her_transitions)

        # create the normalizer
        self.o_norm = normalizer(size=env_params['obs'], default_clip_range=self.args.clip_range)
        self.g_norm = normalizer(size=env_params['goal'], default_clip_range=self.args.clip_range)
        # create the dict for store the model
        if MPI.COMM_WORLD.Get_rank() == 0:
            if not os.path.exists(self.args.save_dir):
                os.mkdir(self.args.save_dir)
            # path to save the model
            self.model_path = os.path.join(self.args.save_dir, self.args.env)
            if not os.path.exists(self.model_path):
                os.mkdir(self.model_path)
            self.model_path = os.path.join(self.model_path, self.args.run_name)

            if not os.path.exists(self.model_path):
                os.mkdir(self.model_path)  

    # hack reward function for Fetch which allows us to pass in different threshold values
    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal, goal)
        return -(d > self.args.threshold).astype(np.float32)

    # pre_process the inputs
    def _preproc_inputs(self, obs, g):
        obs_norm = self.o_norm.normalize(obs)
        g_norm = self.g_norm.normalize(g)
        # concatenate the stuffs
        inputs = np.concatenate([obs_norm, g_norm])
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        if self.args.cuda:
            inputs = inputs.cuda()
        return inputs

    # update the normalizer
    def _update_normalizer(self, episode_batch):
        mb_obs, mb_ag, mb_g, mb_actions = episode_batch
        mb_obs_next = mb_obs[:, 1:, :]
        mb_ag_next = mb_ag[:, 1:, :]
        # get the number of normalization transitions
        num_transitions = mb_actions.shape[1]
        # create the new buffer to store them
        buffer_temp = {'obs': mb_obs, 
                       'ag': mb_ag,
                       'g': mb_g, 
                       'actions': mb_actions, 
                       'obs_next': mb_obs_next,
                       'ag_next': mb_ag_next,
                       }
        # This can create an issue if not updated
        transitions = self.her_module.sample_her_transitions(buffer_temp, num_transitions)
        transitions = transitions['transitions']
        obs, g = transitions['obs'], transitions['g']
        # pre process the obs and g
        transitions['obs'], transitions['g'] = self._preproc_og(obs, g)
        # update
        self.o_norm.update(transitions['obs'])
        self.g_norm.update(transitions['g'])
        # recompute the stats
        self.o_norm.recompute_stats()
        self.g_norm.recompute_stats()

    def _preproc_og(self, o, g):
        o = np.clip(o, -self.args.clip_obs, self.args.clip_obs)
        g = np.clip(g, -self.args.clip_obs, self.args.clip_obs)
        return o, g

    # soft update
    def _soft_update(self):
        self._soft_update_target_network(self.actor_target_network, self.actor_network)
        self._soft_update_target_network(self.critic_target_network, self.critic_network)

    def _soft_update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - self.args.polyak) * param.data + self.args.polyak * target_param.data)

    def _update_network(self):
        pass

    def _update_discriminator(self):
        pass 
    
    def _check_discriminator(self):
        pass 

    def sample_batch(self, future_p=None):
        sample_batch = self.buffer.sample(self.args.batch_size, future_p=future_p)
        transitions = sample_batch['transitions']
        # pre-process the observation and goal
        o, o_next, g, = transitions['obs'], transitions['obs_next'], transitions['g']
        transitions['initial_obs'] = np.clip(transitions['initial_obs'], -self.args.clip_obs, self.args.clip_obs)
        transitions['obs'], transitions['g'] = self._preproc_og(o, g)
        transitions['obs_next'], transitions['g_next'] = self._preproc_og(o_next, g)

        sample_batch['transitions'] = transitions

        self.o_norm.update(transitions['obs'])
        self.g_norm.update(transitions['g'])
        self.o_norm.recompute_stats()
        self.g_norm.recompute_stats()

        return sample_batch

    def collect_rollouts(self):
        mb_obs, mb_ag, mb_g, mb_actions = [], [], [], []
        for _ in range(self.args.num_rollouts_per_mpi):
            self.num_episodes += 1

            # reset the rollouts
            ep_obs, ep_ag, ep_g, ep_actions = [], [], [], []
            # reset the environment
            observation = self.env.reset()
            obs = observation['observation']
            ag = observation['achieved_goal']
            g = observation['desired_goal']

            # start to collect samples
            for t in range(self.env_params['max_timesteps']-1):
                with torch.no_grad():
                    input_tensor = self._preproc_inputs(obs, g)
                    action = self._stochastic_actions(input_tensor)
                # feed the actions into the environment
                observation_new, _, _, info = self.env.step(action)
                obs_new = observation_new['observation']
                ag_new = observation_new['achieved_goal']
                # append rollouts
                ep_obs.append(obs.copy())
                ep_ag.append(ag.copy())
                ep_g.append(g.copy())
                ep_actions.append(action.copy())
                # re-assign the observation
                obs = obs_new
                ag = ag_new

            ep_obs.append(obs.copy())
            ep_ag.append(ag.copy())
            mb_obs.append(ep_obs)
            mb_ag.append(ep_ag)
            mb_g.append(ep_g)
            mb_actions.append(ep_actions)
        # convert them into arrays
        mb_obs = np.array(mb_obs)
        mb_ag = np.array(mb_ag)
        mb_g = np.array(mb_g)
        mb_actions = np.array(mb_actions)
        # store the episodes
        self.buffer.store_episode([mb_obs, mb_ag, mb_g, mb_actions])
        self._update_normalizer([mb_obs, mb_ag, mb_g, mb_actions])

    def _eval_agent(self, make_gif=False, epoch=0):
        total_obs, total_g, total_ag, total_rewards, total_success_rate = [], [], [], [], []
        for i in range(self.args.n_test_rollouts):
            per_obs, per_g, per_ag, per_rewards, per_success_rate = [], [], [], [], []

            observation = self.env.reset()
            obs = observation['observation']
            ag = observation['achieved_goal']
            g = observation['desired_goal']
            imgs = []
            for _ in range(self.env_params['max_timesteps']):
                with torch.no_grad():
                    input_tensor = self._preproc_inputs(obs, g)
                    actions = self._deterministic_action(input_tensor)
                    # convert the actions
                    actions = actions.detach().cpu().numpy().squeeze()
                observation_new, reward, _, info = self.env.step(actions)
                if 'score/success' in info:
                    info['is_success'] = float(info['score/success'])
                if self.args.env.startswith('DClaw'):
                    reward = float(info['score/success']) # hack to get DClawTurn to return sparse reward
                if make_gif:
                    img = self.env.render("rgb_array")
                    # imgs.append(Image.fromarray(img))
                    imgs.append(img)
                per_obs.append(obs)
                per_g.append(g)
                per_ag.append(ag)
                per_rewards.append(reward)
                per_success_rate.append(info['is_success'])

                obs = observation_new['observation']
                ag = observation_new['achieved_goal']
                g = observation_new['desired_goal']

            total_obs.append(per_obs)
            total_g.append(per_g)
            total_ag.append(per_ag)
            total_rewards.append(per_rewards)
            total_success_rate.append(per_success_rate)

            if make_gif:
                imgs = np.array(imgs)
                os.makedirs(f"policy_gifs/{self.args.env_id}", exist_ok=True)
                os.makedirs(f"policy_gifs/{self.args.env_id}/{self.args.run_name}", exist_ok=True)
                # imgs[0].save(f"policy_gifs/{self.args.env_id}/{self.args.run_name}/test{i}.gif", save_all=True,
                # append_images=imgs[1:], duration=10, loop=0)    
                write_apng(f"policy_gifs/{self.args.env_id}/{self.args.run_name}/epoch{epoch}-test{i}.png", imgs, delay=40)
        
        total_obs = np.array(total_obs)
        total_g = np.array(total_g)
        total_ag = np.array(total_ag)
        total_rewards = np.array(total_rewards)
        total_success_rate = np.array(total_success_rate)
        dis_return, undis_return = discounted_return(total_rewards, self.args.gamma)

        local_discounted_return = np.mean(dis_return)
        global_discounted_return = MPI.COMM_WORLD.allreduce(local_discounted_return, op=MPI.SUM) / MPI.COMM_WORLD.Get_size()

        local_undiscounted_return = np.mean(undis_return)
        global_undiscounted_return = MPI.COMM_WORLD.allreduce(local_undiscounted_return, op=MPI.SUM) / MPI.COMM_WORLD.Get_size()

        local_distances = np.mean(np.linalg.norm(total_ag[:, -1] - total_g[:, -1], axis=1))
        global_distances = MPI.COMM_WORLD.allreduce(local_distances, op=MPI.SUM) / MPI.COMM_WORLD.Get_size()

        local_success_rate = np.mean(total_success_rate[:, -1])
        global_success_rate = MPI.COMM_WORLD.allreduce(local_success_rate, op=MPI.SUM) / MPI.COMM_WORLD.Get_size()
        results = {'Test/final_distance': global_distances, 
                   'Test/success_rate': global_success_rate,
                   'Test/discounted_return': global_discounted_return,
                   'Test/undiscounted_return': global_undiscounted_return}
        return results

    def learn(self, evaluate_agent=True):
        load_path_expert = f'offline_data/expert/{self.args.env}/'
        load_path_random = f'offline_data/random/{self.args.env}/'
        
        buffer_name = 'buffer'
        buffer_name = f'buffer-noise{self.args.noise_eps}' if self.args.noise else 'buffer'
        
        # load offline data
        if self.args.expert_percent == 0.:
            self.buffer.load(os.path.join(load_path_random, f'{buffer_name}.pkl'))
        elif self.args.random_percent == 0.:
            self.buffer.load(os.path.join(load_path_expert, f'{buffer_name}.pkl'))
        else:
            self.buffer.load_mixture(os.path.join(load_path_expert, f'{buffer_name}.pkl'), os.path.join(load_path_random, f'{buffer_name}.pkl'), self.args.expert_percent, self.args.random_percent,
            self.args)
        
        # start to collect samples
        for epoch in range(self.args.n_epochs):
            future_p = self.her_module.future_p  
            
            # do evaluation
            if evaluate_agent:
                if MPI.COMM_WORLD.Get_rank() != 0:
                    results = self._eval_agent(make_gif=False, epoch=epoch)
                else:
                    results = self._eval_agent(make_gif=epoch % 10 == 0, epoch=epoch)
                if MPI.COMM_WORLD.Get_rank() == 0:
                    # total_episodes = MPI.COMM_WORLD.allreduce(self.num_episodes, op=MPI.SUM)
                    total_episodes = self.num_episodes
                    results.update({'future_p': future_p, 'epoch':epoch, 'episode': total_episodes, 'step': total_episodes*self.env_params['max_timesteps']})
                    wandb.log(results)
                    print('[{}] epoch is: {}, eval success rate is: {:.3f}, final_distance is: {:.3f}'.format(datetime.now(), epoch, results['Test/success_rate'], results['Test/final_distance']))

                    torch.save([self.o_norm.mean, self.o_norm.std, self.g_norm.mean, self.g_norm.std, self.actor_network], \
                                    self.model_path + f'/{self.args.run_name}-Epoch{epoch}.pt')

            # train discriminator
            # if self.args.use_disc:
            #     for _ in range(self.args.disc_iter):
            #         self._update_discriminator(future_p=future_p)
                    
            # do training
            for _ in tqdm(range(self.args.n_cycles)):
                # train discriminator
                if self.args.use_disc:
                    for _ in range(self.args.disc_iter):
                        self._update_discriminator(future_p=future_p)
                # train policy
                for _ in range(self.args.n_batches):
                    self._update_network(future_p=future_p)
                
                self._soft_update()

        # end-of-training things
        if MPI.COMM_WORLD.Get_rank() == 0:
            results = self._eval_agent(make_gif=True, epoch=epoch)
            # save the model 
            torch.save([self.o_norm.mean, self.o_norm.std, self.g_norm.mean, self.g_norm.std, self.actor_network], \
            self.model_path + f'/{self.args.run_name}.pt')