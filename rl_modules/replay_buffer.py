import threading
import pickle
import numpy as np
import torch 

"""
the replay buffer here is basically from the openai baselines code

"""
class replay_buffer:
    def __init__(self, env_params, buffer_size, sample_func):
        self.env_params = env_params
        self.T = env_params['max_timesteps']
        self.size = buffer_size // self.T
        # memory management
        self.current_size = 0
        self.n_transitions_stored = 0
        self.sample_func = sample_func
        # create the buffer to store info
        # self.buffers = {'obs': np.empty([self.size, self.T+1, self.env_params['obs']]),
        #                 'ag': np.empty([self.size, self.T+1, self.env_params['goal']]),
        #                 'g': np.empty([self.size, self.T, self.env_params['goal']]),
        #                 'actions': np.empty([self.size, self.T, self.env_params['action']]),
        #                 }

        self.buffers = {'obs': np.empty([self.size, self.T, self.env_params['obs']]),
                        'ag': np.empty([self.size, self.T, self.env_params['goal']]),
                        'g': np.empty([self.size, self.T-1, self.env_params['goal']]),
                        'actions': np.empty([self.size, self.T-1, self.env_params['action']]),
                        }
        self.key_map = {'o': 'obs', 'ag': 'ag', 'g': 'g', 'u':'actions'}

        # thread lock
        self.lock = threading.Lock()
    
    # store the episode
    def store_episode(self, episode_batch):
        mb_obs, mb_ag, mb_g, mb_actions = episode_batch
        batch_size = mb_obs.shape[0]
        with self.lock:
            idxs = self._get_storage_idx(inc=batch_size)
            # store the informations
            self.buffers['obs'][idxs] = mb_obs
            self.buffers['ag'][idxs] = mb_ag
            self.buffers['g'][idxs] = mb_g
            self.buffers['actions'][idxs] = mb_actions
            self.n_transitions_stored += self.T * batch_size
    
    def shuffle_goals(self):
        np.random.shuffle(self.buffers['g'][:self.current_size])
    # sample the data from the replay buffer
    def sample(self, batch_size, future_p=None):
        temp_buffers = {}
        with self.lock:
            for key in self.buffers.keys():
                temp_buffers[key] = self.buffers[key][:self.current_size]
        temp_buffers['initial_obs'] = temp_buffers['obs'][:, :1, :]
        temp_buffers['obs_next'] = temp_buffers['obs'][:, 1:, :]
        temp_buffers['ag_next'] = temp_buffers['ag'][:, 1:, :]
        # sample transitions
        transitions = self.sample_func(temp_buffers, batch_size, future_p=future_p)
        return transitions

    def select_goal(self, s0, g, V):
        achieved_goals = np.reshape(self.buffers['ag'], [self.size * (self.T), self.env_params['goal']])
        candidate_idx = np.random.randint(0, self.n_transitions_stored, 1024)
        goal_candidates = achieved_goals[candidate_idx]

        goal_dist = np.linalg.norm(goal_candidates - g, axis=-1)
        s0_tile = np.tile(s0[None], [goal_candidates.shape[0], 1])
        input_V = torch.tensor(np.concatenate([s0_tile, goal_candidates], axis=1), dtype=torch.float32).cuda() # should check for cuda instead

        with torch.no_grad():
            goal_value = V(input_V).cpu().detach().numpy()
        goal_score = goal_dist[:, None] - goal_value
        selected_goal_idx = np.argmin(goal_score, axis=0)
        selected_goal = goal_candidates[selected_goal_idx[0]]

        return selected_goal

    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        if self.current_size+inc <= self.size:
            idx = np.arange(self.current_size, self.current_size+inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)
        self.current_size = min(self.size, self.current_size+inc)
        if inc == 1:
            idx = idx[0]
        return idx


    def save(self, path):
        with open(path, "wb") as fp:
            data = {} 
            for key in self.key_map:
                data[key] = self.buffers[self.key_map[key]][:self.n_transitions_stored]
            pickle.dump(data, fp)

    def load(self, path, percent=1.0):
        with open(path, "rb") as fp:  
            data = pickle.load(fp)
            size = data['o'].shape[0]
            self.current_size = int(size * percent)
            # if size > self.size:
            #     self.buffers = {key: np.empty([size, *shape]) for key, shape in self.buffer_shapes.items()}
            #     self.size = size

            for key in data.keys():
                self.buffers[self.key_map[key]][:self.current_size] = data[key][:self.current_size]

    def load_mixture(self, path_expert, path_random, expert_percent=0.1, random_percent=0.9, args=None):
        # 0 <= expert_percent <= 1, same for random_percent
        
        with open(path_expert, "rb") as fp_expert:  
            with open(path_random, "rb") as fp_random:  
                data_expert = pickle.load(fp_expert)  
                data_random = pickle.load(fp_random)  
                size_expert = data_expert['o'].shape[0]
                size_random = data_random['o'].shape[0]
                assert(size_expert == size_random)
                self.current_size = int(size_expert*expert_percent + size_random*random_percent)
                size = self.current_size
                split_point = int(size_expert*expert_percent)
                # if size > self.size:
                #     self.buffers = {key: np.empty([size, *shape]) for key, shape in self.buffer_shapes.items()}
                #     self.size = size
                    
                for key in data_expert.keys():
                    self.buffers[self.key_map[key]][:split_point] = data_expert[key][:split_point]
                    self.buffers[self.key_map[key]][split_point:size] = data_random[key][:size - split_point]
