import numpy as np

class her_sampler:
    def __init__(self, replay_strategy, relabel_percent, reward_func=None):
        self.replay_strategy = replay_strategy
#        self.replay_k = replay_k
        self.relabel_percent = relabel_percent
        if self.replay_strategy == 'future' or self.replay_strategy == 'clearning':
            # self.future_p = 1 - (1. / (1 + replay_k))
            self.future_p = relabel_percent
        else:
            self.future_p = 0
        self.reward_func = reward_func

    def sample_her_transitions(self, episode_batch, batch_size_in_transitions, future_p=0.):
        if future_p is None:
            future_p = self.future_p
        T = episode_batch['actions'].shape[1]
        rollout_batch_size = episode_batch['actions'].shape[0]
        batch_size = batch_size_in_transitions
        # select which rollouts and which timesteps to be used
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = np.random.randint(T, size=batch_size)

        transitions = {} 
        for key in episode_batch.keys():
            if key != 'initial_obs':
                transitions[key] = episode_batch[key][episode_idxs, t_samples].copy()
            else:
                transitions[key] = episode_batch[key][episode_idxs, 0].copy()
        # her idx
        if self.replay_strategy == 'future':
            her_indexes = np.where(np.random.uniform(size=batch_size) < future_p)
            future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
            future_offset = future_offset.astype(int)
            future_t = (t_samples + 1 + future_offset)[her_indexes]
            # replace go with achieved goal
            future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
            transitions['g'][her_indexes] = future_ag

        elif self.replay_strategy == 'clearning':
            
            # label the first half batch with next goals
            relabel_next_num = int(future_p * batch_size)
            her_indexes = np.arange(relabel_next_num)
            future_offset = np.random.uniform(size=batch_size) * 0. # do next-sampling
            future_offset = future_offset.astype(int)
            future_t = (t_samples + 1 + future_offset)[her_indexes]
            future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
            transitions['g'][her_indexes] = future_ag

            # label the next half batch with random goals
            orig_goals = transitions['g'][her_indexes].copy()
            np.random.shuffle(orig_goals)
            random_goals = orig_goals 
            transitions['g'][relabel_next_num:] = random_goals

        # to get the params to re-compute reward
        transitions['r'] = np.expand_dims(self.reward_func(transitions['ag_next'], transitions['g'], None), 1)
        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:]) for k in transitions.keys()}

        return {'transitions': transitions, 'future_offset':future_offset}
        # return transitions
