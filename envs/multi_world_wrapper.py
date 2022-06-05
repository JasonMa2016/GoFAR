import gym
import numpy as np
from gym.core import Wrapper
from gym.spaces import Dict, Box
import copy
from numpy.linalg.linalg import norm

class FetchGoalWrapper(Wrapper):
    def __init__(self, env):
        Wrapper.__init__(self, env)
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
    
    def reset(self):
        return self.env.reset()
    
    def compute_rewards(self, achieved_goal, desired_goal, info=None):
        return self.env.compute_rewards(achieved_goal, desired_goal, info)
    
    def compute_reward(self, achieved_goal, desired_goal, info=None):
        return self.env.compute_reward(achieved_goal, desired_goal, info)

    def step(self, action):
        return self.env.step(action)
    
    def render(self, mode='human'):
        return self.env.render()
    
    def sample_goal(self):
        import pdb;pdb.set_trace
        return self.env.env._sample_goal()

class TimeLimit(gym.Wrapper):
    def __init__(self, env, max_episode_steps=None):
        super().__init__(env)
        if max_episode_steps is None and self.env.spec is not None:
            max_episode_steps = env.spec.max_episode_steps
        if self.env.spec is not None:
            self.env.spec.max_episode_steps = max_episode_steps
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            info["TimeLimit.truncated"] = not done
            done = True
        return observation, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)

class NoisyAction(Wrapper):
    def __init__(self, env, noise_eps=0.1):
        Wrapper.__init__(self, env=env)
        self.noise_eps = noise_eps
    def step(self, action):
        action += self.noise_eps * self.action_space.high * np.random.randn(*action.shape)
        action = np.clip(action, -self.action_space.high, self.action_space.high)
        observation, reward, done, info = self.env.step(action)
        return observation, reward, done, info 