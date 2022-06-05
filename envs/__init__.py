import gym
from gym.envs.registration import register

def register_envs():
    register(
        id='DClawTurn-v0',
        entry_point='envs.claw_env:ClawEnv',
        max_episode_steps=80)