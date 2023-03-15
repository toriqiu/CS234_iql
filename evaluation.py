from typing import Dict

import flax.linen as nn
import gym
import numpy as np


def evaluate(agent: nn.Module, env: gym.Env,
             num_episodes: int) -> Dict[str, float]:
    stats = {'return': [], 'length': []}

    for _ in range(num_episodes):
        env.observation_space = gym.spaces.Box(low=float('-inf'), high=float('inf'), shape=(6, 29))
        env.action_space = gym.spaces.Box(low=-1, high=1, shape=(6, 8))
        observation, done = env.reset(), False
        # (29) -> # (1, 6, 29)
        padding = np.zeros((1, 6, 29))
        padding[0,5,:] = observation
        observation = padding
        # print(observation.shape)
        while not done:
            action = agent.sample_actions(observation, temperature=0.0)
            observation, _, done, info = env.step(action[0,-1,:])

            for i in range(5):
              padding[0,i,:] = padding[0,i+1,:]
            padding[0,5,:] = observation
            observation = padding


        for k in stats.keys():
            stats[k].append(info['episode'][k])

    for k, v in stats.items():
        stats[k] = np.mean(v)

    return stats
