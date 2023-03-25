from typing import Dict

import flax.linen as nn
import gym
import numpy as np


def evaluate(agent: nn.Module, env: gym.Env,
             num_episodes: int, nonMarkovK: int) -> Dict[str, float]:
    stats = {'return': [], 'length': []}

    for _ in range(num_episodes):
        # input('next episode...')
        observation, done = env.reset(), False
        observation = observation.reshape(1, 1, 29)
        # print(f'done={done}')
        while not done:
            # print(f'{observation.shape}')
            action = agent.sample_actions(observation, temperature=0.0)
            next_observation, _, done, info = env.step(action[0,-1,:])
            next_observation = next_observation.reshape(1, 1, 29)
            
            observation = np.concatenate((observation, next_observation), axis=1)
            if observation.shape[1] > nonMarkovK: #change k here (2)
              observation = observation[:, -nonMarkovK:, :] #change k here (1)
        # print(f'done={done} {info["episode"]}')
        for k in stats.keys():
            stats[k].append(info['episode'][k])

    for k, v in stats.items():
        stats[k] = np.mean(v)

    return stats
