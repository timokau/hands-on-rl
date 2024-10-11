from typing import Optional, Tuple, List
from tqdm import tqdm

import torch


def evaluate(
    agent,
    env,
    n_episodes: int,
    seed: Optional[int] = 0,
) -> Tuple[List, List]:

    # from src.utils import set_seed
    # set_seed(env, seed)

    # output metrics
    reward_per_episode = []
    steps_per_episode = []

    for i in tqdm(range(0, n_episodes)):

        state, _ = env.reset()
        rewards = 0
        steps = 0
        terminated = False
        truncated = False
        while not (terminated or truncated):

            action = agent.act(torch.as_tensor(state, dtype=torch.float32))

            next_state, reward, terminated, truncated, info = env.step(action)

            rewards += reward
            steps += 1
            state = next_state

        reward_per_episode.append(rewards)
        steps_per_episode.append(steps)

    return reward_per_episode, steps_per_episode
