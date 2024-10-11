import numpy as np
import gymnasium as gym


class RandomAgent:
    """
    This agent selects actions randomly.
    """
    def __init__(self, env: gym.Env):
        self.env = env

    def act(self, state: np.array, epsilon: float = None) -> int:
        """
        The agent does not consider the state of the environment when deciding
        what to do next.
        """
        return self.env.action_space.sample()
