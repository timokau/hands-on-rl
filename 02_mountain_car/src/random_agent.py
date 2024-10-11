import gymnasium as gym
from src.base_agent import BaseAgent

class RandomAgent(BaseAgent):
    """
    This agent selects actions randomly.
    """
    def __init__(self, env: gym.Env):
        self.env = env

    def get_action(self, state, epsilon=None) -> int:
        """
        The agent does not consider the state of the environment when deciding
        what to do next.
        """
        return self.env.action_space.sample()

    def update_parameters(self, state, action, reward, next_state, terminated, truncated, epsilon):
        pass

