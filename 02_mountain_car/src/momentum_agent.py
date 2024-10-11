import gymnasium as gym
from src.base_agent import BaseAgent

class MomentumAgent(BaseAgent):

    def __init__(self, env: gym.Env):
        self.env = env

        self.valley_position = -0.5

    def get_action(self, state, epsilon=None) -> int:
        """
        The agent considers the velocity of the car to decide the action.
        """
        velocity = state[1]

        if velocity > 0:
            # accelerate to the right
            action = 2
        else:
            # accelerate to the left
            action = 0

        return action

    def update_parameters(self, state, action, reward, next_state, terminated, truncated, epsilon):
        pass

