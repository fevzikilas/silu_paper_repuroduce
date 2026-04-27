from abc import ABC, abstractmethod

class Environment(ABC):
    @abstractmethod
    def reset(self):
        """
        Reset the environment to initial state.
        Returns:
            state: Initial state representation
        """
        pass

    @abstractmethod
    def step(self, action):
        """
        Execute an action in the environment.
        Args:
            action: Action to take
        Returns:
            next_state: Next state after action
            reward: Reward received
            done: Whether episode is finished
            info: Additional information
        """
        pass