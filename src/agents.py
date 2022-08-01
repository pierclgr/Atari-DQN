from abc import ABC, abstractmethod
import random
from typing import Type

import gym
import numpy as np
import torch
from src.models import RLNetwork


# TODO documentation

# define a superclass "Agent" for all the possible Agents; this will act as an interface for the other classes
class Agent(ABC):
    """
    A class representing a generic agent
    """

    def __init__(self, env: gym.Env, *args) -> None:
        """
        Constructor method of the class that initializes the agent with the environment characteristics

        :param env: the gym environment that the agent should interact with (gym.Env)
        :param args: other arguments passed to the agent
        """

        # set action space size
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.num_actions = env.action_space.n
        self.observation_space_shape = env.observation_space.shape

    @abstractmethod
    def get_action(self, *args) -> int:
        """
        Method to select an action from the action space accordingly to a certain policy

        :param args: arguments passed to the agent to make the selection
        :return: the action selected by the environment (int)
        """
        pass

    @abstractmethod
    def get_values(self, state: torch.Tensor):
        pass


# create an agent that plays by following a DQN learned policy
class DQNRLAgent(Agent):
    """
    A class representing a reinforcement learning agent that follows the policy learned with DQN agent
    """

    def __init__(self, env: gym.Env, model: Type[RLNetwork]) -> None:
        """
        Constructor method for the class that initializes the environment and the model

        :param env: the gym environment that the agent should interact with (gym.Env)
        :param model: the class of the model to instantiate, must be a subclass of RLNetwork (type[RLNetwork])
        """
        super(DQNRLAgent, self).__init__(env)

        # instantiate the value model using the given class and the environment action and observation spaces
        self.value_model = model(self.observation_space_shape, self.num_actions)

        # instantiate the target model using the given class and the environment action and observation spaces
        self.target_model = model(self.observation_space_shape, self.num_actions)

    def get_values(self, state: torch.Tensor) -> torch.Tensor:
        # first, we get the q-values from the network by feeding to it the given state of the environment (the
        # observation)
        q_values = self.value_model(state)

        return q_values

    def get_action(self, *args) -> int:
        pass