from abc import ABC, abstractmethod
import random
import gym


# TODO documentation

# define a superclass "Agent" for all the possible Agents; this will act as an interface for the other classes
class Agent(ABC):
    """
    A class representing an agent interface for other agents
    """

    @abstractmethod
    def __init__(self, env: gym.Env):
        """
        Constructor method of the class that initializes the agent with the environment characteristics

        :param env: the gym environment that the agent should interact with (gym.Env)
        """

        # set action space size
        self.action_space_size = env.action_space.n

    @abstractmethod
    def select_action(self, *args) -> int:
        """
        Method to select an action from the action space accordingly to a certain policy

        :param args: arguments passed to the agent to make the selection
        :return: the action selected by the environment (int)
        """
        pass


# define a class "RandomAgent" which is an agent following the random action selection policy
class RandomAgent(Agent):
    """
    A class representing an agent that follows random action selection policy
    """

    def __init__(self, env):
        """
        Constructor method for the class RandomAgent that just calls the constructor of its superclass Agent

        :param env: the gym environment that the agent should interact with (gym.Env)
        """
        super(RandomAgent, self).__init__(env)

    def select_action(self) -> int:
        """
        Method that selects an action to take from the action space randomly

        :return: action from the action space selected randomly (int)
        """

        # select an action randomly from the action space by using the action space size
        selected_action = random.randint(0, self.action_space_size - 1)
        return selected_action


class CartPoleAgent(Agent):
    def __init__(self, env):
        super(CartPoleAgent, self).__init__(env)

    def select_action(self, state) -> int:
        # select an action randomly from the action space by using the action space size
        pole_angle = state[2]
        if pole_angle < 0:
            selected_action = 0
        else:
            selected_action = 1

        return selected_action
