from collections import deque
from typing import List

from src.exceptions import EmptyReplayBufferException
import random
from utils import StateTransition


class ReplayBuffer(object):
    """
    Class to describe an experience replay buffer
    """

    def __init__(self, capacity: int = 10000) -> None:
        """
        Method to initialize an empty experience replay buffer with a maximum capacity as a deque

        :param capacity: the maximum capacity of the buffer, default value is 10K (int)

        :return: None
        """

        self.capacity = capacity
        self.__content = deque(maxlen=self.capacity)

    def __len__(self) -> int:
        """
        Method that returns the number of samples currently in the buffer

        :return: number of samples in the replay buffer, namely the length of the deque, not the capacity (int)
        """

        return len(self.__content)

    def __repr__(self) -> str:
        """
        Method that returns a string representation of the replay buffer

        :return: string representation of the replay buffer state (str)
        """

        return f"ReplayBuffer(" \
               f"\n\tcontent:\n\t\t{self.__content}" \
               f"\n\tsize: {len(self.__content)}" \
               f"\n\tcapacity: {self.capacity})"

    def store(self, state_transition: StateTransition) -> None:
        """
        Method that stores a STateTransition object in the replay buffer

        :param state_transition: the state transition object to store in the buffer (StateTransition)

        :return: None
        """

        # push the input state transition in the deque
        self.__content.append(state_transition)

    def sample(self, num_samples: int = 1) -> List[StateTransition]:
        """
        Method that samples a defined number of samples from the replay buffer

        :param num_samples: the number of samples to sample from the buffer, default value is 1 (int)

        :return: a list containing the sampled samples (List[StateTransition])
        """

        # if the buffer is empty
        if len(self.__content) is 0:
            # raise an Empty buffer exception
            raise EmptyReplayBufferException("The replay buffer is empty.")
        else:
            # sample num_samples samples and return them
            return random.sample(self.__content, k=num_samples)

    def reset(self) -> None:
        """
        Method to clear the replay buffer

        :return: None
        """

        # clear the buffer deque
        self.__content.clear()
