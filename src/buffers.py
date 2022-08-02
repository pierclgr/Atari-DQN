from collections import deque
from typing import Tuple, List

from src.exceptions import EmptyReplayBufferException
import random
import numpy as np
from utils import StateTransition


# TODO DOCUMENTATION

class ReplayBuffer(object):
    def __init__(self, capacity: int = 10000) -> None:
        # first, let's create an empty buffer with the given capacity; this will be an empty list containing tuples of
        # 4 elements:
        # - state: the state of the environment at that time
        # - action: the action taken from the agent at that time
        # - reward: the reward obtained by the agent from the environment after taking the action
        # - next_state: the state reached after taking the action
        self.capacity = capacity
        self.__content = deque(maxlen=self.capacity)

    def __len__(self) -> int:
        return len(self.__content)

    def __repr__(self) -> str:
        return f"ReplayBuffer(" \
               f"\n\tcontent:\n\t\t{self.__content}" \
               f"\n\tsize: {len(self.__content)}" \
               f"\n\tcapacity: {self.capacity})"

    def store(self, state_transition: StateTransition) -> None:
        self.__content.append(state_transition)

    def sample(self, num_samples: int = 1) -> List[StateTransition]:
        if len(self.__content) is 0:
            raise EmptyReplayBufferException("The replay buffer is empty.")
        else:
            return random.sample(self.__content, k=num_samples)

    def reset(self) -> None:
        # clear the buffer list
        self.__content.clear()
