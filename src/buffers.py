from typing import Tuple, List

from src.exceptions import InvalidTimestepException, EmptyReplayBufferException
import random
import numpy as np
from utils import StateTransition


# TODO DOCUMENTATION

class ReplayBuffer(object):
    def __init__(self, buffer_capacity: int = 10000) -> None:
        # first, let's create an empty buffer with the given capacity; this will be an empty list containing tuples of
        # 4 elements:
        # - state: the state of the environment at that time
        # - action: the action taken from the agent at that time
        # - reward: the reward obtained by the agent from the environment after taking the action
        # - next_state: the state reached after taking the action
        self.__buffer_content = []
        self.__buffer_capacity = buffer_capacity
        self.__overwrite_position = 0

    def __len__(self) -> int:
        return len(self.__buffer_content)

    def __repr__(self) -> str:
        return f"ReplayBuffer(" \
               f"\n\tbuffer_content:\n\t\t{self.__buffer_content}" \
               f"\n\tbuffer_size: {len(self.__buffer_content)}" \
               f"\n\tbuffer_capacity: {self.__buffer_capacity}" \
               f"\n\toverwrite_position: {self.__overwrite_position})"

    def insert(self, state_transition: StateTransition) -> None:
        # if the buffer is full
        if self.__check_full():
            # store the state transition in the current overwrite position
            self.__buffer_content[self.__overwrite_position] = state_transition

            # increment the overwrite position
            self.__increment_overwrite_position()
        else:
            # otherwise the buffer has still not yet reached the maximum capacity, so just append the state transition
            # to the list
            self.__buffer_content.append(state_transition)

    def sample(self) -> StateTransition:
        if len(self.__buffer_content) is 0:
            raise EmptyReplayBufferException("The replay buffer is empty.")
        else:
            return random.choice(self.__buffer_content)

    def get_transition_at_timestep(self, t: int = 0) -> StateTransition:
        if t < len(self.__buffer_content):
            return self.__buffer_content[t]
        else:
            raise InvalidTimestepException(
                f"Timestep {t} does does not exist in this buffer, last timestep is {len(self.__buffer_content) - 1}.")

    def reset(self) -> None:
        # clear the buffer list
        self.__buffer_content.clear()

    def __check_full(self) -> bool:
        return len(self.__buffer_content) >= self.__buffer_capacity

    def __increment_overwrite_position(self) -> None:
        # if the current overwrite position is the last (the maximum capacity allowed size minus one)
        if self.__overwrite_position is self.__buffer_capacity - 1:
            # set it to 0 to restart the storing from the beginning of the list, thus overwriting earliest elements
            self.__overwrite_position = 0
        else:
            # otherwise just increment it
            self.__overwrite_position += 1

    def get_buffer_content(self) -> List[StateTransition]:
        return self.__buffer_content


if __name__ == "__main__":
    bf = ReplayBuffer()
    bf.insert(StateTransition(np.array([]), 1, 1, np.array([]), True))
    bf.insert(StateTransition(np.array([]), 2, 2, np.array([]), True))
    bf.insert(StateTransition(np.array([]), 3, 3, np.array([]), True))
    bf.insert(StateTransition(np.array([]), 4, 4, np.array([]), True))
    bf.insert(StateTransition(np.array([]), 5, 5, np.array([]), True))
    bf.insert(StateTransition(np.array([]), 6, 6, np.array([]), True))
    bf.insert(StateTransition(np.array([]), 7, 7, np.array([]), True))
    bf.insert(StateTransition(np.array([]), 8, 8, np.array([]), True))
    bf.insert(StateTransition(np.array([]), 9, 9, np.array([]), True))
    bf.insert(StateTransition(np.array([]), 10, 10, np.array([]), True))
    print(bf)
    bf.insert(StateTransition(np.array([]), -1, -1, np.array([]), True))
    print(bf.sample())
