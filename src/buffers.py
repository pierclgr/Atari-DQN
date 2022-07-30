import torch
from src.exceptions import InvalidTimestepException, EmptyReplayBufferException
import random


class ReplayBuffer(object):
    def __init__(self) -> None:
        # first, let's create an empty buffer with the given capacity; we'll implement the buffer as a PyTorch tensor
        # in order to support training with PyTorch
        # the buffer will be a 2D PyTorch tensor with 0 rows and 4 columns, where each row is an element
        # of the timeseries of the moves and the columns are the state at that time, the action taken, the state reached
        # after taking the action and the obtained reward
        self.buffer_content = self.__initialize_empty_buffer()
        self.buffer_size = 0

    def __len__(self) -> int:
        return self.buffer_size

    def __repr__(self) -> str:
        return f"ReplayBuffer(\n\tbuffer_content:\n\t\t{self.buffer_content}\n\tbuffer_size: {self.buffer_size})"

    @staticmethod
    def __initialize_empty_buffer() -> torch.Tensor:
        # initialize an empty buffer tensor
        return torch.empty(size=(0, 4))

    def append(self, state, action, next_state, reward) -> None:
        # let's first create the timestep to add as a tuple with the state at that time, the action taken, the state
        # reached after taking the action and the obtained reward
        timestep = (state, action, next_state, reward)

        # let's convert it to tensor
        timestep = torch.tensor(timestep)

        # now we add a dummy dimension at the beginning of the tensor to make it match with the dimension of the
        # buffer tensor
        timestep = timestep[None, :]

        # we now append the timestep tensor to the buffer tensor to save it in the replay buffer
        self.buffer_content = torch.cat((self.buffer_content, timestep), dim=0)

        # increment buffer size by 1
        self.buffer_size += 1

        # delete the timestep tensor in order to free memory
        del timestep

    def sample(self) -> tuple:
        if self.buffer_size is 0:
            raise EmptyReplayBufferException("The replay buffer is empty.")
        else:
            t = random.randint(0, self.buffer_size - 1)
            return t, self.get_timestep(t)

    def get_timestep(self, t: int = 0) -> torch.Tensor:
        if t < self.buffer_size:
            return self.buffer_content[t]
        else:
            raise InvalidTimestepException(
                f"Timestep {t} does does not exist in this buffer, last timestep is {self.buffer_size - 1}.")

    def reset(self) -> None:
        # if the buffer tensor is already initialized, namely is not None
        if self.buffer_content is not None:
            # we delete the tensor
            del self.buffer_content

        # initialize an empty buffer tensor
        self.buffer_content = self.__initialize_empty_buffer()
