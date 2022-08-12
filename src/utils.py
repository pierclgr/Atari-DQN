import random
from typing import Any

import gym
import numpy as np
import torch
import os

import random
import string


def random_string(chars=string.ascii_letters + string.digits, num_char=5):
    return ''.join(random.choice(chars) for _ in range(num_char))


class StateTransition(object):
    def __init__(self, state: Any, action: Any, reward: Any, next_state: Any, done: Any):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done

    def __repr__(self) -> str:
        return f"StateTransition(state: {self.state}, action: {self.action}, reward: {self.reward}, " \
               f"next_state: {self.next_state}, done: {self.done})"


def set_seeds(seed: int = 1507) -> None:
    """
    Method to set the seeds of random components to allow reproducibility

    :param seed: the seed to use (int, default 1507)
    :return: None
    """

    # set random seed
    random.seed(seed)

    # set numpy random seed
    np.random.seed(seed)

    # set pytorch random seed
    torch.manual_seed(seed)


def get_device() -> torch.device:
    """
    Get the current machine device to use

    :return: device to use for training (str)
    """
    # import torch_xla library if runtime is using a Colab TPU
    if 'COLAB_TPU_ADDR' in os.environ:
        import torch_xla.core.xla_model as xm

    # if the current runtime is using a Colab TPU, define a flag specifying that TPU will be used
    if 'COLAB_TPU_ADDR' in os.environ:
        use_tpu = True
    else:
        use_tpu = False

    # if TPU is available, use it as device
    if use_tpu:
        device = xm.xla_device()
    else:
        # otherwise use CUDA device or CPU accordingly to the one available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # if the device is a GPU
        if torch.cuda.is_available():
            # print the details of the given GPU
            stream = os.popen('nvidia-smi')
            output = stream.read()
            print(output)

    return device


def checkpoint_episode_trigger(episode_id: int, checkpoint_every: int):
    return episode_id % checkpoint_every == 0
