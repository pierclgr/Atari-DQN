import random
from typing import Any, Tuple, Optional

import gym
import numpy as np
import torch
import os
from src.wrappers import ReproducibleEnv

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


def set_reproducibility(training_env: gym.Env, testing_env: gym.Env, train_seed: int = 1507,
                        test_seed: int = 2307) -> gym.Env:
    """
    Method to set the seeds of random components to allow reproducibility

    :param seed: the seed to use (int, default 1507)
    :return: None
    """

    # set random seed
    random.seed(train_seed)

    # set numpy random seed
    np.random.seed(train_seed)

    # set pytorch random seed
    torch.manual_seed(train_seed)
    torch.cuda.manual_seed(train_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # set seeds for gym environments
    training_env.reset(seed=train_seed)
    training_env.action_space.seed(train_seed)

    # set reproducibility for the testing environment
    testing_env = ReproducibleEnv(testing_env, seed=test_seed)
    return testing_env


def get_device() -> Tuple[torch.device, Optional[None]]:
    """
    Get the current machine device to use

    :return: device to use for training (str)
    """
    gpu_info = None

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
            gpu_info = stream.read()

    return device, gpu_info


def checkpoint_episode_trigger(episode_id: int, save_video_every: int):
    return episode_id % save_video_every == 0


def manual_record_trigger(step_id: int):
    return step_id < 0
