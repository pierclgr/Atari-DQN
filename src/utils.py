from typing import Any, Tuple, Optional
import gym
import numpy as np
import torch
import os
from src.wrappers import ReproducibleEnv
import random
import string


def random_string(chars: str = string.ascii_letters + string.digits, num_char: int = 5) -> str:
    """
    Function to create a random string using the given characters

    :param chars: string containing all the possible characters to use for the string generation, default is
        string.ascii_letters plus string.digits (str)
    :param num_char: the length of the generated string, default is 5 (int)

    :return: a randomly generated string with the given length (str)
    """

    return ''.join(random.choice(chars) for _ in range(num_char))


class StateTransition(object):
    """
    Class that describes a state transition from one state to another
    """

    def __init__(self, state: Any, action: Any, reward: Any, next_state: Any, done: Any) -> None:
        """
        Constructor method of the class

        :param state: the starting state of the transition (Any)
        :param action: the action applied to the starting state (Any)
        :param reward: the reward obtained after applying the action (Any)
        :param next_state: the state reached after applying the action (Any)
        :param done: indicates if the state reached after applying the action is final or not (Any)
        """

        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done

    def __repr__(self) -> str:
        """
        Method that creates a string representation for an object of this class

        :return: the string representation of the object (str)
        """

        return f"StateTransition(state: {self.state}, action: {self.action}, reward: {self.reward}, " \
               f"next_state: {self.next_state}, done: {self.done})"


def set_reproducibility(training_env: gym.Env,
                        testing_env: gym.Env,
                        train_seed: int = 1507,
                        test_seed: int = 2307) -> gym.Env:
    """
    Method to set the seeds of random components to allow reproducibility

    :param training_env: the training environment to set the seed to (gym.Env)
    :param testing_env: the testing environment to set the seed to (gym.Env)
    :param train_seed: the seed to use for random actions and for the train environment, default is 1507 (int)
    :param test_seed: the seed to use for the test environment, default is 2307 (int)

    :return: the testing environment wrapped with the reproducibility wrapper initialized with the given seed (gym.Env)
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

    # set seeds for the training environment once
    training_env.action_space.seed(train_seed)
    training_env.reset(seed=train_seed)

    # set seed for the testing environment such that everytime it is reset, it gets initialized with the same seed
    testing_env = ReproducibleEnv(testing_env, seed=test_seed)

    return testing_env


def get_device() -> Tuple[torch.device, Optional[None]]:
    """
    Get the current machine device to use

    :return: tuple containing the device to use and an optional value that is the gpu info if any gpu is available
        (tuple)
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


def video_episode_trigger(episode_id: int, save_video_every: int = 5) -> bool:
    """
    Function that is used to trigger the recorder every N testing episodes

    :param episode_id: the id of the current episode (int)
    :param save_video_every: the number of testing episodes after which to trigger video recording and saving, default
        is 5 (int)

    :return: boolean that is true if the current episode must be recorded, false otherwise (bool)
    """

    return episode_id % save_video_every == 0
