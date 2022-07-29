import random
import numpy as np


def set_seeds(seed: int = 1507):
    """
    Method to set the seeds of random operations

    :param seed: the seed to use (int, default 1507)
    :return: None
    """

    # set random seed
    random.seed(seed)

    # set numpy random seed
    np.random.seed(seed)
