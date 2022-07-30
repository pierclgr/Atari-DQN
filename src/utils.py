import random
import numpy as np
import torch
import os


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


def get_device() -> str:
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

    print(f">>> Using {device} device")

    return device