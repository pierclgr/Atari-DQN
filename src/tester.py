import importlib
import os

import hydra
from omegaconf import DictConfig
import sys
from colabgymrender.recorder import Recorder
from pyvirtualdisplay import Display

import gym

from src.agents import DQNAgent
from src.utils import get_device, set_seeds


@hydra.main(version_base=None, config_path="../config/", config_name="breakout")
def tester(config: DictConfig):
    # check if we're running in colab
    in_colab = config.in_colab

    # get the device
    device = get_device()

    # create the environment
    if in_colab:
        env = gym.make(config.env_name, obs_type="rgb")
    else:
        env = gym.make(config.env_name, render_mode="human", obs_type="rgb")

    # set seeds for reproducibility
    # set_seeds(env)

    # apply Atari preprocessing
    env = gym.wrappers.AtariPreprocessing(env,
                                          noop_max=config.preprocessing.n_frames_per_state,
                                          frame_skip=config.preprocessing.n_frames_to_skip,
                                          screen_size=config.preprocessing.patch_size,
                                          grayscale_obs=config.preprocessing.grayscale)
    env = gym.wrappers.FrameStack(env, num_stack=config.preprocessing.n_frames_per_state)

    # import specified model
    model = getattr(importlib.import_module("src.models"), config.model)

    # if running in colab
    if in_colab:
        # Set up display for visualization
        Display(visible=False, size=(400, 300)).start()

        # Instantiate the recorder wrapper around gym's environment to record and
        # visualize the environment
        os.makedirs(f'{config.home_directory}video/test')
        env = Recorder(env, directory=f'{config.home_directory}video/test')

    # initialize the agent
    agent = DQNAgent(env=env, device=device, q_function=model, buffer_capacity=config.buffer_capacity,
                     num_episodes=config.num_episodes, batch_size=config.batch_size, discount_rate=config.gamma,
                     target_update_steps=config.c, logger=None, eps_max=config.eps_max, eps_min=config.eps_min,
                     eps_decay_steps=config.eps_decay_steps, checkpoint_every=config.checkpoint_every,
                     num_testing_episodes=config.num_testing_episodes, home_directory=config.home_directory)

    # load the trained model
    agent.load(config.output_model_file)

    # test the agent
    agent.test()

    # if in colab
    if in_colab:
        # play the game video
        env.play()

    # close the environment
    env.close()


if __name__ == "__main__":
    tester()
