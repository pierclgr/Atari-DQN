import importlib
import math
import os
from functools import partial

import hydra
from omegaconf import DictConfig, OmegaConf

import gym

from src.agents import DQNAgent
from src.logger import WandbLogger
from src.utils import get_device
from src.wrappers import atari_deepmind_env


@hydra.main(version_base=None, config_path="../config/", config_name="test")
def tester(config: DictConfig):
    configuration = OmegaConf.to_object(config)

    # initialize logger
    if config.logging:
        logger = WandbLogger(name=f"{config.wandb_run_name}", config=configuration)
    else:
        logger = None

    in_colab = config.in_colab

    # get the device
    device, gpu_info = get_device()
    if gpu_info:
        print(gpu_info)

    render_mode = "rgb_array" if in_colab else "human"

    # apply Atari preprocessing
    test_env = atari_deepmind_env(env_name=config.env_name,
                                  render_mode=render_mode,
                                  max_episode_steps=config.max_steps_per_episode,
                                  noop_max=config.preprocessing.noop_max,
                                  frame_skip=config.preprocessing.n_frames_to_skip,
                                  episode_life=config.preprocessing.episode_life,
                                  clip_rewards=config.preprocessing.clip_rewards,
                                  frame_stack=config.preprocessing.n_frames_per_state,
                                  scale=config.preprocessing.scale_obs,
                                  patch_size=config.preprocessing.patch_size,
                                  grayscale=config.preprocessing.grayscale,
                                  fire_reset=config.preprocessing.fire_reset)

    # Instantiate the recorder wrapper around test environment to record and
    # visualize the environment learning progress
    if in_colab:
        test_env = gym.wrappers.RecordVideo(test_env,
                                            video_folder=f'{config.home_directory}{config.test_video.output_folder}',
                                            name_prefix=f"{config.test_video.file_name}",
                                            video_length=math.inf)

    # import specified model
    model = getattr(importlib.import_module("src.models"), config.model)

    # import specified agent
    agent = getattr(importlib.import_module("src.agents"), config.agent)

    # initialize the agent
    agent = agent(env=test_env, testing_env=test_env, device=device, q_function=model, logger=logger,
                  home_directory=config.home_directory, buffered_avg_reward_size=config.buffered_avg_reward_size,
                  checkpoint_file="")

    # load the defined trained model
    agent.load(filename=config.output_model_file)

    # play
    agent.play()

    # close the environment
    test_env.close()

    # close the logger
    if config.logging:
        logger.finish()


if __name__ == "__main__":
    tester()
