import os
import pprint

from src.agents import DQNAgent
import importlib
import gym
from logger import WandbLogger
from src.utils import set_reproducibility, get_device, checkpoint_episode_trigger
from functools import partial
import sys
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from src.atari_wrappers import deepmind_atari_wrappers
from gym.wrappers import TimeLimit


@hydra.main(version_base=None, config_path="../config/", config_name="breakout")
def trainer(config: DictConfig) -> None:
    in_colab = config.in_colab

    # get the device
    device = get_device()

    # create the training environment
    train_env = gym.make(config.env_name, obs_type="rgb")

    # create the testing environment
    render_mode = "rgb_array" if in_colab else "human"
    test_env = gym.make(config.env_name, obs_type="rgb", render_mode=render_mode)

    # set seeds for reproducibility
    set_reproducibility(training_env=train_env, testing_env=test_env, train_seed=config.reproducibility.train_seed,
                        test_seed=config.reproducibility.test_seed)

    configuration = OmegaConf.to_object(config)

    # initialize logger
    if config.logging:
        logger = WandbLogger(name=f"{config.game}_DQN", config=configuration)
    else:
        logger = None

    print(f"Using {device} device...")
    print("Training configuration:")
    pprint.pprint(configuration)

    # apply Atari preprocessing
    train_env = deepmind_atari_wrappers(train_env, max_episode_steps=config.max_steps_per_episode,
                                        noop_max=config.preprocessing.noop_max,
                                        frame_skip=config.preprocessing.n_frames_to_skip,
                                        episode_life=config.preprocessing.episode_life,
                                        clip_rewards=config.preprocessing.clip_rewards,
                                        frame_stack=config.preprocessing.n_frames_per_state,
                                        scale=config.preprocessing.scale_obs,
                                        patch_size=config.preprocessing.patch_size,
                                        grayscale=config.preprocessing.grayscale,
                                        fire_reset=config.preprocessing.fire_reset)
    # apply Atari preprocessing
    test_env = deepmind_atari_wrappers(test_env, max_episode_steps=config.max_steps_per_episode,
                                       noop_max=config.preprocessing.noop_max,
                                       frame_skip=config.preprocessing.n_frames_to_skip,
                                       episode_life=config.preprocessing.episode_life,
                                       clip_rewards=config.preprocessing.clip_rewards,
                                       frame_stack=config.preprocessing.n_frames_per_state,
                                       scale=config.preprocessing.scale_obs,
                                       patch_size=config.preprocessing.patch_size,
                                       grayscale=config.preprocessing.grayscale,
                                       fire_reset=config.preprocessing.fire_reset)

    # Instantiate the recorder wrapper around gym's environment to record and
    # visualize the environment
    episode_trigger = partial(checkpoint_episode_trigger, checkpoint_every=config.save_video_every)
    test_env = gym.wrappers.RecordVideo(test_env, video_folder=f'{config.home_directory}videos',
                                        name_prefix=config.video_file, episode_trigger=episode_trigger)
    test_env.episode_id = 1

    # import specified model
    model = getattr(importlib.import_module("src.models"), config.model)

    # initialize the agent
    agent = DQNAgent(env=train_env, testing_env=test_env, device=device, q_function=model,
                     buffer_capacity=config.buffer_capacity, checkpoint_file=config.checkpoint_file,
                     num_episodes=config.num_episodes, batch_size=config.batch_size, discount_rate=config.gamma,
                     target_update_steps=config.c, logger=logger, eps_max=config.eps_max, eps_min=config.eps_min,
                     eps_decay_steps=config.eps_decay_steps, checkpoint_every=config.checkpoint_every,
                     home_directory=config.home_directory, learning_rate=config.optimizer.lr,
                     num_initial_replay_samples=config.num_initial_replay_samples,
                     gradient_momentum=config.optimizer.momentum, gradient_alpha=config.optimizer.squared_momentum,
                     gradient_eps=config.optimizer.min_squared_gradient)

    # train the environment
    agent.train()

    # save the trained model
    agent.save(filename=config.output_model_file)

    # close the environment
    train_env.close()
    test_env.close()

    # close the logger
    if config.logging:
        logger.finish()


if __name__ == "__main__":
    trainer()
