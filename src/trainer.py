import os
import pprint

from gym.wrappers import RecordVideo

from src.agents import DQNAgent
import importlib
import gym
from logger import WandbLogger
from src.utils import set_seeds, get_device
import sys
import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../config/", config_name="breakout")
def trainer(config: DictConfig) -> None:
    # get the device
    device = get_device()

    # set seeds for reproducibility
    set_seeds(seed=config.train_seed)

    # set training hyperparameters
    configuration = {
        "batch_size": config.batch_size,
        "num_episodes": config.num_episodes,
        "buffer_capacity": config.buffer_capacity,
        "n_frames_to_skip": config.preprocessing.n_frames_to_skip,
        "n_frames_per_state": config.preprocessing.n_frames_per_state,
        "patch_size": config.preprocessing.patch_size,
        "grayscale": config.preprocessing.grayscale,
        "discount_factor": config.gamma,
        "target_update_steps": config.c,
        "learning_rate": config.lr,
        "eps_max": config.eps_max,
        "eps_min": config.eps_min,
        "eps_decay_steps": config.eps_decay_steps,
        "checkpoint_every": config.checkpoint_every,
        "env_name": config.env_name,
        "model": config.model,
        "logging": config.logging,
        "home_directory": config.home_directory,
        "train_seed": config.train_seed,
        "test_seed": config.test_seed,
        "max_steps_per_episode": config.max_steps_per_episode
    }

    print("Training configuration:")
    pprint.pprint(configuration)

    # initialize logger
    if config.logging:
        logger = WandbLogger(name="DQN", config=configuration)
    else:
        logger = None

    # create the training environment
    train_env = gym.make(config.env_name, obs_type="rgb")

    # apply Atari preprocessing
    train_env = gym.wrappers.AtariPreprocessing(train_env,
                                                noop_max=config.preprocessing.n_frames_per_state,
                                                frame_skip=config.preprocessing.n_frames_to_skip,
                                                screen_size=config.preprocessing.patch_size,
                                                grayscale_obs=config.preprocessing.grayscale)
    train_env = gym.wrappers.FrameStack(train_env, num_stack=config.preprocessing.n_frames_per_state)

    # create the testing environment
    test_env = gym.make(config.env_name, obs_type="rgb", render_fps=60)

    # apply Atari preprocessing
    test_env = gym.wrappers.AtariPreprocessing(test_env,
                                               noop_max=config.preprocessing.n_frames_per_state,
                                               frame_skip=config.preprocessing.n_frames_to_skip,
                                               screen_size=config.preprocessing.patch_size,
                                               grayscale_obs=config.preprocessing.grayscale)
    test_env = gym.wrappers.FrameStack(test_env, num_stack=config.preprocessing.n_frames_per_state)

    # Instantiate the recorder wrapper around gym's environment to record and
    # visualize the environment
    test_env = RecordVideo(test_env, video_folder=f'{config.home_directory}video')

    # import specified model
    model = getattr(importlib.import_module("src.models"), config.model)

    # initialize the agent
    agent = DQNAgent(env=train_env, testing_env=test_env, device=device, q_function=model,
                     buffer_capacity=config.buffer_capacity, checkpoint_file=config.checkpoint_file,
                     num_episodes=config.num_episodes, batch_size=config.batch_size, discount_rate=config.gamma,
                     target_update_steps=config.c, logger=logger, eps_max=config.eps_max, eps_min=config.eps_min,
                     eps_decay_steps=config.eps_decay_steps, checkpoint_every=config.checkpoint_every,
                     home_directory=config.home_directory, seed=config.train_seed, testing_seed=config.test_seed,
                     max_steps_per_episode=config.max_steps_per_episode, learning_rate=config.lr)

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
