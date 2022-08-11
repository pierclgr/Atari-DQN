import os
import pprint

from colabgymrender.recorder import Recorder
from pyvirtualdisplay import Display

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
    # check if we're running in colab
    in_colab = config.in_colab

    # get the device
    device = get_device()

    # set seeds for reproducibility
    set_seeds()

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
        "num_testing_episodes": config.num_testing_episodes,
        "home_directory": config.home_directory
    }

    print("Training configuration:")
    pprint.pprint(configuration)

    # initialize logger
    if config.logging:
        logger = WandbLogger(name="DQN", config=configuration)
    else:
        logger = None

    # create the environment
    if in_colab:
        env = gym.make(config.env_name, obs_type="rgb")
    else:
        env = gym.make(config.env_name, render_mode="human", obs_type="rgb")

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
        # visualize the environment+
        os.makedirs(f'{config.home_directory}video/train')
        env = Recorder(env, directory=f'{config.home_directory}video/train')

    # initialize the agent
    agent = DQNAgent(env=env, device=device, q_function=model, buffer_capacity=config.buffer_capacity,
                     num_episodes=config.num_episodes, batch_size=config.batch_size, discount_rate=config.gamma,
                     target_update_steps=config.c, logger=logger, eps_max=config.eps_max, eps_min=config.eps_min,
                     eps_decay_steps=config.eps_decay_steps, checkpoint_every=config.checkpoint_every,
                     num_testing_episodes=config.num_testing_episodes, home_directory=config.home_directory)

    # train the environment
    agent.train()

    # save the trained model
    agent.save(filename=config.output_model_file)

    # if in colab
    if in_colab:
        # play the game video
        env.play()

    # close the environment
    env.close()

    # close the logger
    if config.logging:
        logger.finish()


if __name__ == "__main__":
    trainer()
