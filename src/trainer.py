import pprint
import importlib
import gym
from logger import WandbLogger
from src.utils import set_reproducibility, get_device, video_episode_trigger
from functools import partial
import hydra
from omegaconf import DictConfig, OmegaConf
from src.wrappers import vector_atari_deepmind_env, atari_deepmind_env


@hydra.main(version_base=None, config_path="../config/", config_name="breakout_train_dqn")
def trainer(config: DictConfig) -> None:
    """
    Function to train an agent with the parameters specified in the given configuration

    :param config: hydra configuration for the training (DictConfig)

    :return: None
    """

    # load the given configuration
    configuration = OmegaConf.to_object(config)

    # initialize logger if required
    if config.logging:
        logger = WandbLogger(name=config.wandb.run_name, config=configuration, project=config.wandb.project_name,
                             entity=config.wandb.entity_name)
    else:
        logger = None

    # load the save space boolean from the configuration
    save_space = config.save_space

    # get the device and print the gpu info if gpu is available
    device, gpu_info = get_device()
    if gpu_info:
        print(gpu_info)

    # create the training environment as a vectorized and wrapped atari environment with the parameters specified in the
    # configuration file
    train_env = vector_atari_deepmind_env(env_name=config.env_name,
                                          num_envs=config.num_parallel_envs,
                                          max_episode_steps=config.max_steps_per_episode,
                                          noop_max=config.preprocessing.noop_max,
                                          frame_skip=config.preprocessing.n_frames_to_skip,
                                          episode_life=config.preprocessing.episode_life,
                                          clip_rewards=config.preprocessing.clip_rewards,
                                          frame_stack=config.preprocessing.n_frames_per_state,
                                          scale=config.preprocessing.scale_obs,
                                          patch_size=config.preprocessing.patch_size,
                                          grayscale=config.preprocessing.grayscale,
                                          fire_reset=config.preprocessing.fire_reset,
                                          render_mode=None)

    # create the testing environment as a wrapped atari environment with the parameters specified in the configuration
    # file
    test_env = atari_deepmind_env(env_name=config.env_name,
                                  render_mode="rgb_array",
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

    # set seeds for reproducibility
    test_env = set_reproducibility(training_env=train_env, testing_env=test_env,
                                   train_seed=config.reproducibility.train_seed,
                                   test_seed=config.reproducibility.test_seed)

    # print device and training configuration
    print(f"Using {device} device...")
    print("Training configuration:")
    pprint.pprint(configuration)

    # Instantiate the recorder wrapper around test environment to record and visualize the environment learning progress
    episode_trigger = partial(video_episode_trigger,
                              save_video_every=config.test_video.save_every_n_test_episodes)
    test_env = gym.wrappers.RecordVideo(test_env,
                                        video_folder=f'{config.home_directory}{config.test_video.output_folder}',
                                        name_prefix=f"{config.test_video.file_name}", episode_trigger=episode_trigger)

    # import the model specified in the configuration to use for the training
    model = getattr(importlib.import_module("src.models"), config.model)

    # import the agent to train specified in the configuration
    agent = getattr(importlib.import_module("src.agents"), config.agent)

    # initialize the specified agent with the two train and test environments and the parameters defined in the
    # configuration file
    agent = agent(env=train_env, testing_env=test_env, device=device, q_function=model,
                  buffer_capacity=config.buffer_capacity, checkpoint_file=config.checkpoint_file,
                  num_training_steps=config.num_training_steps, batch_size=config.batch_size,
                  target_update_steps=config.c, logger=logger, eps_max=config.eps_max, eps_min=config.eps_min,
                  eps_decay_steps=config.eps_decay_steps, checkpoint_every=config.checkpoint_every_n_gradient_steps,
                  home_directory=config.home_directory, learning_rate=config.optimizer.lr,
                  num_initial_replay_samples=config.num_initial_replay_samples, discount_rate=config.gamma,
                  gradient_momentum=config.optimizer.momentum, gradient_alpha=config.optimizer.squared_momentum,
                  gradient_eps=config.optimizer.min_squared_gradient, save_space=save_space,
                  buffered_avg_reward_size=config.buffered_avg_reward_size,
                  test_every=config.test_every_n_gradient_steps)

    # train the environment
    agent.train()

    # save the final trained model
    agent.save(filename=config.output_model_file)

    # close the training and testing environments
    train_env.close()
    test_env.close()

    # close and finish the logger
    if config.logging:
        logger.finish()


if __name__ == "__main__":
    trainer()
