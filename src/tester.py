import importlib
import math
import hydra
from omegaconf import DictConfig, OmegaConf
import gym
from src.logger import WandbLogger
from src.utils import get_device
from src.wrappers import atari_deepmind_env


@hydra.main(version_base=None, config_path="../config/", config_name="test")
def tester(config: DictConfig) -> None:
    """
    Function to test an agent by watching it play with the learnt policy

    :param config: the configuration for the testing (DictConfig)

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

    # load the in_colab flag to determine if the test is being done in colab or not
    in_colab = config.in_colab

    # get the device and print the gpu info if gpu is available
    device, gpu_info = get_device()
    if gpu_info:
        print(gpu_info)

    # define the render mode for the environment based on the fact that the program is being executed in colab or not
    render_mode = "rgb_array" if in_colab else "human"

    # create the testing environment as a wrapped atari environment with the parameters specified in the configuration
    # file
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

    # Instantiate the recorder wrapper around test environment to record and visualize the environment learning progress
    if in_colab:
        test_env = gym.wrappers.RecordVideo(test_env,
                                            video_folder=f'{config.home_directory}{config.test_video.output_folder}',
                                            name_prefix=f"{config.test_video.file_name}",
                                            video_length=math.inf)

    # import the model specified in the configuration to use for the testing
    model = getattr(importlib.import_module("src.models"), config.model)

    # import the agent to test specified in the configuration
    agent = getattr(importlib.import_module("src.agents"), config.agent)

    # initialize the specified agent with the testing environment and the parameters defined in the testing
    # configuration file
    agent = agent(env=test_env, testing_env=test_env, device=device, q_function=model, logger=logger,
                  home_directory=config.home_directory, buffered_avg_reward_size=config.buffered_avg_reward_size,
                  checkpoint_file="")

    # load the trained model from the file defined in the configuration
    agent.load(filename=config.output_model_file)

    # play
    agent.play()

    # close the testing environment
    test_env.close()

    # close and finish the logger
    if config.logging:
        logger.finish()


if __name__ == "__main__":
    tester()
