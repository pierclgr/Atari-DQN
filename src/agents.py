import glob
import os
from abc import ABC, abstractmethod
import random
from collections import deque
from src.logger import Logger
from typing import Type, Optional
import gym
import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm
from src.models import RLNetwork, DQNNetwork
from src.buffers import ReplayBuffer
from natsort import natsorted
from src.wrappers import VectorEnv
from src.utils import StateTransition


class Agent(ABC):
    """
    Abstract class Agent that defines the basic methods of an agent
    """

    @abstractmethod
    def __init__(self, env: gym.Env, **kwargs) -> None:
        """
        Constructor method that takes a gym environment and sets it as the environment the agent is interacting with

        :return: None
        """
        self.env = env
        pass

    @abstractmethod
    def get_action(self, **kwargs) -> int:
        """
        Method that allows the agent to select the action to perform at a certain timestep

        :return: the selected action to perform (int)
        """
        pass


class TrainableExperienceReplayAgent(Agent):
    """
    Class representing an agent that is trainable with experience replay (subclass of Agent)
    """

    def __init__(self,
                 testing_env: gym.Env,
                 device: torch.device,
                 home_directory: str,
                 checkpoint_file: str,
                 q_function: Type[RLNetwork] = DQNNetwork,
                 num_training_steps: int = 50000000,
                 batch_size: int = 32,
                 eps_max: float = 1,
                 eps_min: float = 0.1,
                 eps_decay_steps: int = 1000000,
                 target_update_steps: int = 10000,
                 learning_rate: float = 0.00025,
                 checkpoint_every: int = 2000,
                 buffer_capacity: int = 100000,
                 num_initial_replay_samples: int = 50000,
                 gradient_momentum: float = 0.95,
                 gradient_alpha: float = 0.95,
                 gradient_eps: float = 0.01,
                 buffered_avg_reward_size: int = 100,
                 criterion: nn.Module = None,
                 optimizer: torch.optim.Optimizer = None,
                 logger: Logger = None,
                 save_space: bool = True,
                 test_every: int = 2000,
                 **kwargs
                 ) -> None:
        """
        Constructor method of the TrainableExperienceReplayAgent class that takes all the parameters to instantiate a
        trainable agent

        :param testing_env: the environment on which the agent is tested (gym.Env)
        :param device: the device on which the agent is executed and the network trained (torch.Device)
        :param home_directory: the director of the current project (str)
        :param checkpoint_file: the prefix of the name of the model checkpoint file (str)
        :param q_function: the value function of the agent, which in this case is a deep neural network, the default is
            DQNNetwork (Type[RLNetwork)
        :param num_training_steps: the number of total training steps, default is 50M (int)
        :param batch_size: the size of the training batches, default is 32 (int)
        :param eps_max: the maximum value of epsilon, default is 1 (float)
        :param eps_min: the minimum value of epsilon, default is 0.1 (float)
        :param eps_decay_steps: the number of training steps in which to decay epsilon, default is 1M (int)
        :param target_update_steps: the number of steps after which to update the target network, default is 10K (int)
        :param learning_rate: the learning rate of the optimizer, default is 0.00025 (float)
        :param checkpoint_every: the number of steps after which to checkpoint the model, default is 2K (int)
        :param buffer_capacity: the capacity of the experience replay buffer, default is 100K (int)
        :param num_initial_replay_samples: the number of samples to put in the replay buffer at the beginning of the
            training, default is 50K (int)
        :param gradient_momentum: the momentum of the optimizer, default is 0.95 (float)
        :param gradient_alpha: the squared momentum of the optimizer, default is 0.95 (float)
        :param gradient_eps: the minimum squared gradient of the optimizer, default is 0.01 (float)
        :param buffered_avg_reward_size: the capacity of the episode reward buffer, default is 100 (int)
        :param criterion: the loss function, default value is None but if no criterion is specified, it is initialized
            to SmooothL1Loss (nn.Module)
        :param optimizer: the optimizer of the learning process, default is None but if no optimizer is specified, it is
            initialized to RMSprop (torch.optim.Optimizer)
        :param logger: the logger to use, if any, default is None (Logger)
        :param save_space: a boolean determining if we need to save space while checkpointing the models, if true, the
            checkpoint folder is emptied everytime we checkpoint the model to save space, default is True (bool),
        :param test_every: the number of training steps after which to test the environment, default is 2K (int)
        :param kwargs: other arguments to pass to the Agent class constructor, including the training environment

        :return: None
        """

        # call the constructor of the superclass Agent, passing the kwargs to initialize the training environment
        super(TrainableExperienceReplayAgent, self).__init__(**kwargs)

        # set the class parameters
        self.testing_env = testing_env
        self.num_training_steps = num_training_steps
        self.batch_size = batch_size
        self.device = device
        self.eps = eps_max
        self.eps_max = eps_max
        self.eps_min = eps_min
        self.eps_decay_steps = eps_decay_steps
        self.target_update_steps = target_update_steps  # // self.env.num_envs
        self.learning_rate = learning_rate
        self.logger = logger
        self.checkpoint_every = checkpoint_every
        self.home_directory = home_directory
        self.checkpoint_file = checkpoint_file
        self.num_initial_replay_samples = num_initial_replay_samples
        self.gradient_momentum = gradient_momentum
        self.gradient_alpha = gradient_alpha
        self.gradient_eps = gradient_eps
        self.save_space = save_space
        self.rew_buf_size = buffered_avg_reward_size
        self.test_every = test_every

        # initialize an empty replay buffer
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)

        # if the current training environment is a vectorized environment
        if isinstance(self.env, VectorEnv):
            # set the input shape of the model as the observation space of one of the environments
            self.input_shape = self.env.observation_space.shape[1:]

            # set the output channels of the model as the length of the action space of one of the environments
            self.output_channels = self.env.action_space[0].n
        else:
            # otherwise, set the input shape as the observation space of the environment
            self.input_shape = self.env.observation_space.shape

            # set the output channels of the model as the length of the action space of the environment
            self.output_channels = self.env.action_space.n

        # initialize the q function of the agent
        self.q_function = q_function(input_shape=self.input_shape,
                                     output_channels=self.output_channels).to(device=self.device)

        # initialize the target q function of the agent
        self.target_q_function = q_function(input_shape=self.input_shape,
                                            output_channels=self.output_channels).to(device=self.device)

        # copy the parameters of the q function to the parameters of the target q function
        self.target_q_function.load_state_dict(self.q_function.state_dict())

        # if criterion is None, set it to SmoothL1Loss
        if not criterion:
            criterion = nn.SmoothL1Loss()
        self.criterion = criterion

        # if optimizer is None, set the default optimizer to RMSprop
        if not optimizer:
            optimizer = torch.optim.RMSprop(self.q_function.parameters(), lr=self.learning_rate,
                                            momentum=self.gradient_momentum, alpha=self.gradient_alpha,
                                            eps=self.gradient_eps)
        self.optimizer = optimizer

    def initialize_experience(self, n_samples: int = None) -> None:
        """
        Method of the agent to initialize the replay buffer with n_samples samples

        :param n_samples: the number of samples to put in the buffer, default is None but if it is None, it is set to
            the number of initial replay samples (int)

        :return: None
        """

        # set the two networks to eval mode
        self.q_function.eval()
        self.target_q_function.eval()

        # disable gradient computation
        with torch.no_grad():
            # if n_samples is not specified we set it to the number of initial replay samples
            if n_samples is None:
                n_samples = self.num_initial_replay_samples

            print(f"Initializing replay buffer with {n_samples} samples...")

            # reset the environment and get the initial state to start a new episode, then convert it to float
            previous_states = self.env.reset()
            previous_states = np.asarray(previous_states)
            previous_states = torch.as_tensor(previous_states).to(self.device).float()

            # initialize the progress bar for the replay buffer initialization
            init_pbar = tqdm(total=n_samples)

            # while the buffer is not full
            full = False
            while not full:
                # select an action to perform randomly, using eps=1 to select the action only randomly
                actions = self.get_action(previous_states, eps=1, train=True)

                # perform the selected action on the training environment and get the new state
                current_states, rewards, dones, _ = self.env.step(actions)

                # convert the initial state to torch tensor
                current_states = np.asarray(current_states)
                current_states = torch.as_tensor(current_states).to(self.device).float()

                # for each state transition
                for previous_state, action, reward, current_state, done in zip(previous_states, actions, rewards,
                                                                               current_states, dones):
                    # store all of them in the replay buffer
                    self.store_experience(
                        StateTransition(state=previous_state.unsqueeze(dim=0), action=action, reward=reward,
                                        next_state=current_state.unsqueeze(dim=0), done=done))

                    # update the progress bar
                    init_pbar.update(1)

                    # stop filling the replay buffer if it has been filled with the specified number of samples
                    if len(self.replay_buffer) >= n_samples:
                        full = True
                        break

                # set the previous state(s) to current state(s)
                previous_states = current_states

    def sample_experience(self, num_samples: int = 1) -> StateTransition:
        """
        Method to sample N samples from the replay buffer

        :param num_samples: the number of samples to sample from the replay buffer, default is 1 (int)

        :return: a state transition, which is composed of batches of states, actions, rewards, next states and dones
            (StateTransition)
        """

        # sample num_samples StateTransition samples from the replay buffer
        batch = self.replay_buffer.sample(num_samples=num_samples)

        # convert the list of StateTransition samples to batches of states, actions, rewards, next states and dones
        states_batch = [sample.state for sample in batch]
        actions_batch = [sample.action for sample in batch]
        rewards_batch = [sample.reward for sample in batch]
        next_states_batch = [sample.next_state for sample in batch]
        dones_batch = [sample.done for sample in batch]

        # convert the batches to PyTorch tensors and send them to the device
        states_batch = torch.cat(states_batch, dim=0).to(self.device)
        actions_batch = torch.as_tensor(actions_batch).to(self.device)
        rewards_batch = torch.as_tensor(rewards_batch).to(self.device)
        next_states_batch = torch.cat(next_states_batch, dim=0).to(self.device)
        dones_batch = torch.as_tensor(dones_batch).to(self.device)

        # return the batches as a StateTransition of batches
        return StateTransition(states_batch, actions_batch, rewards_batch, next_states_batch, dones_batch)

    def reset_experience(self) -> None:
        """
        Method to reset the replay buffer by emptying it

        :return: None
        """

        self.replay_buffer.reset()

    def store_experience(self, state_transition: StateTransition) -> None:
        """
        Method to store samples in the experience replay buffer

        :param state_transition: the StateTransition sample to store in the buffer (StateTransition)

        :return: None
        """

        # move the state and next state tensor to CPU to store them on RAM memory and preserve GPU memory
        state_transition.state = state_transition.state.to("cpu")
        state_transition.next_state = state_transition.next_state.to("cpu")

        # store the StateTransition sample in the buffer
        self.replay_buffer.store(state_transition)

    def train(self) -> None:
        """
        Method to train the agent

        :return: None
        """

        # set the two networks to training mode
        self.q_function.train()
        self.target_q_function.train()

        # load the checkpoint
        checkpoint_info = self.checkpoint_load()

        # if no checkpoint is available
        if not checkpoint_info:
            # set all metrics and counters to 0
            num_done_episodes = 0
            train_loss = 0
            total_reward_buffer = 0
            total_steps = 0
            test_total_reward_buffer = 0
            episode_rewards = np.asarray([0 for _ in range(self.env.num_envs)], dtype=np.float64)
            episode_reward = 0
            test_episode_reward = 0
            num_test_episodes = 0
            value_estimate = 0
            reward_buffer = deque([], maxlen=self.rew_buf_size)
            test_reward_buffer = deque([], maxlen=self.rew_buf_size)

            # initialize replay buffer to specified capacity
            self.initialize_experience()
        else:
            # otherwise, load metrics, counters and buffers from the checkpoint
            num_done_episodes, num_test_episodes, train_loss, total_reward_buffer, eps, total_steps, \
                test_total_reward_buffer, episode_rewards, episode_reward, test_episode_reward, reward_buffer, \
                test_reward_buffer, value_estimate = checkpoint_info

            # set epsilon of the agent to the epsilon loaded from the checkpoint
            self.eps = eps

        # compute the initial average reward for training and testing
        average_episode_reward = float((total_reward_buffer / num_done_episodes) if num_done_episodes > 0 else 0)
        test_average_episode_reward = float(
            (test_total_reward_buffer / num_test_episodes) if num_test_episodes > 0 else 0)

        # compute the initial buffered average reward for training and testing
        buf_average_reward = float(np.mean(reward_buffer).item() if reward_buffer else 0)
        buf_test_average_reward = float(np.mean(test_reward_buffer).item() if test_reward_buffer else 0)

        # if logging is required, namely logger is not None, we log all the data at the beginning of the training
        if self.logger:
            self.logger.log("train_loss", train_loss, total_steps)
            self.logger.log("eps", self.eps, total_steps)
            self.logger.log("train_average_episode_reward", average_episode_reward, total_steps)
            self.logger.log("test_average_episode_reward", test_average_episode_reward, total_steps)
            self.logger.log("total_episodes", num_done_episodes, total_steps)
            self.logger.log("buffer_samples", len(self.replay_buffer), total_steps)
            self.logger.log("train_episode_reward", episode_reward, total_steps)
            self.logger.log("test_episode_reward", test_episode_reward, total_steps)
            self.logger.log("buffered_train_average_episode_reward", buf_average_reward, total_steps)
            self.logger.log("buffered_test_average_episode_reward", buf_test_average_reward, total_steps)
            self.logger.log("value_estimate", value_estimate, total_steps)

        # if checkpoint has been loaded, print all the info of the checkpoint
        if checkpoint_info:
            print(f"Loaded checkpoint:")
            print(f"\t- episodes: {num_done_episodes}\n"
                  f"\t- eps: {self.eps}\n"
                  f"\t- total_steps: {total_steps}\n"
                  f"\t- buffer_samples: {len(self.replay_buffer)}\n"
                  f"\t- train_loss: {train_loss}\n"
                  f"\t- train_average_episode_reward: {average_episode_reward}\n"
                  f"\t- test_average_episode_reward: {test_average_episode_reward}\n"
                  f"\t- buffered_train_average_episode_reward: {buf_average_reward}\n"
                  f"\t- buffered_test_average_episode_reward: {buf_test_average_reward}\n"
                  f"\t- train_episode_reward: {episode_reward}\n"
                  f"\t- test_episode_reward: {test_episode_reward}\n"
                  f"\t- value_estimate: {value_estimate}")

        print(f"Training for {max(0, self.num_training_steps - total_steps)} steps...")

        # reset the environment and get the initial state(s) to start a new episode(s), then convert it to a PyTorch
        # tensor
        previous_states = self.env.reset()
        previous_states = np.asarray(previous_states)
        previous_states = torch.as_tensor(previous_states).to(self.device).float()

        # initialize the training progress bar
        train_pbar = tqdm(total=self.num_training_steps)

        # train the agent for the defined number of training steps
        while total_steps < self.num_training_steps:

            # decay epsilon accordingly to the current number of training steps
            self.eps_decay(total_steps)

            # select an action(s) to perform accordingly to the policy based on the value of epsilon
            actions = self.get_action(previous_states, train=True)

            # perform the selected action(s) and get the new state(s)
            current_states, rewards, dones, _ = self.env.step(actions)

            # convert the current state(s) and cast and convert to a PyTorch tensor
            current_states = np.asarray(current_states)
            current_states = torch.as_tensor(current_states).to(self.device).float()

            # for each environment of the vectorized environment
            for i in range(self.env.num_envs):
                # get the result of the step for the current environment
                previous_state = previous_states[i]
                action = actions[i]
                reward = rewards[i]
                current_state = current_states[i]
                done = dones[i]

                # store the state transition of the current environment in the replay buffer
                self.store_experience(
                    state_transition=StateTransition(state=previous_state.unsqueeze(dim=0), action=action,
                                                     reward=reward, next_state=current_state.unsqueeze(dim=0),
                                                     done=done))

            # sample a random minibatch of state transitions from the replay buffer
            state_transitions_batch = self.sample_experience(num_samples=self.batch_size)

            # do a training step
            train_loss, value_estimate = self.training_step(state_transitions_batch=state_transitions_batch)
            value_estimate = torch.mean(value_estimate).item()

            # update the number of total steps
            total_steps += 1

            # update the progress bar
            train_pbar.update(1)

            # every target_update_steps gradient descent steps, we update the target network weights
            self.update_target_network(total_steps % self.target_update_steps == 0)

            # increment the episode reward(s) with the current reward(s)
            episode_rewards += rewards

            # get the episode reward(s) if the episode(s) is done
            done_episode_rewards = episode_rewards[dones]

            # update the latest episode reward with the most recent done episode
            episode_reward = done_episode_rewards[0] if len(done_episode_rewards) > 1 else episode_reward

            # append the episode reward(s) of the done episode(s) to the reward buffer
            total_reward_buffer += np.sum(done_episode_rewards)
            reward_buffer.extend(done_episode_rewards)

            # increment the number of total episodes
            num_done_episodes += np.count_nonzero(dones)

            # set to 0 the episode reward(s) for done episode(s)
            episode_rewards *= np.logical_not(dones)

            # compute the training average reward
            average_episode_reward = float((total_reward_buffer / num_done_episodes) if num_done_episodes > 0 else 0)
            buf_average_reward = float(np.mean(reward_buffer).item() if reward_buffer else 0)

            # test the agent every test_every gradient steps
            if total_steps % self.test_every == 0:
                # test the agent
                test_episode_reward = self.test()

                # the test method sets the two networks to eval mode, so reset the two networks to training mode
                self.q_function.train()
                self.target_q_function.train()

                # add test reward to the test reward buffer
                test_total_reward_buffer += test_episode_reward
                test_reward_buffer.append(test_episode_reward)

                # increment the number of test episodes
                num_test_episodes += 1

                print(
                    f"Episodes: {num_done_episodes}, num_test_episodes: {num_test_episodes}, "
                    f"eps: {self.eps}, total_steps: {total_steps}, "
                    f"total_env_steps: {total_steps * self.env.num_envs}, "
                    f"buffer_samples: {len(self.replay_buffer)}, train_loss: {train_loss}, "
                    f"train_average_episode_reward: {average_episode_reward}, "
                    f"test_average_episode_reward: {test_average_episode_reward}, "
                    f"buffered_train_average_episode_reward: {buf_average_reward}, "
                    f"buffered_test_average_episode_reward: {buf_test_average_reward}, "
                    f"train_episode_reward: {episode_reward}, "
                    f"test_episode_reward: {test_episode_reward}, "
                    f"value_estimates: {value_estimate}"
                )

            # compute the test average reward
            test_average_episode_reward = float(
                (test_total_reward_buffer / num_test_episodes) if num_test_episodes > 0 else 0)
            buf_test_average_reward = float(np.mean(test_reward_buffer).item() if test_reward_buffer else 0)

            # if logging is required, we log data at every training step
            if self.logger:
                self.logger.log("train_loss", train_loss, total_steps)
                self.logger.log("eps", self.eps, total_steps)
                self.logger.log("train_average_episode_reward", average_episode_reward, total_steps)
                self.logger.log("test_average_episode_reward", test_average_episode_reward, total_steps)
                self.logger.log("total_episodes", num_done_episodes, total_steps)
                self.logger.log("buffer_samples", len(self.replay_buffer), total_steps)
                self.logger.log("train_episode_reward", episode_reward, total_steps)
                self.logger.log("test_episode_reward", test_episode_reward, total_steps)
                self.logger.log("buffered_train_average_episode_reward", buf_average_reward, total_steps)
                self.logger.log("buffered_test_average_episode_reward", buf_test_average_reward, total_steps)
                self.logger.log("value_estimate", value_estimate, total_steps)

            # checkpoint the model every checkpoint_every gradient steps
            if total_steps % self.checkpoint_every == 0:
                print(f"Checkpointing model at step {total_steps}...")
                checkpoint_info = {'episode': num_done_episodes,
                                   'test_episode': num_test_episodes,
                                   'total_steps': total_steps,
                                   'eps': self.eps,
                                   'train_loss': train_loss,
                                   'total_reward_buffer': total_reward_buffer,
                                   'test_total_reward_buffer': test_total_reward_buffer,
                                   "reward_buffer": reward_buffer,
                                   "test_reward_buffer": test_reward_buffer,
                                   "episode_rewards": episode_rewards,
                                   "episode_reward": episode_reward,
                                   "test_episode_reward": test_episode_reward,
                                   "value_estimate": value_estimate}

                filename = f"{self.checkpoint_file}_step_{total_steps}"

                # checkpoint the training
                self.checkpoint_save(filename=filename, checkpoint=checkpoint_info)

            # set previous state(s) to current state(s)
            previous_states = current_states

        train_pbar.close()
        print("Done.")

    def update_target_network(self, update: bool = True) -> None:
        """
        Method to update the target q network by copying weights from the online q network

        :params update: boolean value to decide if to update the network or not, default is True (bool)

        :return: None
        """
        # update the target q network weights if update is True
        if update:
            self.target_q_function.load_state_dict(self.q_function.state_dict())

    def test(self) -> int:
        """
        Method to test the agent for an episode

        :return: the total reward of the test episode (int)
        """

        # set the two networks to eval mode
        self.q_function.eval()
        self.target_q_function.eval()

        # disable gradient computation
        with torch.no_grad():
            print(f"Testing...")

            # reset the environment and get the initial state to start a new testing episode
            previous_state = self.testing_env.reset()

            # convert the initial state to torch tensor, unsqueeze it to feed it as a single sample to the network and
            # convert it to float PyTorch tensor
            previous_state = np.asarray(previous_state)
            previous_state = torch.as_tensor(previous_state).to(self.device).unsqueeze(axis=0).float()
            done = False
            test_episode_reward = 0

            # while the episode is not done
            test_pbar = tqdm(total=self.testing_env.spec.max_episode_steps, position=0)
            while not done:
                # select an action to perform based on the agent policy using epsilon=0 to use the learnt policy
                action = self.get_action(previous_state, eps=0, train=False)

                # perform the selected action on the testing environment and get the new state
                current_state, reward, done, _ = self.testing_env.step(action)

                # add the reward to the total reward of the current episode
                test_episode_reward += reward

                # convert the current state to PyTorch float tensor
                current_state = np.asarray(current_state)
                current_state = torch.as_tensor(current_state).to(self.device).unsqueeze(axis=0).float()

                # set the next previous state to the current one
                previous_state = current_state

                # update the test progress bar
                test_pbar.update(1)
            test_pbar.close()

        return test_episode_reward

    def save(self, filename: str) -> None:
        """
        Method to save the model weights

        :param filename: the name of the file in which to save the weights of the network (str)

        :return: None
        """

        trained_model_path = f"{self.home_directory}trained_models/"
        if not os.path.isdir(trained_model_path):
            os.makedirs(trained_model_path)
        file_path = f"{trained_model_path}{filename}.pt"

        print(f"Saving trained model to {filename}.pt...")

        # save network weights
        checkpoint = {"model_weights": self.q_function.state_dict()}
        torch.save(checkpoint, file_path)

    def checkpoint_save(self, filename: str, checkpoint: dict) -> None:
        """
        Method to checkpoint the model and save the training state

        :param filename: the name of the ifle in which to save the checkpoint (str)
        :param checkpoint: a dictionary containing the informations to save in the checkpoint (dict)

        :return: None
        """

        checkpoint_path = f"{self.home_directory}trained_models/checkpoints/"
        if not os.path.isdir(checkpoint_path):
            os.makedirs(checkpoint_path)
        file_path = f"{checkpoint_path}{filename}.pt"

        checkpoint['model_weights'] = self.q_function.state_dict()
        checkpoint['optimizer_weights'] = self.optimizer.state_dict()
        checkpoint['replay_buffer'] = self.replay_buffer

        # if save_space is required, remove old checkpoints to save storage
        folder = glob.glob(f"{checkpoint_path}*")
        if self.save_space:
            for file in folder:
                os.remove(file)

        # checkpoint the training
        torch.save(checkpoint, file_path)

    def load(self, filename: str) -> None:
        """
        Method to load a trained model from a file

        :param filename: the name of the file from which to load the model (str)

        :return: None
        """

        filename = f"{filename}.pt"
        trained_model_path = f"{self.home_directory}trained_models/"
        if os.path.isdir(trained_model_path):
            file_path = f"{trained_model_path}{filename}"
            if os.path.isfile(file_path):
                print(f"Loading model from {filename}...")
                checkpoint = torch.load(file_path)
                self.q_function.load_state_dict(checkpoint['model_weights'])
            else:
                print("The specified file does not exist in the trained models directory.")
        else:
            print("The directory of the trained models does not exist.")

    def checkpoint_load(self) -> Optional[tuple]:
        """
        Method to load the latest training checkpoint

        :return: either None if no checkpoint is available or a tuple with the checkpoint data if available
            (Optional[Tuple])
        """

        checkpoint_path = f"{self.home_directory}trained_models/checkpoints/"

        # if the folder with checkpoints exists and is not empty
        if os.path.isdir(checkpoint_path):
            # sort the files in the folder in alphabelical order
            sorted_checkpoint_files = natsorted(glob.glob(f'{checkpoint_path}{self.checkpoint_file}*.pt'))

            # if there's any file in the folder
            if len(sorted_checkpoint_files) > 0:

                # get the most recent checkpoint
                latest_checkpoint_file = sorted_checkpoint_files[-1]

                # load checkpoint information from the file
                print(f"Loading checkpoint from file {os.path.basename(latest_checkpoint_file)}...")
                checkpoint = torch.load(latest_checkpoint_file)

                self.q_function.load_state_dict(checkpoint['model_weights'])
                self.optimizer.load_state_dict(checkpoint['optimizer_weights'])
                self.replay_buffer = checkpoint['replay_buffer']

                print("Model, optimizer and replay buffer loaded from checkpoint.")

                episode = checkpoint['episode']
                num_test_episodes = checkpoint["test_episode"]
                eps = checkpoint['eps']
                total_steps = checkpoint['total_steps']
                train_loss = checkpoint['train_loss']
                total_reward_buffer = checkpoint['total_reward_buffer']
                test_total_reward_buffer = checkpoint['test_total_reward_buffer']
                reward_buffer = checkpoint['reward_buffer']
                test_reward_buffer = checkpoint['test_reward_buffer']
                episode_rewards = checkpoint["episode_rewards"]
                episode_reward = checkpoint["episode_reward"]
                test_episode_reward = checkpoint["test_episode_reward"]
                value_estimate = checkpoint["value_estimate"]

                return_tuple = (episode,
                                num_test_episodes,
                                train_loss,
                                total_reward_buffer,
                                eps,
                                total_steps,
                                test_total_reward_buffer,
                                episode_rewards,
                                episode_reward,
                                test_episode_reward,
                                reward_buffer,
                                test_reward_buffer,
                                value_estimate)

                return return_tuple
            else:
                # no file exists in the folder, so return None
                print("No checkpoint file found. Training is starting from the beginning...")
                return None
        else:
            # the checkpoint folder does not exist, so return None
            print("The checkpoint folder does not exist. Training is starting from the beginning...")
            return None

    def play(self) -> None:
        """
        Method to see the agent play with the learnt policy

        :return: None
        """

        print("Playing...")
        # set the two networks to eval mode
        self.q_function.eval()
        self.target_q_function.eval()

        # while the episode is not done
        with torch.no_grad():
            test_pbar = tqdm()

            # reset the environment and get the initial state to start a new episode
            previous_state = self.testing_env.reset()

            # convert the initial state to PyTorch float tensor, unsqueeze it to feed it as a sample to the network
            previous_state = np.asarray(previous_state)
            previous_state = torch.as_tensor(previous_state).to(self.device).unsqueeze(axis=0).float()
            exit = False

            # initialize the metrics
            test_episode_reward = 0
            test_total_reward_buffer = 0
            test_reward_buffer = deque([], maxlen=self.rew_buf_size)
            cur_episode = 0
            total_steps = 0

            # let the agent play for multiple episodes until explicitly interrupted
            print(f"Episode {cur_episode + 1}...")
            while not exit:
                try:
                    # select an action to perform based on the agent policy using epsilon=0 to use the learnt policy
                    action = self.get_action(previous_state, eps=0, train=False)

                    # perform the selected action and get the new state
                    current_state, reward, done, info = self.testing_env.step(action)

                    # add the reward to the total reward of the current episode
                    test_episode_reward += reward

                    # convert the initial state to PyTorch float tensor and unsqueeze it
                    current_state = np.asarray(current_state)
                    current_state = torch.as_tensor(current_state).to(self.device).unsqueeze(axis=0).float()

                    # compute the test average reward
                    test_average_episode_reward = test_total_reward_buffer / (cur_episode + 1)
                    buf_test_average_reward = float(np.mean(test_reward_buffer).item() if test_reward_buffer else 0)

                    # if logging is required, we log the data at the end of each step
                    if self.logger:
                        self.logger.log("test_average_episode_reward", test_average_episode_reward, total_steps)
                        self.logger.log("total_episodes", cur_episode, total_steps)
                        self.logger.log("test_episode_reward", test_episode_reward, total_steps)
                        self.logger.log("buffered_test_average_episode_reward", buf_test_average_reward, total_steps)

                    # if the episode is done
                    if done:

                        # add the episode reward to the episode reward buffer
                        test_reward_buffer.append(test_episode_reward)
                        test_total_reward_buffer += test_episode_reward

                        print(f"Episode {cur_episode + 1} - "
                              f"total_steps: {total_steps},"
                              f"test_average_episode_reward: {test_average_episode_reward}, "
                              f"buffered_test_average_episode_reward: {buf_test_average_reward}, "
                              f"test_episode_reward: {test_episode_reward}")

                        # increment the number of episodes
                        cur_episode += 1

                        # reset the progress bar
                        test_pbar.reset()

                        # reset episode reward
                        test_episode_reward = 0

                        print(f"Episode {cur_episode + 1}...")

                        # reset the environment and get the initial state of the environment, then convert as PyTorch
                        # float tensor
                        previous_state = self.env.reset()
                        previous_state = np.asarray(previous_state)
                        previous_state = torch.as_tensor(previous_state).to(self.device).unsqueeze(axis=0).float()
                    else:
                        # if the episode is not done, set the current state as the previous state
                        previous_state = current_state

                        # update the testing progress bar
                        test_pbar.update(1)

                    # increment the number of total steps
                    total_steps += 1

                # catch interruption exception (like pressing CTRL + C) in order to exit the training
                except (KeyboardInterrupt, SystemExit):
                    print("Playing interrupted, exiting...")
                    exit = True

            # close the progress bar
            test_pbar.close()

    @abstractmethod
    def compute_labels_and_predictions(self, state_transitions_batch: StateTransition) -> tuple:
        """
        Abstract method to compute the label and predictions for the loss

        :param state_transitions_batch: the StateTransition batch to use for the computation of labels (target q values)
            and predictions (q values)

        :return: tuple containing the computed target q values and q values (tuple)
        """
        pass

    def get_action(self, state: torch.Tensor, eps: float = None, train: bool = True) -> torch.Tensor:
        """
        Method to get an action accordingly to the policies defined by the values of epsilon

        :param state: the current state to evaluate to select the action ot take (torch.Tensor)
        :param eps: the value of epsilon to use to select under which policy to act, default is None but if no value is
            provided then the current agent epsilon is used (float)
        :param train: boolean that defines if to use the training or testing environment, default is True (bool)

        :return: a PyTorch tensor containing the actions for each environment (if vectorized) selected by the agent
            (torch.Tensor)
        """

        # if epsilon is not specified, use the current agent epsilon
        if eps is None:
            eps = self.eps

        # select if to explore or exploit accordingly to given epsilon
        explore = random.choices([True, False], weights=[eps, 1 - eps], k=1)[0]

        if explore:
            # if explore, select action(s) to perform randomly
            if train:
                action = self.env.action_space.sample()
            else:
                action = self.testing_env.action_space.sample()

            # convert to PyTorch tensor
            action = torch.as_tensor(np.asarray(action))
        else:
            # otherwise, exploit the learnt policy and select action(s) to take based on the learnt policy
            q_values = self.q_function(state)

            # get the action(s) with the maximum value as the action(s) to perform
            action = torch.argmax(q_values, dim=1)

        return action

    def eps_decay(self, steps: int) -> None:
        """
        Method to decay agent's epsilon accordingly to a certain number of steps

        :param steps: the number of steps defining the value of epsilon (int)

        :return: None
        """

        # decay epsilon accordingly to the specified number of steps and the range of min-max epislon value, also
        # considering the maximum number of epsilon decay steps
        new_eps = (steps * ((self.eps_min - self.eps_max) / self.eps_decay_steps)) + self.eps_max

        # clip epsilon value to the minimum possible epsilon
        self.eps = max(new_eps, self.eps_min)

    def compute_loss(self, target_q_values: torch.Tensor, q_values: torch.Tensor) -> torch.Tensor:
        """
        Method to compute the loss between target q values and q values

        :param target_q_values: the target q values (labels) to use for the loss computation (torch.Tensor)
        :param q_values: the q values (predictions) to use for the loss computation (torch.Tensor)

        :return: the loss value (torch.Tensor)
        """

        # compute the loss accordingly to the agent's criterion and return it
        loss = self.criterion(target_q_values, q_values)
        return loss

    @staticmethod
    def gradient_descent_step(loss: torch.Tensor, optimizer: torch.optim.Optimizer) -> None:
        """
        Static method to perform a gradient descent step accordingly to a computed loss and an optimizer

        :param loss: the computed loss to use for the gradient descent step (torch.Tensor)
        :param optimizer: the optimizer to use for the gradient descent (torch.optim.Optimizer)

        :return: None
        """

        # zero the gradients of the optimizer and do a backpropagation gradient step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def training_step(self, state_transitions_batch: StateTransition) -> tuple:
        """
        Method to perform a training step

        :param state_transitions_batch: a state transition composed of batches of states, actions, rewards, next states
            and dones (StateTransition)

        :return: a tuple containing the loss value and the value estimate (tuple)
        """

        # compute target q values and q values for loss computation
        target_q_values, q_values = self.compute_labels_and_predictions(state_transitions_batch)

        # now we compute the loss between predicted values and the target values
        loss = self.compute_loss(target_q_values, q_values)

        # now we perform a gradient descent step over the parameters of the q_function
        self.gradient_descent_step(loss, self.optimizer)

        return loss.detach().item(), target_q_values


class DQNAgent(TrainableExperienceReplayAgent):
    """
    Class that describes an agent learning using the DQN algorithm, subclass of TrainableExperienceReplayAgent
    """

    def __init__(self, discount_rate: float = 0.99, **kwargs) -> None:
        """
        Constructor method to initialize the agent

        :param discount_rate: the discount rate for value estimates computation, default is 0.99 (float)
        :param kwargs: other arguments to pass to the superclass constructor (dict)

        :return: None
        """
        super().__init__(**kwargs)
        self.discount_rate = discount_rate

    def compute_labels_and_predictions(self, state_transitions_batch: StateTransition) -> tuple:
        """
        Method to compute labels and predictions

        :param state_transitions_batch: the StateTransition batch to use for the computation of labels (target q values)
            and predictions (q values)

        :return: tuple containing the computed target q values and q values (tuple)
        """

        # compute the target q values accordingly to the agent's algorithm
        target_q_values = self.compute_target_q_values(state_transitions_batch)

        # then, we apply a discount rate to the future rewards
        target_q_values = self.discount_rate * target_q_values

        # now, we need to zero the future rewards that correspond to states that are final states; in fact,
        # as said before, future rewards are used in the computations just for current states which are not
        # final states; for final states, the actual reward is just the reward of the current state
        # we can do this by multiplying a boolean tensor with the tensor of future rewards: this will
        # produce a tensor where the discounted future rewards are zeroed where the boolean tensor is False
        # we need to zero the discounted future rewards for the final states, so the states that are True in
        # the done batch, so we simply multiply the discounted future rewards tensor by the opposite of the
        # done batch
        target_q_values = target_q_values * torch.logical_not(state_transitions_batch.done)

        # finally, we sum the resulting tensor with the reward batch, resulting thus in a tensor with only
        # the reward for final states and the discounted future reward plus the actual reward for non-final
        # states
        target_q_values = state_transitions_batch.reward + target_q_values

        # we now compute the estimated rewards for the current states using the q_function
        q_values = self.q_function(state_transitions_batch.state)

        # let's now get the number of possible actions for the current environment
        num_actions = self.env.action_space[0].n if isinstance(self.env, VectorEnv) else self.env.action_space.n

        # the previously computed tensor contains the estimated reward for each of the possible action;
        # since we need to compute the loss between these estimated rewards and the target rewards computed
        # before, this latter tensor only contains the future reward with no information about the action,
        # so what we do is computing a one-hot tensor which zeroes the estimated rewards for actions that
        # are not the actual taken actions
        one_hot_actions = torch.nn.functional.one_hot(state_transitions_batch.action, num_actions)

        # by multiplying the estimated reward tensor and the one hot action tensor, we will get a tensor of
        # the same shape that contains 0 as a reward for actions that are not the current action while
        # contains the estimated reward for the action that is the current action
        q_values *= one_hot_actions

        # we then sum the estimated along the actions dimension to get the final a tensor with only one
        # reward per sample that will be the only reward that was not zeroed out in the previous step
        # (because we will sum zeros with only one reward value)
        q_values = torch.sum(q_values, dim=1)

        return target_q_values, q_values

    def compute_target_q_values(self, state_transitions_batch: StateTransition) -> torch.Tensor:
        """
        Method to compute the target q values accordingly to DQN algorithm

        :param state_transitions_batch: the StateTransition batch to use for the computation of labels (target q values)

        :return: a PyTorch tensor containing the computed target q values (torch.Tensor)
        """
        # as a first step, we feed the batch of next states to the target network to compute the future
        # rewards, namely the rewards for the next state
        target_q_values = self.target_q_function(state_transitions_batch.next_state)

        # as second step, we get the maximum value for each of the predicted future rewards, so we select
        # the reward corresponding to the action with the highest return
        target_q_values, _ = torch.max(target_q_values, dim=1)

        return target_q_values


class DoubleDQNAgent(DQNAgent):
    """
    Class that describes an agent learning using the DQN algorithm, subclass of DQNAgent
    """

    def __init__(self, **kwargs) -> None:
        """
        Constructor method to initialize the agent

        :param kwargs: other arguments to pass to the superclass constructor (dict)

        :return: None
        """

        super().__init__(**kwargs)

    def compute_target_q_values(self, state_transitions_batch: StateTransition) -> torch.Tensor:
        """
        Method to compute the target q values accordingly to Dobule DQN algorithm

        :param state_transitions_batch: the StateTransition batch to use for the computation of labels (target q values)

        :return: a PyTorch tensor containing the computed target q values (torch.Tensor)
        """

        # as a first step, we feed the next states to the q function (q_network) to get a value for each possible
        # action; the output will be a tensor containing a value for each possible action for all the next states of
        # the batch
        q_values = self.q_function(state_transitions_batch.next_state)

        # once we did this, for each next state we get the action to select with the highest value
        actions_with_max_value = torch.argmax(q_values, dim=1)

        # now, we feed the batch of next states to the target q function (target q network) to compute the target action
        # value estimates for each of the next states; the output will be a tensor containing an estimated value for
        # each possible action for all the next states of the batch
        target_q_values = self.target_q_function(state_transitions_batch.next_state)

        # let's now get the number of possible actions for the current environment
        num_actions = self.env.action_space[0].n if isinstance(self.env, VectorEnv) else self.env.action_space.n

        # now we compute a one-hot tensor that we will use to zero out the target q values of actions that are not the
        # actions with the maximum estimated values
        one_hot_actions = torch.nn.functional.one_hot(actions_with_max_value, num_actions)

        # by multiplying the tensor of the estimates with the one hot action tensor, we will get a tensor of the same
        # shape that contains 0 as reward for actions that are not the actions with the highest value while it
        # will also contain the estimated value for actions that are the actions with the highest value
        target_q_values *= one_hot_actions

        # we then sum the estimated along the actions dimension to get the final a tensor and to get rid of the extra
        # dimension; since we have 0 for the actions for each next state that we're not interested into, by summing
        # along a dimension, we get rid of the "action" dimension by only keeping the estimated value for the action
        # with the highest value
        target_q_values = torch.sum(target_q_values, dim=1)

        return target_q_values
