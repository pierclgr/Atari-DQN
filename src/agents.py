import glob
import math
import os
from abc import ABC, abstractmethod
import random
from collections import deque

from copy import deepcopy
from ctypes import Array

from src.logger import Logger, WandbLogger
from typing import Type, Tuple, Callable, Optional, List, Union, Collection

import gym
import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm

from src.models import RLNetwork, DQNNetwork
from src.buffers import ReplayBuffer
from natsort import natsorted

import signal

# TODO documentation

# define a superclass "Agent" for all the possible Agents; this will act as an interface for the other classes
from src.utils import StateTransition
from src.wrappers import SubprocVecEnv


class Agent(ABC):
    @abstractmethod
    def __init__(self, env: gym.Env, **kwargs) -> None:
        self.env = env
        pass

    @abstractmethod
    def get_action(self, **kwargs) -> int:
        pass


# create a trainable agent with experience replay
class TrainableExperienceReplayAgent(Agent):
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
                 test_every: int = 10,
                 **kwargs
                 ) -> None:

        super(TrainableExperienceReplayAgent, self).__init__(**kwargs)

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

        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)

        if isinstance(self.env, SubprocVecEnv):
            self.input_shape = self.env.observation_space.shape[1:]
            self.output_channels = self.env.action_space[0].n
        else:
            self.input_shape = self.env.observation_space.shape
            self.output_channels = self.env.action_space.n

        self.q_function = q_function(input_shape=self.input_shape,
                                     output_channels=self.output_channels).to(device=self.device)

        self.target_q_function = q_function(input_shape=self.input_shape,
                                            output_channels=self.output_channels).to(device=self.device)
        self.target_q_function.load_state_dict(self.q_function.state_dict())

        if not criterion:
            criterion = nn.SmoothL1Loss()

        self.criterion = criterion

        if not optimizer:
            optimizer = torch.optim.RMSprop(self.q_function.parameters(), lr=self.learning_rate,
                                            momentum=self.gradient_momentum, alpha=self.gradient_alpha,
                                            eps=self.gradient_eps)

        self.optimizer = optimizer

    def initialize_experience(self, n_samples: int = None):
        self.q_function.eval()
        self.target_q_function.eval()
        with torch.no_grad():
            # if n_samples is not specified we set it to the initial number of replay samples
            if n_samples is None:
                n_samples = self.num_initial_replay_samples

            print(f"Initializing replay buffer with {n_samples} samples...")

            # reset the environment and get the initial state to start a new episode
            previous_states = self.env.reset()
            previous_states = np.asarray(previous_states)
            previous_states = torch.as_tensor(previous_states).to(self.device).float()

            # fill the replay buffer with the defined number of samples
            init_pbar = tqdm(total=self.num_initial_replay_samples)
            full = False
            while not full:
                # select an action to perform randomly, using eps=1 to select the action only randomly
                actions = self.get_action(previous_states, eps=1, train=True)

                # perform the selected action and get the new state
                current_states, rewards, dones, _ = self.env.step(actions)

                # convert the initial state to torch tensor and cast to float tensor
                current_states = np.asarray(current_states)
                current_states = torch.as_tensor(current_states).to(self.device).float()

                # add all the state transitions to the buffer
                for previous_state, action, reward, current_state, done in zip(previous_states, actions, rewards,
                                                                               current_states, dones):
                    self.store_experience(
                        StateTransition(state=previous_state.unsqueeze(dim=0), action=action, reward=reward,
                                        next_state=current_state.unsqueeze(dim=0), done=done))

                    init_pbar.update(1)

                    # stop filling if replay buffer is full
                    if len(self.replay_buffer) >= self.num_initial_replay_samples:
                        full = True
                        break

                # set the previous state to current state
                previous_states = current_states

    def sample_experience(self, num_samples: int = 1) -> StateTransition:
        batch = self.replay_buffer.sample(num_samples=num_samples)

        states_batch = [sample.state for sample in batch]
        actions_batch = [sample.action for sample in batch]
        rewards_batch = [sample.reward for sample in batch]
        next_states_batch = [sample.next_state for sample in batch]
        dones_batch = [sample.done for sample in batch]

        states_batch = torch.cat(states_batch, dim=0).to(self.device)
        actions_batch = torch.as_tensor(actions_batch).to(self.device)
        rewards_batch = torch.as_tensor(rewards_batch).to(self.device)
        next_states_batch = torch.cat(next_states_batch, dim=0).to(self.device)
        dones_batch = torch.as_tensor(dones_batch).to(self.device)

        return StateTransition(states_batch, actions_batch, rewards_batch, next_states_batch, dones_batch)

    def reset_experience(self):
        self.replay_buffer.reset()

    def store_experience(self, state_transition: StateTransition):
        # store to RAM to preserve GPU memory
        state_transition.state = state_transition.state.to("cpu")
        state_transition.next_state = state_transition.next_state.to("cpu")
        self.replay_buffer.store(state_transition)

    def train(self):
        # set the two networks to training mode
        self.q_function.train()
        self.target_q_function.train()

        # load checkpoint if any
        checkpoint_info = self.checkpoint_load()
        if not checkpoint_info:
            num_done_episodes = 0
            train_loss = 0
            total_reward_buffer = 0
            total_steps = 0
            test_total_reward_buffer = 0
            episode_rewards = np.asarray([0 for _ in range(self.env.num_envs)], dtype=np.float64)
            episode_reward = 0
            test_episode_reward = 0
            num_test_episodes = 0
            reward_buffer = deque([], maxlen=self.rew_buf_size)
            test_reward_buffer = deque([], maxlen=self.rew_buf_size)

            # initialize replay buffer to specified capacity by following the agent policy and setting the q_function
            # network to eval mode to avoid any training; at the same time
            self.initialize_experience()
        else:
            num_done_episodes, num_test_episodes, train_loss, total_reward_buffer, eps, total_steps, \
            test_total_reward_buffer, episode_rewards, episode_reward, test_episode_reward, reward_buffer, \
            test_reward_buffer = checkpoint_info
            self.eps = eps

        self.testing_env.step_id += total_steps

        # compute the average reward
        average_episode_reward = float((total_reward_buffer / num_done_episodes) if num_done_episodes > 0 else 0)
        test_average_episode_reward = float(
            (test_total_reward_buffer / num_test_episodes) if num_test_episodes > 0 else 0)

        # compute the buffered average reward
        buf_average_reward = float(np.mean(reward_buffer).item() if reward_buffer else 0)
        buf_test_average_reward = float(np.mean(test_reward_buffer).item() if test_reward_buffer else 0)

        # if logging is required, we log the data of the beginning
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
                  f"\t- test_episode_reward: {test_episode_reward}\n")

        # for each episode
        print(f"Training for {max(0, self.num_training_steps - total_steps)} steps...")

        # reset the environment and get the initial state to start a new episode
        previous_states = self.env.reset()
        previous_states = np.asarray(previous_states)
        previous_states = torch.as_tensor(previous_states).to(self.device).float()

        train_pbar = tqdm(total=self.num_training_steps)
        while total_steps < self.num_training_steps:
            # decay eps accordingly to the current number of steps
            self.eps_decay(total_steps)

            # select an action to perform based on the value of eps
            actions = self.get_action(previous_states, train=True)

            # perform the selected action and get the new state
            current_states, rewards, dones, _ = self.env.step(actions)

            # convert the initial state to torch tensor and cast to float tensor
            current_states = np.asarray(current_states)
            current_states = torch.as_tensor(current_states).to(self.device).float()

            # for each environment
            for i in range(self.env.num_envs):
                # get the result of the step for the current environment
                previous_state = previous_states[i]
                action = actions[i]
                reward = rewards[i]
                current_state = current_states[i]
                done = dones[i]

                # store the state transition in the replay buffer
                self.store_experience(
                    state_transition=StateTransition(state=previous_state.unsqueeze(dim=0), action=action,
                                                     reward=reward, next_state=current_state.unsqueeze(dim=0),
                                                     done=done))

            # sample a random minibatch of transitions
            state_transitions_batch = self.sample_experience(num_samples=self.batch_size)

            # do a training step
            train_loss = self.training_step(state_transitions_batch=state_transitions_batch)

            # update the number of total steps
            total_steps += self.env.num_envs
            gradient_descent_steps = total_steps // self.env.num_envs

            # update the progress bar
            train_pbar.update(self.env.num_envs)

            # every C gradient descent steps, we need to reset the target_q_function weights by setting its
            # weights
            # to the weights of the q_function
            self.update_target_network(gradient_descent_steps % self.target_update_steps == 0)

            # add the rewards to the corresponding episode reward list
            episode_rewards += rewards

            # get the episode rewards for agents that are done
            done_episode_rewards = episode_rewards[dones]

            # get the latest episode reward
            episode_reward = done_episode_rewards[0] if len(done_episode_rewards) > 1 else episode_reward

            # append the episode rewards for agents that are done to the reward buffer if any
            total_reward_buffer += np.sum(done_episode_rewards)
            reward_buffer.extend(done_episode_rewards)

            # increment the number of total episodes by the number of done agents
            num_done_episodes += np.count_nonzero(dones)

            # set to 0 the episode rewards for done agents
            episode_rewards *= np.logical_not(dones)

            # compute the training average reward
            average_episode_reward = float((total_reward_buffer / num_done_episodes) if num_done_episodes > 0 else 0)
            buf_average_reward = float(np.mean(reward_buffer).item() if reward_buffer else 0)

            # test the agent every test_every gradient steps
            if gradient_descent_steps % self.test_every == 0:
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

                # compute the test average reward
                test_average_episode_reward = float(
                    (test_total_reward_buffer / num_test_episodes) if num_test_episodes > 0 else 0)
                buf_test_average_reward = float(np.mean(test_reward_buffer).item() if test_reward_buffer else 0)

                print(
                    f"Episodes: {num_done_episodes}, num_test_episodes: {num_test_episodes},"
                    f" eps: {self.eps}, total_ steps: {total_steps}, "
                    f"buffer_samples: {len(self.replay_buffer)}, train_loss: {train_loss}, "
                    f"train_average_episode_reward: {average_episode_reward}, "
                    f"test_average_episode_reward: {test_average_episode_reward}, "
                    f"buffered_train_average_episode_reward: {buf_average_reward}, "
                    f"buffered_test_average_episode_reward: {buf_test_average_reward}, "
                    f"train_episode_reward: {episode_reward}, "
                    f"test_episode_reward: {test_episode_reward}")
            else:
                # compute the test average reward
                test_average_episode_reward = float(
                    (test_total_reward_buffer / num_done_episodes) if num_done_episodes > 0 else 0)
                buf_test_average_reward = float(np.mean(test_reward_buffer).item() if test_reward_buffer else 0)

            # if logging is required, we update it for every training step
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

            # checkpoint the training every checkpoint_every gradient steps
            if gradient_descent_steps % self.checkpoint_every == 0:
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
                                   "test_episode_reward": test_episode_reward}

                filename = f"{self.checkpoint_file}_step_{total_steps}"

                # checkpoint the training
                self.checkpoint_save(filename=filename, checkpoint=checkpoint_info)

            # set previous states to current states
            previous_states = current_states

        train_pbar.close()

        print("Done.")

    def update_target_network(self, update: bool):
        if update:
            self.target_q_function.load_state_dict(self.q_function.state_dict())

    def test(self):
        # set the two networks to eval mode
        self.q_function.eval()
        self.target_q_function.eval()

        with torch.no_grad():
            print(f"Testing...")

            # reset the environment and get the initial state to start a new episode
            previous_state = self.testing_env.reset()

            # convert the initial state to torch tensor, unsqueeze it to feed it as a sample to the network and cast
            # to float tensor
            previous_state = np.asarray(previous_state)
            previous_state = torch.as_tensor(previous_state).to(self.device).unsqueeze(axis=0).float()
            done = False
            test_episode_reward = 0

            # while the episode is not done
            test_pbar = tqdm(total=self.testing_env.spec.max_episode_steps, position=0)
            while not done:
                # select an action to perform based on the agent policy using eps=0 to use only exploitation
                action = self.get_action(previous_state, eps=0, train=False)

                # perform the selected action and get the new state
                current_state, reward, done, _ = self.testing_env.step(action)

                # add the reward to the total reward of the current episode
                test_episode_reward += reward

                # convert the initial state to torch tensor and cast to float tensor
                current_state = np.asarray(current_state)
                current_state = torch.as_tensor(current_state).to(self.device).unsqueeze(axis=0).float()

                # set the next previous state to the current one and unsqueeze it to feed it as a sample to the
                # network
                previous_state = current_state

                test_pbar.update(1)
            test_pbar.close()
        return test_episode_reward

    def save(self, filename: str):
        trained_model_path = f"{self.home_directory}trained_models/"
        if not os.path.isdir(trained_model_path):
            os.makedirs(trained_model_path)
        file_path = f"{trained_model_path}{filename}.pt"

        print(f"Saving trained model to {filename}.pt...")

        # save network weights
        checkpoint = {"model_weights": self.q_function.state_dict()}
        torch.save(checkpoint, file_path)

    def checkpoint_save(self, filename: str, checkpoint: dict):
        checkpoint_path = f"{self.home_directory}trained_models/checkpoints/"
        if not os.path.isdir(checkpoint_path):
            os.makedirs(checkpoint_path)
        file_path = f"{checkpoint_path}{filename}.pt"

        checkpoint['model_weights'] = self.q_function.state_dict()
        checkpoint['optimizer_weights'] = self.optimizer.state_dict()
        checkpoint['replay_buffer'] = self.replay_buffer

        # if in colab, remove old checkpoints to save storage
        folder = glob.glob(f"{checkpoint_path}*")
        if self.save_space:
            for file in folder:
                os.remove(file)

        # save the checkpoint info
        torch.save(checkpoint, file_path)

    def load(self, filename: str) -> None:
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
        checkpoint_path = f"{self.home_directory}trained_models/checkpoints/"
        # if the folder with checkpoints exists and is not empty
        if os.path.isdir(checkpoint_path):
            # define file path as the latest available checkpoint
            sorted_checkpoint_files = natsorted(glob.glob(f'{checkpoint_path}{self.checkpoint_file}*.pt'))

            if len(sorted_checkpoint_files) > 0:
                latest_checkpoint_file = sorted_checkpoint_files[-1]

                print(f"Loading checkpoint from file {os.path.basename(latest_checkpoint_file)}...")

                # load information saved in the file
                checkpoint = torch.load(latest_checkpoint_file)

                # load all checkpoint informations
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
                                test_reward_buffer)

                return return_tuple
            else:
                print("No checkpoint file found. Training is starting from the beginning...")
                return None
        else:
            print("No checkpoint file found. Training is starting from the beginning...")
            return None

    def play(self):
        print("Playing...")
        # set the two networks to eval mode
        self.q_function.eval()
        self.target_q_function.eval()

        # while the episode is not done
        with torch.no_grad():
            test_pbar = tqdm()

            # reset the environment and get the initial state to start a new episode
            previous_state = self.testing_env.reset()
            # convert the initial state to torch tensor, unsqueeze it to feed it as a sample to the network and cast
            # to float tensor
            previous_state = np.asarray(previous_state)
            previous_state = torch.as_tensor(previous_state).to(self.device).unsqueeze(axis=0).float()
            exit = False
            test_episode_reward = 0
            test_total_reward_buffer = 0
            test_reward_buffer = deque([], maxlen=self.rew_buf_size)
            cur_episode = 0
            total_steps = 0
            print(f"Episode {cur_episode + 1}...")
            while not exit:
                try:
                    # select an action to perform based on the agent policy using eps=0 to use only exploitation
                    action = self.get_action(previous_state, eps=0, train=False)

                    # perform the selected action and get the new state
                    current_state, reward, done, info = self.testing_env.step(action)

                    # add the reward to the total reward of the current episode
                    test_episode_reward += reward

                    # convert the initial state to torch tensor and cast to float tensor
                    current_state = np.asarray(current_state)
                    current_state = torch.as_tensor(current_state).to(self.device).unsqueeze(axis=0).float()

                    # compute the test average reward
                    test_average_episode_reward = test_total_reward_buffer / (cur_episode + 1)
                    buf_test_average_reward = float(np.mean(test_reward_buffer).item() if test_reward_buffer else 0)

                    # if logging is required, we update it at the end of every training step
                    if self.logger:
                        self.logger.log("test_average_episode_reward", test_average_episode_reward, total_steps)
                        self.logger.log("total_episodes", cur_episode, total_steps)
                        self.logger.log("test_episode_reward", test_episode_reward, total_steps)
                        self.logger.log("buffered_test_average_episode_reward", buf_test_average_reward, total_steps)

                    if done:
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

                        # reset the environment and set the previous state to the initial state of the environment
                        previous_state = self.env.reset()
                        previous_state = np.asarray(previous_state)
                        previous_state = torch.as_tensor(previous_state).to(self.device).unsqueeze(axis=0).float()
                    else:
                        previous_state = current_state

                        test_pbar.update(1)

                    # increment the number of total steps
                    total_steps += 1

                except (KeyboardInterrupt, SystemExit):
                    print("Playing interrupted, exiting...")
                    exit = True

            test_pbar.close()

    @abstractmethod
    def compute_labels_and_predictions(self, state_transitions_batch: StateTransition):
        pass

    def get_action(self, state, eps=None, train: bool = True) -> torch.Tensor:
        if eps is None:
            eps = self.eps

        # select if to explore or exploit accordingly to epsilon probability
        explore = random.choices([True, False], weights=[eps, 1 - eps], k=1)[0]

        # select if to explore or exploit using eps probability
        if explore:
            # print("random")
            # explore using random selection of actions
            if train:
                action = self.env.action_space.sample()
            else:
                action = self.testing_env.action_space.sample()
            action = torch.as_tensor(np.asarray(action))
        else:
            # print("using policy")
            # exploit the action with the highest value
            q_values = self.q_function(state)
            action = torch.argmax(q_values, dim=1)

        return action

    def eps_decay(self, steps: int):
        new_eps = (steps * ((self.eps_min - self.eps_max) / self.eps_decay_steps)) + self.eps_max
        self.eps = max(new_eps, self.eps_min)

    def compute_loss(self, target_q_values, q_values):
        loss = self.criterion(target_q_values, q_values)
        return loss

    def gradient_descent_step(self, loss, optimizer):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def training_step(self, state_transitions_batch):
        # compute target q values and q values for loss computation accordingly to the agent algorithm
        target_q_values, q_values = self.compute_labels_and_predictions(state_transitions_batch)

        # now we compute the loss between predicted rewards and target rewards
        loss = self.compute_loss(target_q_values, q_values)

        # now we perform a gradient descent step over the parameters of the q_function
        self.gradient_descent_step(loss, self.optimizer)

        return loss.detach().item()


class DQNAgent(TrainableExperienceReplayAgent):
    def __init__(self, discount_rate: float = 0.99, **kwargs):
        super().__init__(**kwargs)

        self.discount_rate = discount_rate

    def compute_labels_and_predictions(self, state_transitions_batch: StateTransition):
        # compute the target q values accordingly to the agent's algorithm
        target_q_values = self.compute_target_q_values(state_transitions_batch)

        # then, we apply a discount rate to the future rewards
        target_q_values = self.discount_rate * target_q_values

        # now, we need to zero the future rewards that correspond to states that are final states; in fact,
        # as said before, future rewards are used in the computations just for current states which are not
        # final states; for final states, the actual reward is just the reward of the current state
        # we can do this by mulitplying a boolean tensor with the tensor of future rewards: this will
        # produce a tensor where the discounted future rewards are zeroed where the boolean tensor is False
        # we need to zero the discounted future rewards for the final states, so the states that are True in
        # the done batch, so we simply mulitply the discounted future rewards tensor by the opposite of the
        # done batch
        target_q_values = target_q_values * torch.logical_not(state_transitions_batch.done)

        # finally, we sum the resulting tensor with the reward batch, resulting thus in a tensor with only
        # the reward for final states and the discounted future reward plus the actual reward for non-final
        # states
        target_q_values = state_transitions_batch.reward + target_q_values

        # we now compute the estimated rewards for the current states using the q_function, but before we
        # zero the gradients of the optimizer
        q_values = self.q_function(state_transitions_batch.state)

        # the previously computed tensor contains the estimated reward for each of the possible action;
        # since we need to compute the loss between these estimated rewards and the target rewards computed
        # before, this latter tensor only contains the future reward with no information about the action,
        # so what we do is computing a one-hot tensor which zeroes the estimated rewards for actions that
        # are not the actual taken actions
        one_hot_actions = torch.nn.functional.one_hot(state_transitions_batch.action, self.env.action_space[0].n)

        # by multiplying the estimated reward tensor and the one hot action tensor, we will get a tensor of
        # the same shape that contains 0 as a reward for actions that are not the current action while
        # contains the estimated reward for the action that is the current action
        q_values *= one_hot_actions

        # we then sum the estimated along the actions dimension to get the final a tensor with only one
        # reward per sample that will be the only reward that was not zeroed out in the previous step
        # (because we will sum zeros with only one reward value)
        q_values = torch.sum(q_values, dim=1)

        return target_q_values, q_values

    def compute_target_q_values(self, state_transitions_batch: StateTransition):
        # compute the labels for the loss computation: if a state is the final state, namely done is true,
        # the label is the reward at that time, otherwise the label is computed as the sum between the
        # reward and the discounted reward at the next state, which in this case is computed by the
        # target_q_function

        # as a first step, we feed the batch of next states to the target network to compute the future
        # rewards, namely the rewards for the next state
        target_q_values = self.target_q_function(state_transitions_batch.next_state)

        # as second step, we get the maximum value for each of the predicted future rewards, so we select
        # the reward corresponding to the action with the highest return
        target_q_values, _ = torch.max(target_q_values, dim=1)

        return target_q_values


class DoubleDQNAgent(DQNAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compute_target_q_values(self, state_transitions_batch: StateTransition):
        # compute the labels for the loss computation: if a state is the final state, namely done is true,
        # the label is the reward at that time, otherwise the label is computed as the sum between the
        # reward and the discounted reward at the next state, which in this case is computed by the
        # target_q_function

        # as a first step, we feed the batch of next states to the q function (q network) to compute the action value
        # estimates
        q_values = self.q_function(state_transitions_batch.next_state)

        # once we did this, we get the actions having the maximum estimated q value
        actions_with_max_value = torch.argmax(q_values, dim=1)

        # now, we feed the batch of next states to the target q function (target q network) to compute the target action
        # value estimates
        target_q_values = self.target_q_function(state_transitions_batch.next_state)

        # now we compute a one-hot tensor that we will use to zero out the target q values of actions that are not the
        # actions with the maximum estimated values
        one_hot_actions = torch.nn.functional.one_hot(actions_with_max_value, self.env.action_space[0].n)

        # by multiplying the estimated reward tensor and the one hot action tensor, we will get a tensor of
        # the same shape that contains 0 as a reward for actions that are not the current action while
        # contains the estimated reward for the action that is the current action
        target_q_values *= one_hot_actions

        # we then sum the estimated along the actions dimension to get the final a tensor with only one
        # reward per sample that will be the only reward that was not zeroed out in the previous step
        # (because we will sum zeros with only one reward value)
        target_q_values = torch.sum(target_q_values, dim=1)

        return target_q_values
