from abc import ABC, abstractmethod
import random

from copy import deepcopy
from typing import Type, Tuple, Callable

import gym
import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm

from src.models import RLNetwork, DQNNetwork
from src.buffers import ReplayBuffer

# TODO documentation

# define a superclass "Agent" for all the possible Agents; this will act as an interface for the other classes
from src.utils import StateTransition


class Agent(ABC):
    @abstractmethod
    def __init__(self, *args) -> None:
        pass

    @abstractmethod
    def get_action(self, *args) -> int:
        pass


# create agent playing by following a learned DQN policy
class DQNAgent(Agent):
    def __init__(self,
                 env: gym.Env,
                 q_function: Type[RLNetwork],
                 device: torch.device,
                 buffer_capacity: int = 10000,
                 num_episodes: int = 100,
                 batch_size: int = 16,
                 eps: float = 0.5) -> None:

        self.env = env
        self.num_episodes = num_episodes
        self.batch_size = batch_size
        self.device = device
        self.eps = eps

        # initialize replay buffer
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)

        # initialize action-value function with random weights
        print("Initializing action-value function with random weights...")
        self.q_function = q_function(input_shape=env.observation_space.shape,
                                     output_channels=env.action_space.n)

        print("Initializing target action-value function with the same weights of the action-value function...")
        self.target_q_function = q_function(input_shape=env.observation_space.shape,
                                            output_channels=env.action_space.n)
        self.target_q_function.load_state_dict(self.q_function.state_dict())

        # initialize replay buffer to specified capacity by following the agent policy and setting the q_function
        # network to eval mode to avoid any training; at the same time, we use torch.no_grad to avoid gradient
        # computation
        self.target_q_function.eval()
        self.q_function.eval()
        with torch.no_grad():
            print(f"Initializing replay buffer with {self.replay_buffer.capacity} samples...")
            self.initialize_experience()

    def initialize_experience(self, n_samples: int = None):
        # if n_samples is not specified we set it to the maximum buffer capacity
        if n_samples is None:
            n_samples = self.replay_buffer.capacity

        # if the replay buffer is has not been filled with the defined amount of samples yes
        full = False
        with tqdm(total=n_samples) as progress_bar:
            while not full:
                # reset the environment and get the initial state to start a new episode
                previous_state = self.env.reset()

                # convert the initial state to np array
                previous_state = np.asarray(previous_state)

                # convert the initial state to torch tensor, unsqueeze it to feed it as a sample to the network and cast
                # to float tensor
                previous_state = torch.as_tensor(previous_state).unsqueeze(axis=0).float()
                done = False

                # while the current episode is not done
                while not done and not full:
                    # select an action to perform based on the agent policy
                    action = self.get_action(previous_state)

                    # perform the selected action and get the new state
                    current_state, reward, done, info = self.env.step(action)

                    # convert the new state to numpy array
                    current_state = np.asarray(current_state)

                    # convert the initial state to torch tensor and cast to float tensor
                    current_state = torch.as_tensor(current_state).float()

                    # squeeze the previous state in order to store it in the buffer
                    previous_state = previous_state.squeeze()

                    # add the state transition to the replay buffer
                    self.store_experience(StateTransition(state=previous_state, action=action, reward=reward,
                                                          next_state=current_state, done=done))
                    progress_bar.update(1)

                    # set the next previous state to the current one and unsqueeze it to feed it as a sample to the
                    # network
                    previous_state = current_state.unsqueeze(axis=0)

                    # if the replay buffer has the specified amount of samples, break the filling operation
                    if len(self.replay_buffer) >= n_samples:
                        full = True

    def sample_experience(self, num_samples: int = 1) -> StateTransition:
        batch = self.replay_buffer.sample(num_samples=num_samples)

        states_batch = [sample.state for sample in batch]
        actions_batch = [sample.action for sample in batch]
        rewards_batch = [sample.reward for sample in batch]
        next_states_batch = [sample.next_state for sample in batch]
        dones_batch = [sample.done for sample in batch]

        states_batch = torch.stack(states_batch)
        actions_batch = torch.tensor(actions_batch)
        rewards_batch = torch.tensor(rewards_batch)
        next_states_batch = torch.stack(next_states_batch)
        dones_batch = torch.tensor(dones_batch)

        return StateTransition(states_batch, actions_batch, rewards_batch, next_states_batch, dones_batch)

    def reset_experience(self):
        self.replay_buffer.reset()

    def store_experience(self, state_transition: StateTransition):
        self.replay_buffer.store(state_transition)

    def train(self):
        # set the two networks to training mode
        self.q_function.train()
        self.target_q_function.train()

        # for each episode
        print(f"Training for {self.num_episodes} episodes...")
        for _ in tqdm(range(self.num_episodes)):
            # reset the environment and get the initial state to start a new episode
            previous_state = self.env.reset()

            # convert the initial state to np array
            previous_state = np.asarray(previous_state)

            # convert the initial state to torch tensor, unsqueeze it to feed it as a sample to the network and cast
            # to float tensor
            previous_state = torch.as_tensor(previous_state).unsqueeze(axis=0).float()
            done = False

            # while the episode is not done
            while not done:
                # select an action to perform based on the agent policy
                action = self.get_action(previous_state)

                # perform the selected action and get the new state
                current_state, reward, done, info = self.env.step(action)

                # convert the new state to numpy array
                current_state = np.asarray(current_state)

                # convert the initial state to torch tensor and cast to float tensor
                current_state = torch.as_tensor(current_state).float()

                # squeeze the previous state in order to store it in the buffer
                previous_state = previous_state.squeeze()

                # store the transition to the replay buffer memory
                state_transition = StateTransition(state=previous_state, action=action, reward=reward,
                                                   next_state=current_state, done=done)
                self.store_experience(state_transition=state_transition)

                # set the next previous state to the current one and unsqueeze it to feed it as a sample to the
                # network
                previous_state = current_state.unsqueeze(axis=0)

                # sample a random minibatch of transitions
                state_transitions_batch = self.sample_experience(num_samples=self.batch_size)

    def loss(self) -> None:
        pass

    def get_action(self, state) -> int:
        # select if to explore or exploit accordingly to epsilon probability
        explore = random.choices([True, False], weights=[self.eps, 1 - self.eps], k=1)[0]

        # select if to explore or exploit using eps probability
        if explore:
            # explore using random selection of actions (random policy)
            action = random.randint(0, self.env.action_space.n - 1)
        else:
            # exploit the action with the highest value (greedy policy)
            q_values = self.q_function(state)
            action = int(torch.argmax(q_values))

        return action
