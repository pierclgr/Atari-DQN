from abc import ABC, abstractmethod
import random

from copy import deepcopy
from src.logger import Logger
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
                 buffer_capacity: int = 1000000,
                 num_episodes: int = 100,
                 batch_size: int = 32,
                 eps_max: float = 1,
                 eps_min: float = 0.01,
                 eps_decay_steps: float = 1000000,
                 discount_rate: float = 0.90,
                 target_update_steps: int = 500,
                 learning_rate: float = 0.00025,
                 criterion: nn.Module = None,
                 optimizer: torch.optim.Optimizer = None,
                 logger: Logger = None) -> None:

        self.env = env
        self.num_episodes = num_episodes
        self.batch_size = batch_size
        self.device = device
        self.eps = eps_max
        self.eps_max = eps_max
        self.eps_min = eps_min
        self.eps_decay_steps = eps_decay_steps
        self.discount_rate = discount_rate
        self.target_update_steps = target_update_steps
        self.learning_rate = learning_rate

        self.logger = logger

        # initialize replay buffer
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)

        # initialize action-value function with random weights
        print("Initializing action-value function with random weights...")
        self.q_function = q_function(input_shape=env.observation_space.shape,
                                     output_channels=env.action_space.n).to(device=self.device)

        print("Initializing target action-value function with the same weights of the action-value function...")
        self.target_q_function = q_function(input_shape=env.observation_space.shape,
                                            output_channels=env.action_space.n).to(device=self.device)
        self.target_q_function.load_state_dict(self.q_function.state_dict())

        if not criterion:
            criterion = nn.MSELoss()

        self.criterion = criterion

        if not optimizer:
            optimizer = torch.optim.RMSprop(self.q_function.parameters(), lr=self.learning_rate)

        self.optimizer = optimizer

        # initialize replay buffer to specified capacity by following the agent policy and setting the q_function
        # network to eval mode to avoid any training; at the same time, we use torch.no_grad to avoid gradient
        # computation
        print(f"Initializing replay buffer with {self.replay_buffer.capacity} samples...")
        self.initialize_experience()

    def initialize_experience(self, n_samples: int = None):
        self.q_function.eval()
        self.target_q_function.eval()
        with torch.no_grad():
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

                    # convert the initial state to torch tensor, unsqueeze it to feed it as a sample to the network and
                    # cast to float tensor
                    previous_state = torch.as_tensor(previous_state).unsqueeze(axis=0).float().to(self.device)
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
                        current_state = torch.as_tensor(current_state).float().to(self.device)

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

        states_batch = torch.stack(states_batch).to(self.device)
        actions_batch = torch.as_tensor(actions_batch).to(self.device)
        rewards_batch = torch.as_tensor(rewards_batch).to(self.device)
        next_states_batch = torch.stack(next_states_batch).to(self.device)
        dones_batch = torch.as_tensor(dones_batch).to(self.device)

        return StateTransition(states_batch, actions_batch, rewards_batch, next_states_batch, dones_batch)

    def reset_experience(self):
        self.replay_buffer.reset()

    def store_experience(self, state_transition: StateTransition):
        self.replay_buffer.store(state_transition)

    def train(self):
        # set the two networks to training mode
        self.q_function.train()
        self.target_q_function.eval()

        # for each episode
        print(f"Training for {self.num_episodes} episodes...")
        num_steps = 0
        episodes_rewards = []
        total_reward = 0
        for _ in tqdm(range(self.num_episodes)):
            # append total reward to the list of episodes rewards
            episodes_rewards.append(total_reward)

            # set total reward for the new episode to 0
            total_reward = 0

            # reset the environment and get the initial state to start a new episode
            previous_state = self.env.reset()

            # convert the initial state to np array
            previous_state = np.asarray(previous_state)

            # convert the initial state to torch tensor, unsqueeze it to feed it as a sample to the network and cast
            # to float tensor
            previous_state = torch.as_tensor(previous_state).unsqueeze(axis=0).float().to(self.device)
            done = False

            # while the episode is not done
            while not done:
                # decay eps accordingly to the current number of steps
                self.eps_decay(num_steps)

                # select an action to perform based on the agent policy
                self.q_function.eval()
                action = self.get_action(previous_state)
                self.q_function.train()

                # perform the selected action and get the new state
                current_state, reward, done, info = self.env.step(action)

                # add the reward to the total reward
                total_reward += reward

                # convert the new state to numpy array
                current_state = np.asarray(current_state)

                # convert the initial state to torch tensor and cast to float tensor
                current_state = torch.as_tensor(current_state).float().to(self.device)

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

                # compute the labels for the loss computation: if a state is the final state, namely done is true, the
                # label is the reward at that time, otherwise the label is computed as the sum between the reward and
                # the discounted reward at the next state, which in this case is computed by the target_q_function

                # as a first step, we feed the batch of next states to the target network to compute the future rewards,
                # namely the rewards for the next state
                with torch.no_grad():
                    target_q_values = self.target_q_function(state_transitions_batch.next_state)

                # as second step, we get the maximum value for each of the predicted future rewards, so we select the
                # reward corresponding to the action with the highest return
                target_q_values, _ = torch.max(target_q_values, dim=1)

                # then, we apply a discount rate to the future rewards
                target_q_values = self.discount_rate * target_q_values

                # now, we need to zero the future rewards that correspond to states that are final states; in fact, as
                # said before, future rewards are used in the computations just for current states which are not final
                # states; for final states, the actual reward is just the reward of the current state
                # we can do this by mulitplying a boolean tensor with the tensor of future rewards: this will produce a
                # tensor where the discounted future rewards are zeroed where the boolean tensor is False
                # we need to zero the discounted future rewards for the final states, so the states that are True in
                # the done batch, so we simply mulitply the discounted future rewards tensor by the opposite of the
                # done batch
                target_q_values = target_q_values * torch.logical_not(state_transitions_batch.done)

                # finally, we sum the resulting tensor with the reward batch, resulting thus in a tensor with only the
                # reward for final states and the discounted future reward plus the actual reward for non-final states
                target_q_values = state_transitions_batch.reward + target_q_values

                # we now compute the estimated rewards for the current states using the q_function, but before we zero
                # the gradients of the optimizer
                self.optimizer.zero_grad()
                q_values = self.q_function(state_transitions_batch.state)

                # the previously computed tensor contains the estimated reward for each of the possible action; since
                # we need to compute the loss between these estimated rewards and the target rewards computed before,
                # this latter tensor only contains the future reward with no information about the action, so what we do
                # is computing a one-hot tensor which zeroes the estimated rewards for actions that are not the actual
                # taken actions
                one_hot_actions = torch.nn.functional.one_hot(state_transitions_batch.action, self.env.action_space.n)

                # by multiplying the estimated reward tensor and the one hot action tensor, we will get a tensor of the
                # same shape that contains 0 as a reward for actions that are not the current action while contains the
                # estimated reward for the action that is the current action
                q_values *= one_hot_actions

                # we then sum the estimated along the actions dimension to get the final a tensor with only one reward
                # per sample that will be the only reward that was not zeroed out in the previous step (because we will
                # sum zeros with only one reward value)
                q_values = torch.sum(q_values, dim=1)

                # now we compute the loss between predicted rewards and target rewards and perform a gradient descent
                # step over the parameters of the q_funciton
                loss = self.criterion(target_q_values, q_values)
                loss.backward()
                self.optimizer.step()

                num_steps += 1

                # if logging is required, we update it at every training step
                if self.logger:
                    self.logger.log("train_loss", loss.detach().item(), num_steps)
                    self.logger.log("eps", self.eps, num_steps)
                    self.logger.log("average_reward", float(np.mean(episodes_rewards)), num_steps)

                # every C gradient descent steps, we need to reset the target_q_function weights by setting its weights
                # to the weights of the q_function
                if num_steps % self.target_update_steps == 0:
                    self.target_q_function.load_state_dict(self.q_function.state_dict())

    def evaluate(self):
        pass

    def get_action(self, state) -> int:
        # select if to explore or exploit accordingly to epsilon probability
        explore = random.choices([True, False], weights=[self.eps, 1 - self.eps], k=1)[0]

        # select if to explore or exploit using eps probability
        if explore:
            # explore using random selection of actions (random policy)
            action = self.env.action_space.sample()
        else:
            # exploit the action with the highest value (greedy policy)
            with torch.no_grad():
                q_values = self.q_function(state)
            action = int(torch.argmax(q_values))

        return action

    def eps_decay(self, steps: int):
        new_eps = (steps * ((self.eps_min - self.eps_max) / self.eps_decay_steps)) + self.eps_max
        self.eps = max(new_eps, self.eps_min)
