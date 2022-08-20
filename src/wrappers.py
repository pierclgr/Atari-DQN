import os
from collections import deque
from multiprocessing import Process, Pipe
from typing import Callable, Optional, Union, List

from baselines.common.vec_env import VecEnv, CloudpickleWrapper
from gym.core import RenderFrame
from gym.vector.utils import spaces
from gym.wrappers import TimeLimit
import gym
import numpy as np
import cv2
import gym

from baselines.common.atari_wrappers import EpisodicLifeEnv, FireResetEnv, ClipRewardEnv
from gym.vector.utils import batch_space


class MaxAndSkipEnvCustom(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2: self._obs_buffer[0] = obs
            if i == self._skip - 1: self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class FrameStackCustom(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.

        Returns lazy array, which is much more memory efficient.

        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=((shp[0] * k,) + shp[1:]),
                                                dtype=env.observation_space.dtype)

    def reset(self, **kwargs):
        ob = self.env.reset(**kwargs)
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFramesCustom(list(self.frames), axis=0)


class LazyFramesCustom(object):
    def __init__(self, frames, axis=-1):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.

        This object should only be converted to numpy array before being passed to the model.

        You'd not believe how complex the previous solution was."""
        self._frames = frames
        self._out = None
        self._axis = axis

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=self._axis)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]


class NoopResetEnvCustom(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.integers(1, self.noop_max + 1)  # pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


class WarpFrameCustom(gym.ObservationWrapper):
    def __init__(self, env, width: int = 84, height: int = 84, grayscale: bool = True):
        """Warp frames to 84x84 as done in the Nature paper and later work."""
        gym.ObservationWrapper.__init__(self, env)
        self.width = width
        self.height = height
        self.grayscale = grayscale
        self.channels = 1 if self.grayscale else 3
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(self.height, self.width, self.channels), dtype=np.uint8)

    def observation(self, frame):
        if self.grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame[:, :, None]


class ImageTransposeWrapper(gym.ObservationWrapper):
    def observation(self, observation):
        return self.transpose(observation)

    def __init__(self, env):
        super().__init__(env)
        obs_shape = (self.env.observation_space.shape[2],
                     self.env.observation_space.shape[0],
                     self.env.observation_space.shape[1])
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=obs_shape,
                                                dtype=self.env.observation_space.dtype)

    def transpose(self, observation):
        return np.transpose(observation, (2, 0, 1))


class ReproducibleEnv(gym.Wrapper):
    def __init__(self, env, seed):
        gym.Wrapper.__init__(self, env)
        self.seed = seed

    def reset(self, **kwargs):
        self.env.action_space.seed(seed=self.seed)
        ob = self.env.reset(seed=self.seed, **kwargs)
        return ob


class ScaledFloatFrameCustom(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=env.observation_space.shape, dtype=np.float32)

    def observation(self, observation):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        return np.array(observation).astype(np.float32) / 255.0


def deepmind_atari_wrappers(env, max_episode_steps: int = None, noop_max: int = 30, frame_skip: int = 4,
                            episode_life: bool = True, clip_rewards: bool = True, frame_stack: int = 4,
                            scale: bool = True, patch_size: int = 84, grayscale: bool = True,
                            fire_reset: bool = True):
    if noop_max > 0:
        env = NoopResetEnvCustom(env, noop_max=noop_max)
    if frame_skip > 0:
        env = MaxAndSkipEnvCustom(env, skip=frame_skip)
    if max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
    if episode_life:
        env = EpisodicLifeEnv(env)
    if 'FIRE' in env.unwrapped.get_action_meanings() and fire_reset:
        env = FireResetEnv(env)
    if patch_size is not None:
        env = WarpFrameCustom(env, width=patch_size, height=patch_size, grayscale=grayscale)
    if scale:
        env = ScaledFloatFrameCustom(env)
    if clip_rewards:
        env = ClipRewardEnv(env)

    env = ImageTransposeWrapper(env)

    if frame_stack > 0:
        env = FrameStackCustom(env, frame_stack)

    return env


class VectorEnv(gym.Env):
    def __init__(self, environment_maker, num_envs):
        self.envs = [environment_maker() for _ in range(num_envs)]
        self.num_envs = num_envs

        self.observation_space = batch_space(self.envs[0].observation_space, n=num_envs)
        self.action_space = batch_space(self.envs[0].action_space, n=num_envs)

    def seed(self, seed: Optional[int] = None):
        seeds = [seed + i for i in range(self.num_envs)]
        return [env.seed(s) for env, s in zip(self.envs, seeds)]

    def reset(self, **kwargs):
        if "seed" in kwargs.keys():
            seed = kwargs["seed"]
            kwargs = [{**kwargs, 'seed': seed + i} for i in range(self.num_envs)]
        else:
            kwargs = [kwargs for _ in range(self.num_envs)]

        assert len(kwargs) == len(self.envs)

        return [env.reset(**args) for env, args in zip(self.envs, kwargs)]

    def step(self, actions):
        assert len(self.envs) == len(actions)
        observations = []
        rewards = []
        dones = []
        infos = []
        for env, a in zip(self.envs, actions):
            observation, reward, done, info = env.step(a)
            if done:
                observation = env.reset()

            observations.append(observation)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
        return observations, rewards, dones, infos

    def close(self):
        for env in self.envs:
            env.close()

    def render(self, mode="human") -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        pass


def atari_deepmind_env(env_name, max_episode_steps: int = None, noop_max: int = 30, frame_skip: int = 4,
                       episode_life: bool = True, clip_rewards: bool = True, frame_stack: int = 4,
                       scale: bool = True, patch_size: int = 84, grayscale: bool = True,
                       fire_reset: bool = True, render_mode: str = None):
    env = gym.make(env_name, obs_type="rgb", render_mode=render_mode, new_step_api=False)
    env = deepmind_atari_wrappers(env, max_episode_steps=max_episode_steps, noop_max=noop_max, frame_skip=frame_skip,
                                  episode_life=episode_life, clip_rewards=clip_rewards, frame_stack=frame_stack,
                                  scale=scale, patch_size=patch_size, grayscale=grayscale, fire_reset=fire_reset)
    return env


def vector_atari_deepmind_env(env_name, num_envs: int, max_episode_steps: int = None,
                              noop_max: int = 30, frame_skip: int = 4,
                              episode_life: bool = True, clip_rewards: bool = True, frame_stack: int = 4,
                              scale: bool = True, patch_size: int = 84, grayscale: bool = True,
                              fire_reset: bool = True, render_mode: str = None):
    make_atari_deepmind_env = lambda: atari_deepmind_env(env_name=env_name, max_episode_steps=max_episode_steps,
                                                         noop_max=noop_max,
                                                         frame_skip=frame_skip, episode_life=episode_life,
                                                         clip_rewards=clip_rewards,
                                                         frame_stack=frame_stack, scale=scale,
                                                         patch_size=patch_size,
                                                         grayscale=grayscale,
                                                         fire_reset=fire_reset, render_mode=render_mode)

    env = VectorEnv(environment_maker=make_atari_deepmind_env, num_envs=num_envs)

    return env
