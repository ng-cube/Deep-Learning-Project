import gymnasium
from gymnasium import Wrapper
import numpy as np 



def transform_reward(reward):
    return np.clip(reward, -1.0, 1.0)


def create_atari_environment(
    env_name: str,
    seed: int=1,
    frame_skip: int=4,
    frame_stack: int=32,
    screen_height: int=96,
    screen_width: int=96,
    noop_max: int=30,
    max_episode_steps: int=108000,
    terminal_on_life_loss: bool=False,
    clip_reward: bool=False,
    episodic_life: bool=False,
):
    env = gymnasium.make(f"{env_name}NoFrameskip-v4")
    env.seed(seed)

    # set episode step limit
    env = gymnasium.wrappers.TimeLimit(env=env, max_episode_steps=max_episode_steps)

    # apply remaining wrappers
    env = gymnasium.wrappers.AtariPreprocessing(
        env=env,
        noop_max=noop_max,
        frame_skip=frame_skip,
        terminal_on_life_loss=terminal_on_life_loss,
        grayscale_obs=True,
        scale_obs=True,
        screen_size=screen_height,
    )
    env = gymnasium.wrappers.FrameStack(
        env=env, 
        num_stack=frame_stack
    )
    if clip_reward:
        env = gymnasium.wrappers.TransformReward(
            env=env,
            f=transform_reward
        )
    """env = gymnasium.wrappers.TransformReward(
        env=env,
        f=lambda reward: np.clip(reward, -1.0, 1.0) if clip_reward else reward
    )"""
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireReset(env=env)

    return env




# Custom wrappers
class FireReset(Wrapper):
    """ Take fire action on reset for environments like Breakout. """
    def __init__(self, env):
        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, term, trunc, _ = self.env.step(1)
        done = term or trunc
        if done:
            self.env.reset(**kwargs)
        obs, _, term, trunc, _ = self.env.step(2)
        done = term or trunc
        if done:
            self.env.reset(**kwargs)
        return np.asarray(obs)

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        done = term or trunc
        return np.asarray(obs), reward, term, trunc, info