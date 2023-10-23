import time

import gym
import imageio
import numpy as np
import myosuite
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from envs.environment_factory import EnvironmentFactory

def inference():
    env = DummyVecEnv([lambda: EnvironmentFactory.create('CustomMyoRelocateP2')])
    # env = DummyVecEnv([lambda: gym.make("myoChallengeRelocateP2-v0")])

    env.envs[0].mj_render()
    # time.sleep(5)

    frames = []
    for e in range(1):
        print(e)
        state = env.reset()
        done = False
        score = 0
        while not done:
            frame = env.envs[0].render('rgb_array')
            frames.append(frame)
            env.step(env.action_space.sample().reshape(1, -1))
            env.envs[0].mj_render()
            time.sleep(0.1)
        print(score)
    env.close()

    path = 'eval.mp4'
    writer = imageio.get_writer(path, fps=25)
    for i in frames:
        writer.append_data(i)
    writer.close()


if __name__ == '__main__':
    inference()
