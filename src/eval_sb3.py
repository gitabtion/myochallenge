import os.path
import time
import gym
import imageio
import numpy as np
# import myosuite
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from tqdm import tqdm

from envs.environment_factory import EnvironmentFactory
from src.definitions import ROOT_DIR


def inference():
    config = {
        "obs_keys": ['hand_qpos', 'hand_qvel', 'obj_pos', 'goal_pos', 'pos_err', 'obj_rot', 'goal_rot',
                     'rot_err'],
        "weighted_reward_keys": {
            "pos_dist": 10.0,
            "rot_dist": 0,
            "act_reg": 0.01,
            # "solved": 10.,
            "drop": -1.,
            # "sparse": 10.0,
            "keep_time": -200.,
            # "reach_dist": 4,
            "norm_solved": 10.,
        },
        'normalize_act': True,
        'frame_skip': 5,
        'pos_th': 0.1,  # cover entire base of the receptacle
        'rot_th': np.inf,  # ignore rotation errors
        'qpos_noise_range': 0.01,  # jnt initialization range
        'target_xyz_range': {'high': [0.3, -.1, 1.05], 'low': [0.0, -.45, 0.9]},
        'target_rxryrz_range': {'high': [0.2, 0.2, 0.2], 'low': [-.2, -.2, -.2]},
        'obj_xyz_range': {'high': [0.1, -.15, 1.0], 'low': [-0.1, -.35, 1.0]},
        'obj_geom_range': {'high': [.025, .025, .025], 'low': [.015, 0.015, 0.015]},
        'obj_mass_range': {'high': 0.200, 'low': 0.050},  # 50gms to 200 gms
        'obj_friction_range': {'high': [1.2, 0.006, 0.00012], 'low': [0.8, 0.004, 0.00008]}
    }

    # env = DummyVecEnv([lambda: EnvironmentFactory.create('CustomMyoRelocateP2', **config)])
    env = DummyVecEnv([lambda: gym.make("myoChallengeRelocateP2-v0")])

    norm_env = VecNormalize.load(
        os.path.join(ROOT_DIR, 'trained_models/step6/env.pkl'),
        env)
    norm_env.norm_obs = True
    norm_env.training = False

    policy = PPO.load(os.path.join(ROOT_DIR, 'trained_models/step6/rl_models.zip'))
    # env.envs[0].mj_render()
    # time.sleep(5)
    score = 0
    for e in tqdm(range(1000)):
        state = env.reset()
        done = False
        while not done:
            norm_obs = norm_env.normalize_obs(state)
            action, _ = policy.predict(norm_obs)
            state, reward, done, info = env.step(action)
            # time.sleep(0.5)
        if info[0]['solved']:
            score += 1
            print(score)
    print(f'sr: {score/1000}')
    env.close()


if __name__ == '__main__':
    inference()
